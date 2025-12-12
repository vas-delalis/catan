use std::collections::{HashMap, VecDeque};

use rand::{rng, seq::index::sample_weighted};
use rand_distr::{Distribution, Gamma};

use crate::{Action, Agent, GameState, Player};

pub struct Node<A: Action, P: Player> {
    // id: String,
    visits: u16,
    children: HashMap<A, Box<Node<A, P>>>,
    to_play: Option<P>,
    prior: f64, // TODO: smaller? quantized?
    total_value: f64,
}

impl<A: Action, P: Player> Node<A, P> {
    pub fn new(prior: f64) -> Self {
        Node {
            // id,
            prior,
            to_play: None,
            children: HashMap::new(),
            visits: 0,
            total_value: 0.0,
        }
    }

    pub fn value(&self) -> f64 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_value / (self.visits as f64)
    }
}

pub struct Search<A: Action, P: Player> {
    pb_c_base: f64,
    pb_c_init: f64,
    dirichlet_alpha: f64,
    // dirichlet alpha:
    // branching factor * alpha = average actions observed per node
    // alphazero have chosen a number around 10 for that last quantity,
    // so for chess, with its branching factor of ~35, we get alpha = .3
    max_evals: usize,
    history: Vec<(P, A)>,
    value: f64,
}

impl<A: Action, P: Player> Search<A, P> {
    pub fn new(
        max_evals: usize,
        pb_c_base: f64,
        pb_c_init: f64,
        dirichlet_alpha: f64,
        value: f64,
    ) -> Self {
        Search {
            pb_c_base,
            pb_c_init,
            dirichlet_alpha,
            max_evals,
            history: vec![],
            value,
        }
    }

    fn score(&self, parent: &Node<A, P>, child: &Node<A, P>) -> f64 {
        let mut pb_c =
            ((parent.visits as f64 + self.pb_c_base + 1.0) / self.pb_c_base).ln() + self.pb_c_init;
        pb_c *= f64::sqrt(parent.visits as f64) / (child.visits as f64 + 1.0);

        let prior_score = pb_c * child.prior;
        let value_score = child.value();
        prior_score + value_score
    }

    fn evaluate<'a, G>(&self, node: &'a mut Box<Node<A, P>>, game: &G) -> f64
    where
        G: GameState<A, P>,
    {
        let to_play = game.current_player();
        node.to_play = Some(to_play);
        if let Some((_, value)) = game.terminal_value(to_play) {
            return value;
        }
        let actions = game.get_actions(to_play);

        // Run inference
        let (value, policy_logits) = (self.value, vec![0f64; actions.len()]);
        let policy: Vec<f64> = policy_logits.into_iter().map(|l| l.exp()).collect();
        let sum: f64 = policy.iter().sum();

        // Expand node (initialize children)
        for (action, p) in actions.iter().zip(policy) {
            let child = Node::new(p / sum);
            node.children.insert(*action, Box::new(child));
        }
        value
    }

    pub fn run<G: GameState<A, P>>(&self, mut game_state: G) -> A {
        let mut root = Box::new(Node::new(0.0));
        self.evaluate(&mut root, &mut game_state);
        self.add_exploration_noise(&mut root);

        for _ in 0..self.max_evals {
            let mut node = &mut root;
            let mut scratch = game_state.clone();
            let mut search_path: Vec<A> = vec![];

            while !node.children.is_empty() {
                let (action, _) = self.select_child(node);
                node = node.children.get_mut(&action).unwrap();
                scratch.apply_action(action);
                search_path.push(action);
            }

            let value = self.evaluate(node, &scratch);

            // Backpropagate
            root.total_value += if root.to_play.unwrap() == scratch.current_player() {
                1.0 - value
            } else {
                value
            };
            root.visits += 1;
            let mut node = &mut root;
            for a in search_path {
                node = node.children.get_mut(&a).unwrap();
                node.total_value += if node.to_play.unwrap() == scratch.current_player() {
                    1.0 - value
                } else {
                    value
                };
                node.visits += 1;
            }
        }
        let mut q: VecDeque<(Option<A>, &Node<A, P>, usize)> = VecDeque::new();
        q.push_back((None, &root, 0));

        while let Some((action, node, level)) = q.pop_front() {
            if level > 2 {
                continue;
            }
            // print!("{}", &String::from(" ").repeat(level));
            // println!("{:?} {} {:.1}", action, node.visits, node.value());
            for (&action, child) in node.children.iter() {
                q.push_back((Some(action), child, level + 1));
            }
        }

        return self.select_action(&root);
    }

    fn select_action(&self, root: &Node<A, P>) -> A {
        let visit_counts = root.children.iter().map(|(&a, v)| (v.visits, a)).collect();
        // TODO: parameterize
        if self.history.len() < 100 {
            return softmax_sample(visit_counts);
        }
        visit_counts.iter().max_by_key(|(c, _)| c).unwrap().1
    }

    fn select_child<'a>(&self, node: &'a Node<A, P>) -> (A, &'a Box<Node<A, P>>) {
        let (&action, child) = node
            .children
            .iter()
            .max_by(|(_, a), (_, b)| self.score(node, a).total_cmp(&self.score(node, b)))
            .unwrap();

        (action, child)
    }

    fn add_exploration_noise(&self, node: &mut Node<A, P>) {
        let mut rng = rng();
        let gamma = Gamma::new(self.dirichlet_alpha, 0.5).unwrap();
        let fraction = 0.25; // TODO: parameterize
        for (_, child) in node.children.iter_mut() {
            child.prior *= 1.0 - fraction;
            child.prior += gamma.sample(&mut rng) * fraction;
        }
    }
}

impl<A: Action, P: Player, G: GameState<A, P>> Agent<A, P, G> for Search<A, P> {
    fn get_action(&self, game_state: G) -> A {
        self.run(game_state)
    }
}

fn softmax_sample<A: Copy>(vec: Vec<(u16, A)>) -> A {
    let max_exponent = vec.iter().map(|(e, _)| *e).max().unwrap() as f64;
    let exponentials = vec
        .iter()
        // Subtract largest exponent to avoid huge values ("safe softmax")
        .map(|(n, _)| f64::exp(*n as f64 - max_exponent));
    let sum: f64 = exponentials.clone().sum();
    let probabilities: Vec<f64> = exponentials.map(|x| x / sum).collect();

    let mut rng = rng();
    let index = sample_weighted(&mut rng, probabilities.len(), |i| probabilities[i], 1)
        .unwrap()
        .index(0);
    vec[index].1
}
