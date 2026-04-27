use std::collections::HashMap;
use std::marker::PhantomData;

use rand::{rng, seq::index::sample_weighted};
use rand_distr::{Distribution, Gamma};

use crate::{Agent, GameState};

pub trait Evaluator<G: GameState> {
    fn evaluate(&self, game_state: G) -> f64;
}

impl<G: GameState, E: Evaluator<G>> Evaluator<G> for &E {
    fn evaluate(&self, game_state: G) -> f64 {
        (**self).evaluate(game_state)
    }
}

pub struct Node<G: GameState> {
    visits: u16,
    children: HashMap<G::Action, Box<Node<G>>>,
    to_play: Option<G::Player>,
    played: Option<G::Player>,
    prior: f64, // TODO: smaller? quantized?
    total_value: f64,
}

impl<G: GameState> Node<G> {
    pub fn new(prior: f64) -> Self {
        Node {
            prior,
            to_play: None,
            played: None,
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

#[derive(Clone)]
pub struct Search<G: GameState, E: Evaluator<G>> {
    greedy: bool,
    pb_c_base: f64,
    pb_c_init: f64,
    dirichlet_alpha: f64,
    // dirichlet alpha:
    // branching factor * alpha = average actions observed per node
    // alphazero have chosen a number around 10 for that last quantity,
    // so for chess, with its branching factor of ~35, we get alpha = .3
    max_evals: usize,
    evaluator: E,
    _phantom: PhantomData<G>,
}

impl<G: GameState, E: Evaluator<G>> Search<G, E> {
    pub fn new(
        evaluator: E,
        max_evals: usize,
        greedy: bool,
        pb_c_base: f64,
        pb_c_init: f64,
        dirichlet_alpha: f64,
    ) -> Self {
        Search {
            greedy,
            evaluator,
            max_evals,
            pb_c_base,
            pb_c_init,
            dirichlet_alpha,
            _phantom: PhantomData,
        }
    }

    fn score(&self, parent: &Node<G>, child: &Node<G>) -> f64 {
        let mut pb_c =
            ((parent.visits as f64 + self.pb_c_base + 1.0) / self.pb_c_base).ln() + self.pb_c_init;
        pb_c *= f64::sqrt(parent.visits as f64) / (child.visits as f64 + 1.0);

        let prior_score = pb_c * child.prior;
        let value_score = child.value();
        prior_score + value_score
    }

    fn evaluate<'a>(&self, node: &'a mut Box<Node<G>>, game: &G) -> f64 {
        let to_play = game.current_player();
        node.to_play = Some(to_play);
        if let Some((_, value)) = game.outcome(node.played.unwrap()) {
            return value;
        }
        let actions = game.get_actions(to_play);

        let value = self.evaluator.evaluate(game.clone());

        // Run inference
        let (_, policy_logits) = (value, vec![0f64; actions.len()]);
        let policy: Vec<f64> = policy_logits.into_iter().map(|l| l.exp()).collect();
        let sum: f64 = policy.iter().sum();

        // Expand node (initialize children)
        for (action, p) in actions.iter().zip(policy) {
            let child = Node::new(p / sum);
            node.children.insert(*action, Box::new(child));
        }
        value
    }

    pub fn run(&self, mut game_state: G) -> G::Action {
        let mut root = Box::new(Node::new(0.0));
        root.played = Some(game_state.current_player());
        self.evaluate(&mut root, &mut game_state);
        self.add_exploration_noise(&mut root);

        for _ in 0..self.max_evals {
            let mut node = &mut root;
            let mut scratch = game_state.clone();
            let mut search_path: Vec<G::Action> = vec![];
            let mut prev_player = node.to_play;

            while !node.children.is_empty() {
                let (action, _) = self.select_child(node);
                prev_player = node.to_play;
                node = node.children.get_mut(&action).unwrap();
                node.played = prev_player;
                scratch.apply_action(action);
                search_path.push(action);
            }

            // The correct value is the value from the perspective of the previous player.
            let value = self.evaluate(node, &scratch);

            // Backpropagate
            root.total_value += if root.played == prev_player {
                value
            } else {
                -value
            };
            root.visits += 1;
            let mut node = &mut root;
            for a in search_path {
                node = node.children.get_mut(&a).unwrap();
                node.total_value += if node.played == prev_player {
                    value
                } else {
                    -value
                };
                node.visits += 1;
            }
        }
        // let mut q: Vec<(Option<G::Action>, &Node<G>, usize)> = Vec::new();
        // q.push((None, &root, 0));
        // while let Some((action, node, level)) = q.pop() {
        //     if level > 6 || node.visits == 0 {
        //         continue;
        //     }
        //     print!("{}", &String::from(" ").repeat(level));
        //     println!(
        //         "{:?} {:?} {} {:.1}{}",
        //         node.played,
        //         action,
        //         node.visits,
        //         node.value(),
        //         if node.children.is_empty() { "!" } else { "" }
        //     );
        //     for (&action, child) in node.children.iter() {
        //         q.push((Some(action), child, level + 1));
        //     }
        // }

        return self.select_action(&root, self.greedy);
    }

    fn select_action(&self, root: &Node<G>, greedy: bool) -> G::Action {
        let visit_counts: Vec<(u16, <G as GameState>::Action)> =
            root.children.iter().map(|(&a, v)| (v.visits, a)).collect();
        if greedy {
            return visit_counts.iter().max_by_key(|(c, _)| c).unwrap().1;
        }
        softmax_sample(visit_counts)
    }

    fn select_child<'a>(&self, node: &'a Node<G>) -> (G::Action, &'a Box<Node<G>>) {
        let (&action, child) = node
            .children
            .iter()
            .max_by(|(_, a), (_, b)| self.score(node, a).total_cmp(&self.score(node, b)))
            .unwrap();

        (action, child)
    }

    fn add_exploration_noise(&self, node: &mut Node<G>) {
        let mut rng = rng();
        let gamma = Gamma::new(self.dirichlet_alpha, 1.0).unwrap();
        let fraction = 0.25; // TODO: parameterize
        for (_, child) in node.children.iter_mut() {
            child.prior *= 1.0 - fraction;
            child.prior += gamma.sample(&mut rng) * fraction;
        }
    }
}

impl<G: GameState, E: Evaluator<G>> Agent<G> for Search<G, E> {
    fn get_action(&self, game_state: G) -> G::Action {
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
