use std::collections::HashMap;

use rand::{rng, seq::index::sample_weighted};
use rand_distr::{Distribution, Gamma};

use crate::{Agent, GameState};

pub trait Evaluator<G: GameState> {
    fn evaluate(&self, game_state: &G, arbiter: G::Player) -> f32;
}

impl<G: GameState, E: Evaluator<G>> Evaluator<G> for &E {
    fn evaluate(&self, game_state: &G, arbiter: G::Player) -> f32 {
        (**self).evaluate(game_state, arbiter)
    }
}

#[derive(Clone)]
pub struct Node<G: GameState> {
    visits: u16,
    children: HashMap<G::Action, Box<Node<G>>>,
    prior: f64, // TODO: smaller? quantized?
    total_value: f64,
}

impl<G: GameState> Node<G> {
    pub fn new(prior: f64) -> Self {
        Node {
            prior,
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
    tree: std::cell::RefCell<Option<(G, Box<Node<G>>)>>,
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
            tree: std::cell::RefCell::new(None),
        }
    }

    fn score(&self, parent: &Node<G>, child: &Node<G>) -> f32 {
        let prior_score = {
            let mut pb_c =
                ((parent.visits as f32 + self.pb_c_base as f32 + 1.0) / self.pb_c_base as f32).ln()
                    + self.pb_c_init as f32;
            pb_c *= f32::sqrt(parent.visits as f32) / (child.visits as f32 + 1.0);
            pb_c * child.prior as f32
        };
        let value_score = child.value();
        prior_score + value_score as f32
    }

    fn evaluate<'a>(&self, node: &'a mut Box<Node<G>>, game: &G, arbiter: G::Player) -> f64 {
        let to_play = game.current_player();
        if let Some((_, value)) = game.outcome(arbiter) {
            return value as f64;
        }
        let actions = game.get_actions(to_play);

        // Run inference
        let value = self.evaluator.evaluate(game, arbiter);
        let (_, policy_logits) = (value, vec![0f64; actions.len()]);
        let policy: Vec<f64> = policy_logits.into_iter().map(|l| l.exp()).collect();
        let sum: f64 = policy.iter().sum();

        // Expand node (initialize children)
        node.children.reserve(actions.len());
        for (action, p) in actions.iter().zip(policy) {
            let child = Node::new(p / sum);
            node.children.insert(*action, Box::new(child));
        }
        value as f64
    }

    pub fn run(&self) -> G::Action {
        let (game_state, mut root) = self.tree.borrow_mut().take().unwrap();
        let arbiter = game_state.current_player();

        if root.children.is_empty() {
            self.evaluate(&mut root, &game_state, arbiter);
            self.add_exploration_noise(&mut root);
        }

        for _ in 0..self.max_evals {
            let mut node = &mut root;
            let mut scratch = game_state.clone();
            let mut search_path: Vec<G::Action> = vec![];

            while !node.children.is_empty() {
                let action = self.select_child(node);
                node = node.children.get_mut(&action).unwrap();
                scratch.apply_action(action);
                search_path.push(action);
            }

            let value = self.evaluate(node, &scratch, arbiter);

            // Backpropagate
            root.total_value += value;
            root.visits += 1;
            let mut node = &mut root;
            for a in search_path {
                node = node.children.get_mut(&a).unwrap();
                node.total_value += value;
                node.visits += 1;
            }
        }

        let action = self.select_action(&root, self.greedy);
        *self.tree.borrow_mut() = Some((game_state, root));

        action
    }

    fn select_action(&self, root: &Node<G>, greedy: bool) -> G::Action {
        let visit_counts: Vec<(u16, <G as GameState>::Action)> =
            root.children.iter().map(|(&a, v)| (v.visits, a)).collect();
        if greedy {
            return visit_counts.iter().max_by_key(|(c, _)| c).unwrap().1;
        }
        softmax_sample(visit_counts)
    }

    fn select_child<'a>(&self, node: &'a Node<G>) -> G::Action {
        let (&action, _) = node
            .children
            .iter()
            .max_by(|(_, a), (_, b)| self.score(node, a).total_cmp(&self.score(node, b)))
            .unwrap();
        action
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
        if self.tree.borrow().is_none() {
            *self.tree.borrow_mut() = Some((game_state, Box::new(Node::new(0.0))));
        }
        self.run()
    }

    fn inform(&self, action: G::Action) {
        let mut tree_opt = self.tree.borrow_mut();
        if let Some((mut state, mut node)) = tree_opt.take() {
            state.apply_action(action);
            *tree_opt = if let Some(child) = node.children.remove(&action) {
                Some((state, child))
            } else {
                Some((state, Box::new(Node::new(0.0))))
            }
        }
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
