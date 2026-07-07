use nohash_hasher::IntMap;
use rand::{rng, seq::index::sample_weighted};
use rand_distr::{Distribution, Gamma};
use std::cell::RefCell;

use crate::{Agent, GameState};
use common::Evaluation;

pub trait Evaluator<G: GameState> {
    fn evaluate(&self, game_state: &G) -> Evaluation<G>;
}

impl<G: GameState, E: Evaluator<G>> Evaluator<G> for &E {
    fn evaluate(&self, game_state: &G) -> Evaluation<G> {
        (**self).evaluate(game_state)
    }
}

#[derive(Clone)]
pub struct Node {
    visits: u16,
    children: IntMap<usize, Node>,
    prior: f64, // TODO: smaller? quantized?
    total_value: f64,
}

impl Node {
    pub fn new(prior: f64) -> Self {
        Node {
            prior,
            children: IntMap::default(),
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

    fn display<G: GameState>(&self, depth: usize, max_depth: usize, action: Option<String>) {
        if depth > max_depth {
            return;
        }

        let indent = "  ".repeat(depth);
        if let Some(action_str) = action {
            println!(
                "{}{} {} | P: {:.2} | V: {:.2}",
                indent,
                action_str,
                self.visits,
                self.prior,
                self.value()
            );
        } else {
            println!(
                "{}Root {} | P: {:.2} | V: {:.2}",
                indent,
                self.visits,
                self.prior,
                self.value()
            );
        }

        for (&action_id, child) in &self.children {
            let action_obj = G::Action::from(action_id);
            Self::display::<G>(
                child,
                depth + 1,
                max_depth,
                Some(format!("{:?}", action_obj)),
            );
        }
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
    tree: RefCell<Option<(G, Node)>>,
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
            tree: RefCell::new(None),
        }
    }

    fn evaluate(&self, node: &mut Node, game: &G) -> Evaluation<G> {
        let to_play = game.current_player();
        if game.is_terminal() {
            return game.scores().unwrap();
        }
        let (actions, probs) = game.get_actions(to_play);

        // Run inference
        let values = self.evaluator.evaluate(game);
        // Convert priors into odds
        let policy: Vec<f64> = if let Some(probs) = probs {
            probs.into_iter().map(|p| p / (1.0 - p)).collect()
        } else {
            let logits = vec![0f64; actions.len()];
            logits.into_iter().map(|l| l.exp()).collect()
        };
        let sum: f64 = policy.iter().sum();

        // Expand node (initialize children)
        node.children.reserve(actions.len());
        for (&action, p) in actions.iter().zip(policy) {
            let child = Node::new(p / sum);
            node.children.insert(action.into(), child);
        }
        values
    }

    pub fn run(&self) -> G::Action {
        let (game_state, mut root) = self.tree.borrow_mut().take().unwrap();

        if root.children.is_empty() {
            self.evaluate(&mut root, &game_state);
            self.add_exploration_noise(&mut root);
        }

        let mut search_path: Vec<(G::Player, G::Action)> = vec![];
        for _ in 0..self.max_evals {
            let mut node = &mut root;
            let mut scratch = game_state.clone();
            search_path.clear();

            while !node.children.is_empty() {
                let action = self.select_child(node);
                node = node.children.get_mut(&action.into()).unwrap();
                search_path.push((scratch.current_player(), action));
                scratch.apply_action(action);
            }

            let values = self.evaluate(node, &scratch);

            // Backpropagate
            root.total_value += values[game_state.current_player().into()] as f64;
            root.visits += 1;
            let mut node = &mut root;
            for &(played, a) in &search_path {
                node = node.children.get_mut(&a.into()).unwrap();
                node.total_value += values[played.into()] as f64;
                node.visits += 1;
            }
        }

        let action = self.select_action(&root, self.greedy);
        *self.tree.borrow_mut() = Some((game_state, root));

        action
    }

    fn select_action(&self, root: &Node, greedy: bool) -> G::Action {
        let visit_counts: Vec<(u16, <G as GameState>::Action)> = root
            .children
            .iter()
            .map(|(&a, v)| (v.visits, G::Action::from(a)))
            .collect();
        if greedy {
            return visit_counts.iter().max_by_key(|(c, _)| c).unwrap().1;
        }
        softmax_sample(visit_counts)
    }

    fn select_child(&self, node: &Node) -> G::Action {
        let pb_c_base = self.pb_c_base as f32;
        let pb_c_init = self.pb_c_init as f32;
        let parent_visits = node.visits as f32;

        let pb_c = ((parent_visits + pb_c_base + 1.0) / pb_c_base).ln() + pb_c_init;
        let pb_c = pb_c * parent_visits.sqrt();

        let (&action, _) = node
            .children
            .iter()
            .map(|(action, child)| {
                let prior_score = pb_c * (child.prior as f32) / (child.visits as f32 + 1.0);
                let score = prior_score + child.value() as f32;
                (action, score)
            })
            .max_by(|(_, a_score), (_, b_score)| a_score.total_cmp(b_score))
            .unwrap();

        G::Action::from(action)
    }

    fn add_exploration_noise(&self, node: &mut Node) {
        let mut rng = rng();
        let gamma = Gamma::new(self.dirichlet_alpha, 1.0).unwrap();
        let fraction = 0.25; // TODO: parameterize
        for child in node.children.values_mut() {
            child.prior *= 1.0 - fraction;
            child.prior += gamma.sample(&mut rng) * fraction;
        }
    }

    pub fn display_tree(&self, max_depth: usize) {
        if let Some((_, ref root)) = *self.tree.borrow() {
            root.display::<G>(0, max_depth, None);
        }
    }
}

impl<G: GameState, E: Evaluator<G>> Agent<G> for Search<G, E> {
    fn get_action(&self, game_state: G) -> G::Action {
        if self.tree.borrow().is_none() {
            *self.tree.borrow_mut() = Some((game_state, Node::new(0.0)));
        }
        self.run()
    }

    fn inform(&self, action: G::Action) {
        let mut tree_opt = self.tree.borrow_mut();
        if let Some((mut state, mut node)) = tree_opt.take() {
            state.apply_action(action);
            *tree_opt = if let Some(child) = node.children.remove(&action.into()) {
                Some((state, child))
            } else {
                Some((state, Node::new(0.0)))
            }
        }
    }

    fn reset(&self) {
        self.tree.borrow_mut().take();
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
