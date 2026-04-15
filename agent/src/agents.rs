mod human;
mod mcts;
mod random;

pub use human::Human;
pub use mcts::{Evaluator, Search};
pub use random::Random;

use crate::GameState;

#[derive(Clone)]
pub struct ConstantEvaluator {
    pub c: f64,
}
impl<G: GameState> Evaluator<G> for ConstantEvaluator {
    fn evaluate(&self, _: G) -> f64 {
        self.c
    }
}
