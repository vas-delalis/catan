mod human;
mod mcts;
mod random;

pub use human::Human;
pub use mcts::{Evaluator, Search};
pub use random::Random;

use crate::GameState;

pub struct ConstantEvaluator {}
impl<G: GameState> Evaluator<G> for ConstantEvaluator {
    fn evaluate(&self, _: G) -> f64 {
        0.0
    }
}
