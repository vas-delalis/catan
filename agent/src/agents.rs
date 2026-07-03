mod human;
mod mcts;
mod random;

use common::Evaluation;
use generic_array::sequence::GenericSequence;
pub use human::Human;
pub use mcts::{Evaluator, Search};
pub use random::Random;

use crate::GameState;

#[derive(Clone)]
pub struct ConstantEvaluator {
    pub constant: f32,
}
impl ConstantEvaluator {
    pub fn new(constant: f32) -> Self {
        ConstantEvaluator { constant }
    }
}
impl<G: GameState> Evaluator<G> for ConstantEvaluator {
    fn evaluate(&self, _: &G) -> Evaluation<G> {
        Evaluation::<G>::repeat(self.constant)
    }
}
