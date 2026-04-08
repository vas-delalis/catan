mod data;
mod model;
mod training;

pub use model::{Model, ModelEvaluator, create_model};
use tch::Tensor;
pub use training::{TrainingConfig, train};

use crate::GameState;

pub trait Batch: GameState {
    fn batch(&self) -> Tensor;
}
