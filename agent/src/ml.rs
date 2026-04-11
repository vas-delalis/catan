mod data;
mod model;
mod training;

pub use model::{Model, ModelEvaluator, create_model};
use tch::Tensor;
pub use training::{TrainingConfig, train};

use crate::GameState;

pub trait Batch: GameState {
    const BATCH_DIM: i64;
    fn batch(&self) -> Tensor;
}
