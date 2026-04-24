mod data;
mod model;
mod training;

pub use model::{Model, two_layers};
use tch::Tensor;
pub use training::{TrainingConfig, train};

use crate::GameState;

pub trait Batch: GameState {
    const BATCH_DIM: i64;
    fn batch(&self) -> Tensor;
}
