mod data;
mod model;
mod training;

pub use model::{Model, vanilla};
use serde::{Deserialize, Serialize};
use tch::Tensor;
pub use training::train;

use crate::GameState;

pub trait Image: GameState {
    const IMAGE_SIZE: i64;
    fn image(&self, arbiter: Self::Player) -> Tensor;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model_config: ModelConfig,
    pub hyperparameters: Hyperparameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub game: String,
    pub layers: usize,
    pub hidden_dim: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub epochs: usize,
    pub learning_rate: f64,
    pub train_replays: usize,
    pub test_replays: usize,
    pub batch_size: usize,
    pub search_evals: usize,
    pub dirichlet_alpha: f64,
}
