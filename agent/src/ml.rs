mod data;
mod model;
mod quantization;
mod training;

pub use model::{Model, vanilla};
pub use quantization::{ACTIVATION_SCALE, QuantizedEvaluator, allocate_aligned_slice};
use serde::{Deserialize, Serialize};
pub use training::train;

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
    pub search_evals: usize,
    pub batch_size: usize,
    pub train_replays: usize,
    pub test_replays: usize,
    pub learning_rate: f64,
    pub dirichlet_alpha: f64,
    pub self_play_sampling_rate: f64,
    pub self_play_random_action_chance: f64,
    pub weight_decay: f64,
}
