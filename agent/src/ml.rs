mod data;
mod model;
mod training;

pub use data::TicTacToeBatcher;
pub use model::ModelConfig;
pub use training::{TrainingConfig, train};
