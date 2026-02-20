mod data;
mod model;
mod training;

pub use data::TicTacToeBatcher;
pub use model::{Model, ModelConfig};
pub use training::{TrainingConfig, train};
