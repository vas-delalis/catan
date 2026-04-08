mod data;
mod model;
mod training;

pub use model::{Model, ModelEvaluator, create_model};
pub use training::{TrainingConfig, train};
