mod data;
mod model;
mod training;

pub use model::{Model, vanilla};
use tch::Tensor;
pub use training::{TrainingConfig, train};

pub trait Image {
    const IMAGE_SIZE: i64;
    fn image(&self) -> Tensor;
}
