use agent::ml::{ModelConfig, TrainingConfig, train};
use burn::{
    backend::{Autodiff, LibTorch, libtorch::LibTorchDevice},
    optim::AdamConfig,
};

fn main() {
    type B = LibTorch<f32>;
    type AB = Autodiff<B>;

    let device = LibTorchDevice::Cpu;
    let artifact_dir = "./model";
    let model_config = ModelConfig::new(128);

    train::<AB>(
        artifact_dir,
        TrainingConfig::new(model_config, AdamConfig::new()),
        device.clone(),
    );
}
