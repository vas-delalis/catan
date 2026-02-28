use burn::{
    data::dataloader::batcher::Batcher,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use crate::{agents::Evaluator, games::TicTacToe, ml::TicTacToeBatcher};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([3, 8], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([3, 3]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 3 * 3, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, 1).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, planes, height, width]
    ///   - Output [batch_size]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch_size, _, _, _] = images.dims();

        // Create a channel at the second dimension.
        let x = images; //.reshape([batch_size, 3, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 3, 3]
        let x = x.reshape([batch_size, 16 * 3 * 3]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x).reshape([-1]) // [batch_size]
    }
}

pub struct ModelEvaluator<B: Backend> {
    pub model: Model<B>,
    pub batcher: TicTacToeBatcher,
    pub device: B::Device,
}

impl<B: Backend> Evaluator<TicTacToe> for ModelEvaluator<B> {
    fn evaluate(&self, game_state: TicTacToe) -> f64 {
        let batch = self.batcher.batch(vec![(game_state, None)], &self.device);
        let output = self.model.forward(batch.images);
        output.into_scalar().to_f64()
    }
}
