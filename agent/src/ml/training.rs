use burn::{
    backend::LibTorch,
    data::dataloader::DataLoaderBuilder,
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{Transaction, backend::AutodiffBackend},
    train::{
        InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{Adaptor, LossInput, LossMetric},
    },
};

use crate::ml::{
    data::{TicTacToeBatch, TicTacToeBatcher, TicTacToeDataset, selfplay},
    model::{Model, ModelConfig},
};

pub struct RegressionOutput1d<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 1>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> RegressionOutput1d<B> {
    fn new(loss: Tensor<B, 1>, output: Tensor<B, 1>, targets: Tensor<B, 1>) -> Self {
        RegressionOutput1d {
            loss,
            output,
            targets,
        }
    }
}

impl<B: Backend> ItemLazy for RegressionOutput1d<B> {
    type ItemSync = RegressionOutput1d<LibTorch>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        RegressionOutput1d {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for RegressionOutput1d<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

// impl<B: Autodiff<Backend>> Adaptor<LossInput<B>> for RegressionOutput1d<B> {
//     fn adapt(&self) -> LossInput<B> {
//         LossInput::new(self.loss.clone())
//     }
// }

impl<B: Backend> Model<B> {
    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1>,
    ) -> RegressionOutput1d<B> {
        let output = self.forward(images);
        let loss =
            MseLoss::new().forward(output.clone(), targets.clone(), nn::loss::Reduction::Auto);

        RegressionOutput1d::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = TicTacToeBatch<B>;
    type Output = RegressionOutput1d<B>;

    fn step(&self, batch: TicTacToeBatch<B>) -> TrainOutput<RegressionOutput1d<B>> {
        let item = self.forward_regression(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = TicTacToeBatch<B>;
    type Output = RegressionOutput1d<B>;

    fn step(&self, batch: TicTacToeBatch<B>) -> RegressionOutput1d<B> {
        self.forward_regression(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device)
// where
//     RegressionOutput1d<<B as AutodiffBackend>::InnerBackend>: Adaptor<LossInput<B>>,
{
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = TicTacToeBatcher::default();
    // let items = vec![(TicTacToeDataset {}).get(0).unwrap()];
    // let batch = <TicTacToeBatcher as Batcher<B, TicTacToeItem, TicTacToeBatch<B>>>::batch(
    //     &batcher, items, &device,
    // );
    // dbg!(batch.images.shape());

    let mut dataset_train = TicTacToeDataset::new();
    selfplay(&mut dataset_train);

    let mut dataset_test = TicTacToeDataset::new();
    selfplay(&mut dataset_test);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
