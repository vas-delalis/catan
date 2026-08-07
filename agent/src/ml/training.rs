use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use crate::ml::TrainingConfig;
use crate::ml::{
    Hyperparameters, Model, ModelConfig, ModelMetadata, data::Dataset, quantization::CLAMP_LIMIT,
    vanilla,
};
use crate::{
    INTERRUPTED, Tournament,
    agents::{ConstantEvaluator, Search},
    boxed,
};
use common::{GameState, Image, Player};
use itertools::Itertools;
use tch::{
    Tensor,
    nn::{self, Optimizer, OptimizerConfig},
};

pub struct Trainer<G: GameState + Image> {
    pub model: Model<G>,
    pub model_id: usize,
    model_config: ModelConfig,
    pub params: Hyperparameters,
    optimizer: Optimizer,
    dataset_train: Dataset<G>,
    dataset_test: Dataset<G>,
    pub epoch: usize,
    loss: Vec<f32>,
    norm: Vec<f32>,
    time: Duration,
}

impl<G: GameState + Image> Trainer<G> {
    pub fn new(config: TrainingConfig) -> Self {
        let model_config = config.model_config.clone();
        let params = config.hyperparameters.clone();

        let arch = vanilla::<G>(model_config.layers, model_config.hidden_dim);
        let model = Model::new(arch);

        let mut optimizer = nn::AdamW::default()
            .build(model.var_store(), params.learning_rate)
            .unwrap();
        optimizer.set_weight_decay(params.weight_decay);

        let dataset_train = Dataset::new(params.train_replays);
        let dataset_test = Dataset::new(params.test_replays);

        Self {
            loss: Vec::with_capacity(params.max_epochs),
            norm: Vec::with_capacity(params.max_epochs),
            model_config,
            params,
            model,
            optimizer,
            dataset_train,
            dataset_test,
            epoch: 1,
            model_id: Model::<G>::get_next_id(),
            time: Duration::ZERO,
        }
    }

    pub fn run_tournament(&self) {
        let evals = self.params.search_evals;
        let alpha = self.params.dirichlet_alpha;
        let model = &self.model;
        let agent_factory = || boxed(Search::new(model, evals, false, 1.41, 1.0, alpha));
        let reference_factory = || {
            boxed(Search::new(
                ConstantEvaluator::new(0.0),
                evals,
                false,
                1.41,
                1.0,
                alpha,
            ))
        };

        let mut tournament: Tournament<G> = Tournament::new(0.05, 0.05)
            .max_moves(1000)
            .max_time(Duration::from_secs(10));
        tournament.add(agent_factory, "agent", true);
        tournament.add(reference_factory, "reference", true);
        for _ in 2..G::Player::LEN {
            // Fill remaining slots
            tournament.add(reference_factory, "reference", false);
        }
        tournament.play();
        tournament.leaderboard();
    }

    pub fn run_epoch(&mut self) {
        let start = Instant::now();
        let params = &self.params;

        self.dataset_train
            .self_play(&self.model, &params, default_threads());
        self.dataset_test
            .self_play(&self.model, &params, default_threads());

        // Train
        let mut train_loss = 0.0;
        let n_iterations = self.dataset_train.len().div_ceil(params.batch_size);

        for batch in &self.dataset_train.drain().chunks(params.batch_size) {
            let mut images = Vec::with_capacity(params.batch_size * G::Player::LEN);
            let mut targets = Vec::with_capacity(params.batch_size * G::Player::LEN);
            for (state, values) in batch {
                let x = state.tensor_image();
                let y = Tensor::from_slice(&values);
                images.push(x);
                targets.push(y);
            }
            let images = Tensor::stack(&images, 1).transpose(0, 1);
            let targets = Tensor::stack(&targets, 0);

            let output = self.model.infer(images);

            let loss = output.mse_loss(&targets, tch::Reduction::Mean);
            self.optimizer.backward_step(&loss);
            {
                // Clamp weights for later quantization
                let _no_grad = tch::no_grad_guard();
                for p in &mut self.model.var_store().trainable_variables() {
                    let _ = p.clamp_(-CLAMP_LIMIT, CLAMP_LIMIT);
                }
            }

            let loss: f32 = loss.try_into().unwrap();
            train_loss += loss;
        }

        let epoch_avg_loss = train_loss / n_iterations as f32;
        self.loss.push(epoch_avg_loss);
        print!("[Epoch {}] Train: {:.3} / ", self.epoch, epoch_avg_loss);

        // Test
        let _no_grad = tch::no_grad_guard(); // Turn off gradient computation
        let mut test_losses = vec![];
        for (state, values) in self.dataset_test.drain() {
            let x = state.tensor_image();
            let y = Tensor::from_slice(&values);
            let loss = self.model.infer(x).mse_loss(&y, tch::Reduction::Mean);
            let loss: f32 = loss.try_into().unwrap();
            test_losses.push(loss);
        }

        let n = test_losses.len();
        let sum: f32 = test_losses.iter().sum();
        println!("Test: {:.3}", sum / n as f32);

        self.time += start.elapsed();

        let mut norm = 0.0;
        for p in self.model.var_store().trainable_variables() {
            let flat = p.flatten(0, -1);
            let x: f32 = flat.linalg_norm(2, 0, false, None).try_into().unwrap();
            norm += x.powi(2);
        }
        let norm = norm.sqrt();
        self.norm.push(norm);

        // Save periodically
        if self.epoch % 100 == 0 && self.epoch < params.max_epochs {
            let name = format!("{}-{}", self.model_id, self.epoch);
            self.model.save(&name).unwrap();
        }

        // Save and quit when interrupted
        if INTERRUPTED.read() {
            println!(
                "\r\x1b[KInterrupted at epoch {}. Saving checkpoint...",
                self.epoch
            );
            let (checkpoint_path, _) = self
                .model
                .save_with_metadata(&self.model_id.to_string(), self.metadata())
                .unwrap();
            println!("Saved at {}", checkpoint_path.to_string_lossy());
            return;
        }

        self.epoch += 1;
    }

    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            model_config: self.model_config.clone(),
            hyperparameters: self.params.clone(),
            training_data: super::TrainingData {
                epochs: self.epoch,
                epoch_loss: self.loss.clone(),
                duration_secs: self.time.as_secs_f64(),
            },
        }
    }

    pub fn save_model(&self) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
        self.model
            .save_with_metadata(&self.model_id.to_string(), self.metadata())
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
