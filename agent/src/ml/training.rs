use std::path::PathBuf;
use std::{fs, time::Instant};

use crate::ml::model::{Metadata, ModelConfig};
use crate::{
    GameState, Player, Tournament,
    agents::{ConstantEvaluator, Search},
    games::DotsAndBoxes,
    ml::{Image, Model, data::Dataset, vanilla},
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

type GAME = DotsAndBoxes;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub train_replays: usize,
    pub test_replays: usize,
    pub batch_size: usize,
    pub search_evals: usize,
    pub dirichlet_alpha: f64,
}

pub fn train(config: TrainingConfig) {
    let hidden_dim = 16;
    let layers = 2;
    let arch = vanilla::<GAME>(layers, hidden_dim);
    let model = Model::new::<GAME>(arch);
    let mut optimizer = nn::Adam::default()
        .build(model.var_store(), config.learning_rate)
        .unwrap();

    let agent = Search::new(
        &model,
        config.search_evals,
        false,
        1.41,
        1.0,
        config.dirichlet_alpha,
    );
    let reference_agent = Search::new(
        ConstantEvaluator::new(0.0),
        config.search_evals,
        false,
        1.41,
        1.0,
        config.dirichlet_alpha,
    );

    let mut tournament: Tournament<GAME> = Tournament::new(0.25, 0.25);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    println!("Training {} parameters...", model.parameter_count());
    let start = Instant::now();

    for epoch in 1..=config.epochs {
        let mut dataset_train = Dataset::new(config.train_replays);
        let mut dataset_test = Dataset::new(config.test_replays);
        dataset_train.selfplay(&agent);
        dataset_test.selfplay(&agent);

        // Train
        let mut train_loss = 0.0;
        let n_states = dataset_train.len();
        for batch in &dataset_train.into_iter().chunks(config.batch_size) {
            let mut images = Vec::with_capacity(config.batch_size);
            let mut targets = Vec::with_capacity(config.batch_size);
            for (state, values) in batch {
                let x = state.image();
                let y = Tensor::from_slice(&values);
                images.push(x);
                targets.push(y);
            }
            let images = Tensor::stack(&images, 1).transpose(0, 1);
            let targets = Tensor::stack(&targets, 1).transpose(0, 1);
            let output = model.infer(images);
            let loss = output.mse_loss(&targets, tch::Reduction::Sum);

            optimizer.backward_step(&loss);
            let loss: f32 = loss.try_into().unwrap();
            train_loss += loss;
        }

        print!(
            "[Epoch {}] Train: {:.3} / ",
            epoch,
            train_loss / n_states as f32 / <GAME as GameState>::Player::LEN as f32
        );

        // Test
        let _no_grad = tch::no_grad_guard(); // Turn off gradient computation
        let mut test_losses = vec![];
        for (state, values) in dataset_test {
            let x = state.image();
            let y = Tensor::from_slice(&values);
            let loss = model.infer(x).mse_loss(&y, tch::Reduction::Mean);
            let loss: f32 = loss.try_into().unwrap();
            test_losses.push(loss);
        }

        let n = test_losses.len();
        let sum: f32 = test_losses.iter().sum();
        println!("Test: {:.3}", sum / n as f32);
    }

    println!("Training complete. Elapsed: {}s", start.elapsed().as_secs());

    let mut tournament: Tournament<GAME> = Tournament::new(0.05, 0.05);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    let path = get_save_path();
    model
        .save_with_metadata(
            &path,
            &Metadata {
                model_config: ModelConfig {
                    game: GAME::name(),
                    layers,
                    hidden_dim,
                },
                training_config: config,
            },
        )
        .unwrap();
    println!("Saved as {}", &path.file_name().unwrap().to_string_lossy());
}

fn get_save_path() -> PathBuf {
    let mut path = PathBuf::from("./models");
    path.push(GAME::name());
    fs::create_dir_all(&path).unwrap();

    // Get highest id in directory
    let prev = fs::read_dir(&path)
        .unwrap()
        .filter_map(|e| {
            e.unwrap()
                .path()
                .file_prefix()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<usize>()
                .ok()
        })
        .max();
    let next = match prev {
        Some(x) => x + 1,
        None => 0,
    };
    path.push(format!("{}.safetensors", next));
    path
}
