use std::path::PathBuf;
use std::time::Duration;
use std::{fs, time::Instant};

use crate::ml::TrainingConfig;
use crate::{
    Tournament,
    agents::{ConstantEvaluator, Search},
    ml::{Model, data::Dataset, vanilla},
};
use common::{GameState, Image, Player};
use itertools::Itertools;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

pub fn train<G: GameState + Image + Send>(config: TrainingConfig) {
    let model_config = config.model_config.clone();
    let params = config.hyperparameters.clone();

    let arch = vanilla::<G>(model_config.layers, model_config.hidden_dim);
    let model = Model::new::<G>(arch);
    let mut optimizer = nn::Adam::default()
        .build(model.var_store(), params.learning_rate)
        .unwrap();

    let agent = Search::new(
        &model,
        params.search_evals,
        false,
        1.41,
        1.0,
        params.dirichlet_alpha,
    );
    let reference_agent = Search::new(
        ConstantEvaluator::new(0.0),
        params.search_evals,
        false,
        1.41,
        1.0,
        params.dirichlet_alpha,
    );

    let mut tournament: Tournament<G> = Tournament::new(0.05, 0.05)
        .max_moves(1000)
        .max_time(Duration::from_secs(10));
    tournament.add(Box::new(agent.clone()), "agent", true);
    tournament.add(Box::new(reference_agent.clone()), "reference", true);
    for _ in 2..G::Player::LEN {
        // Fill remaining slots
        tournament.add(Box::new(reference_agent.clone()), "reference", false);
    }
    tournament.play();
    tournament.leaderboard();

    println!("Training {} parameters...", model.parameter_count());

    let save_path = get_save_path(&G::name());

    let mut dataset_train = Dataset::new(params.train_replays);
    let mut dataset_test = Dataset::new(params.test_replays);
    let start = Instant::now();

    for epoch in 1..=params.epochs {
        dataset_train.self_play(&agent, params.self_play_sampling_rate, default_threads());
        dataset_test.self_play(&agent, params.self_play_sampling_rate, default_threads());

        // Train
        let mut train_loss = 0.0;
        let n_states = dataset_train.len();
        for batch in &dataset_train.drain().chunks(params.batch_size) {
            let mut images = Vec::with_capacity(params.batch_size * G::Player::LEN);
            let mut targets = Vec::with_capacity(params.batch_size * G::Player::LEN);
            for (state, values) in batch {
                for (&arbiter, &value) in G::Player::list().iter().zip(values.iter()) {
                    let x = state.image(arbiter);
                    let y = Tensor::from(value);
                    images.push(x);
                    targets.push(y);
                }
            }
            let images = Tensor::stack(&images, 1).transpose(0, 1);
            let targets = Tensor::stack(&targets, 0).reshape([-1, 1]);

            let output = model.infer(images);

            let loss = output.mse_loss(&targets, tch::Reduction::Sum);
            optimizer.backward_step(&loss);
            let loss: f32 = loss.try_into().unwrap();
            train_loss += loss;
        }

        print!(
            "[Epoch {}] Train: {:.3} / ",
            epoch,
            train_loss / n_states as f32 / G::Player::LEN as f32
        );

        // Test
        let _no_grad = tch::no_grad_guard(); // Turn off gradient computation
        let mut test_losses = vec![];
        for (state, values) in dataset_test.drain() {
            for (&arbiter, &value) in G::Player::list().iter().zip(values.iter()) {
                let x = state.image(arbiter);
                let y = Tensor::from(value);
                let loss = model.infer(x).mse_loss(&y, tch::Reduction::Sum);
                let loss: f32 = loss.try_into().unwrap();
                test_losses.push(loss);
            }
        }

        let n = test_losses.len();
        let sum: f32 = test_losses.iter().sum();
        println!("Test: {:.3}", sum / n as f32);

        // Save periodically
        if epoch % 100 == 0 && epoch < params.epochs {
            let checkpoint_path = save_path.with_file_name(format!(
                "{}-{}.safetensors",
                save_path.file_stem().unwrap().to_string_lossy(),
                epoch,
            ));
            model.var_store().save(&checkpoint_path).unwrap();
        }
    }

    println!("Training complete. Elapsed: {}s", start.elapsed().as_secs());

    let mut tournament = Tournament::new(0.05, 0.05)
        .max_moves(1000)
        .max_time(Duration::from_secs(10));
    tournament.add(Box::new(agent.clone()), "agent", true);
    tournament.add(Box::new(reference_agent.clone()), "reference", true);
    for _ in 2..G::Player::LEN {
        // Fill remaining slots
        tournament.add(Box::new(reference_agent.clone()), "reference", false);
    }
    tournament.play();
    tournament.leaderboard();

    model.save_with_config(&save_path, &config).unwrap();
    println!(
        "Saved as {}",
        &save_path.file_name().unwrap().to_string_lossy()
    );
}

fn get_save_path(game_name: &str) -> PathBuf {
    let mut path = PathBuf::from("./models");
    path.push(game_name);
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

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
