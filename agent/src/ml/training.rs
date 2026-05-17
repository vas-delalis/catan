use std::path::PathBuf;
use std::{fs, time::Instant};

use crate::ml::TrainingConfig;
use crate::{
    GameState, Player, Tournament,
    agents::{ConstantEvaluator, Search},
    games::DotsAndBoxes,
    ml::{Image, Model, data::Dataset, vanilla},
};
use itertools::Itertools;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

type GAME = DotsAndBoxes;

pub fn train(config: TrainingConfig) {
    let model_config = config.model_config.clone();
    let params = config.hyperparameters.clone();

    let arch = vanilla::<GAME>(model_config.layers, model_config.hidden_dim);
    let model = Model::new::<GAME>(arch);
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

    let mut tournament: Tournament<GAME> = Tournament::new(0.05, 0.05);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    println!("Training {} parameters...", model.parameter_count());
    let start = Instant::now();

    for epoch in 1..=params.epochs {
        let mut dataset_train = Dataset::new(params.train_replays);
        let mut dataset_test = Dataset::new(params.test_replays);
        dataset_train.selfplay(&agent);
        dataset_test.selfplay(&agent);

        // Train
        let mut train_loss = 0.0;
        let n_states = dataset_train.len();
        for batch in &dataset_train.into_iter().chunks(params.batch_size) {
            let mut images = Vec::with_capacity(params.batch_size);
            let mut targets = Vec::with_capacity(params.batch_size);
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
    model.save_with_config(&path, &config).unwrap();
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
