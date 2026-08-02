use std::time::Duration;
use std::time::Instant;

use crate::ml::TrainingConfig;
use crate::ml::quantization::CLAMP_LIMIT;
use crate::{
    INTERRUPTED, Tournament,
    agents::{ConstantEvaluator, Search},
    boxed,
    ml::{Model, data::Dataset, vanilla},
};
use common::{GameState, Image, Player};
use itertools::Itertools;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

pub fn train<G: GameState + Image + Send>(mut config: TrainingConfig) {
    let model_config = config.model_config.clone();
    let params = config.hyperparameters.clone();

    let arch = vanilla::<G>(model_config.layers, model_config.hidden_dim);
    let model = Model::new(arch);
    let mut optimizer = nn::AdamW::default()
        .build(model.var_store(), params.learning_rate)
        .unwrap();
    optimizer.set_weight_decay(params.weight_decay);

    let search_evals = params.search_evals;
    let dirichlet_alpha = params.dirichlet_alpha;
    let agent_factory = || {
        boxed(Search::new(
            &model,
            search_evals,
            false,
            1.41,
            1.0,
            dirichlet_alpha,
        ))
    };
    let reference_factory = || {
        boxed(Search::new(
            ConstantEvaluator::new(0.0),
            search_evals,
            false,
            1.41,
            1.0,
            dirichlet_alpha,
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

    let id = Model::<G>::get_next_id();

    let mut dataset_train = Dataset::new(params.train_replays);
    let mut dataset_test = Dataset::new(params.test_replays);

    println!("Training {} parameters...", model.parameter_count());
    let start = Instant::now();

    for epoch in 1..=params.epochs {
        dataset_train.self_play(&model, &params, default_threads());
        dataset_test.self_play(&model, &params, default_threads());

        // Train
        let mut train_loss = 0.0;
        let n_iterations = dataset_train.len().div_ceil(params.batch_size);

        for batch in &dataset_train.drain().chunks(params.batch_size) {
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

            let output = model.infer(images);

            let loss = output.mse_loss(&targets, tch::Reduction::Mean);
            optimizer.backward_step(&loss);
            {
                // Clamp weights for later quantization
                let _no_grad = tch::no_grad_guard();
                for p in &mut model.var_store().trainable_variables() {
                    let _ = p.clamp_(-CLAMP_LIMIT, CLAMP_LIMIT);
                }
            }

            let loss: f32 = loss.try_into().unwrap();
            train_loss += loss;
        }

        config.hyperparameters.epochs = epoch;
        print!(
            "[Epoch {}] Train: {:.3} / ",
            epoch,
            train_loss / n_iterations as f32
        );

        // Test
        let _no_grad = tch::no_grad_guard(); // Turn off gradient computation
        let mut test_losses = vec![];
        for (state, values) in dataset_test.drain() {
            let x = state.tensor_image();
            let y = Tensor::from_slice(&values);
            let loss = model.infer(x).mse_loss(&y, tch::Reduction::Mean);
            let loss: f32 = loss.try_into().unwrap();
            test_losses.push(loss);
        }

        let n = test_losses.len();
        let sum: f32 = test_losses.iter().sum();
        println!("Test: {:.3}", sum / n as f32);

        // Save periodically
        if epoch % 100 == 0 && epoch < params.epochs {
            let name = format!("{}-{}", id, epoch);
            model.save(&name).unwrap();
        }

        // Save and quit when interrupted
        if INTERRUPTED.read() {
            println!(
                "\r\x1b[KInterrupted at epoch {}. Saving checkpoint...",
                epoch
            );
            let (checkpoint_path, _) = model.save_with_config(&id.to_string(), &config).unwrap();
            println!("Saved at {}", checkpoint_path.to_string_lossy());
            return;
        }
    }

    let mut norm = 0.0;
    for p in &mut model.var_store().trainable_variables() {
        let flat = p.flatten(0, -1);
        let x: f32 = flat.linalg_norm(2, 0, false, None).try_into().unwrap();
        norm += x.powi(2);
    }
    let norm = norm.sqrt();
    dbg!(&norm);

    println!("Training complete. Elapsed: {}s", start.elapsed().as_secs());

    let mut tournament = Tournament::new(0.05, 0.05)
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

    let (checkpoint_path, _) = model.save_with_config(&id.to_string(), &config).unwrap();
    println!(
        "Saved at {}",
        checkpoint_path.file_name().unwrap().to_string_lossy()
    );
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
