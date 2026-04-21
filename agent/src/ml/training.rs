use crate::{
    Tournament,
    agents::{ConstantEvaluator, Search},
    games::DotsAndBoxes,
    ml::{self, Batch, data::Dataset, model::ModelEvaluator},
};
use itertools::Itertools;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

type GAME = DotsAndBoxes;

pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub train_replays: usize,
    pub test_replays: usize,
    pub batch_size: usize,
    pub search_evals: usize,
}

pub fn train(config: TrainingConfig) {
    let device = tch::Device::Cpu;
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut opt = nn::Adam::default()
        .build(&vs, config.learning_rate)
        .unwrap();

    let model = ml::create_model::<GAME>(&root, 32);
    let evaluator = ModelEvaluator::new(&model);
    let agent = Search::new(
        evaluator.clone(),
        config.search_evals,
        false,
        1.41,
        1.0,
        0.01,
    );
    let reference_agent = Search::new(
        ConstantEvaluator::new(0.0),
        config.search_evals,
        false,
        1.41,
        1.0,
        0.01,
    );

    let mut tournament: Tournament<GAME> = Tournament::new(0.1, 0.1);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    let params: usize = vs.trainable_variables().iter().map(|t| t.numel()).sum();
    println!("Training {} parameters...", params);

    for epoch in 1..=config.epochs {
        let mut dataset_train = Dataset::new(config.train_replays);
        let mut dataset_test = Dataset::new(config.test_replays);
        dataset_train.selfplay(&agent);
        dataset_test.selfplay(&agent);
        // Train
        let mut train_losses = vec![];
        for batch in &dataset_train.into_iter().chunks(config.batch_size) {
            let mut loss = Tensor::from(0f32);
            for (state, value) in batch {
                let x = state.batch();
                let y = Tensor::from(value as f32);
                let output = model(&x);
                loss += output.mse_loss(&y, tch::Reduction::Mean);
                // dbg!(&x);
                // dbg!(&y);
                // let output: f32 = output.try_into().unwrap();
                // dbg!(output);
            }
            opt.backward_step(&loss);
            let loss: f32 = loss.try_into().unwrap();
            train_losses.push(loss);
        }

        let n = train_losses.len();
        let sum: f32 = train_losses.iter().sum();
        println!(
            "[Train - Epoch {}] Loss {:.3}",
            epoch,
            sum / n as f32 / config.batch_size as f32
        );

        // Test
        let _no_grad = tch::no_grad_guard(); // Turn off gradient computation
        let mut test_losses = vec![];
        for (state, value) in dataset_test {
            let x = state.batch();
            let y = Tensor::from(value as f32);
            let loss = model(&x).mse_loss(&y, tch::Reduction::Mean);
            let loss: f32 = loss.try_into().unwrap();
            test_losses.push(loss);
        }

        let n = test_losses.len();
        let sum: f32 = test_losses.iter().sum();
        println!("[Test - Epoch {}] Loss {:.3}", epoch, sum / n as f32);
    }

    let mut tournament: Tournament<GAME> = Tournament::new(0.05, 0.05);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    vs.save("model.safetensors").unwrap();
}
