use rand::seq::IteratorRandom;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

use crate::{
    Agent, Tournament,
    agents::{ConstantEvaluator, Search},
    games::DotsAndBoxes,
    ml::{self, Batch, data::Dataset, model::ModelEvaluator},
};

pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub train_iters: usize,
    pub test_iters: usize,
    pub replay_count: usize,
}

pub fn train(config: TrainingConfig) {
    let device = tch::Device::Cpu;
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut opt = nn::Adam::default()
        .build(&vs, config.learning_rate)
        .unwrap();

    let model = ml::create_model::<DotsAndBoxes>(&root, 16);
    let evaluator = ModelEvaluator { model: &model };
    let agent = Search::new(evaluator.clone(), 100, false, 1.41, 1.0, 0.01);
    let reference_agent = Search::new(ConstantEvaluator { c: 0.0 }, 100, false, 1.41, 1.0, 0.01);

    let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(1e-1, 1e-1);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    let params: usize = vs.trainable_variables().iter().map(|t| t.numel()).sum();
    println!("Training {} parameters", params);

    for epoch in 1..=config.epochs {
        let mut dataset_train = Dataset::new(config.replay_count);
        let mut dataset_test = Dataset::new(config.test_iters);
        dataset_train.selfplay(&agent);
        dataset_test.selfplay(&agent);

        // Train
        for (state, value) in dataset_train {
            let x = state.batch();
            let y = Tensor::from(value as f32);
            let loss = model(&x).mse_loss(&y, tch::Reduction::Mean);
            opt.backward_step(&loss);
        }

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

    let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(1e-2, 1e-2);
    tournament.add(Box::new(agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), true);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.add(Box::new(reference_agent.clone()), false);
    tournament.play();
    tournament.leaderboard();

    vs.save("model.safetensors").unwrap();
}
