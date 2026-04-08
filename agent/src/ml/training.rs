use rand::seq::IteratorRandom;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

use crate::{
    Agent, Tournament,
    agents::{ConstantEvaluator, Search},
    games::TicTacToe,
    ml::{
        self,
        data::Dataset,
        model::{ModelEvaluator, batch},
    },
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

    let model = ml::create_model(&root, 4);
    let evaluator = ModelEvaluator { model: &model };
    let agent = Search::new(evaluator.clone(), 100, false, 1.41, 1.0, 0.01);
    let reference_agent = Search::new(ConstantEvaluator {}, 100, false, 1.41, 1.0, 0.01);

    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    agents.push(Box::new(reference_agent.clone()));
    agents.push(Box::new(agent.clone()));
    let mut tournament = Tournament::new(agents, 1e-2, 1e-2);
    tournament.play();
    tournament.leaderboard();

    for epoch in 1..=config.epochs {
        let mut dataset_train = Dataset::new(config.replay_count);
        let mut dataset_test = Dataset::new(config.replay_count);
        dataset_train.selfplay(&agent);
        dataset_test.selfplay(&agent);

        // Train
        for _ in 1..=config.train_iters {
            let index = (0..dataset_train.len()).choose(&mut rand::rng()).unwrap();
            let (state, value) = dataset_train.get(index);
            let x = batch(&state);
            let y = Tensor::from(value as f32);
            let loss = model(&x).mse_loss(&y, tch::Reduction::Mean);
            opt.backward_step(&loss);
        }

        // Test
        let _no_grad = tch::no_grad_guard(); // Turn off gradient computation
        let mut test_losses = vec![];
        for _ in 1..=config.test_iters {
            let index = (0..dataset_test.len()).choose(&mut rand::rng()).unwrap();
            let (state, value) = dataset_test.get(index);
            let x = batch(&state);
            let y = Tensor::from(value as f32);
            let loss = model(&x).mse_loss(&y, tch::Reduction::Mean);
            let loss: f32 = loss.try_into().unwrap();
            test_losses.push(loss);
        }

        let n = test_losses.len();
        let sum: f32 = test_losses.iter().sum();
        println!("[Test - Epoch {}] Loss {:.3}", epoch, sum / n as f32);
    }

    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    agents.push(Box::new(reference_agent.clone()));
    agents.push(Box::new(agent.clone()));
    let mut tournament = Tournament::new(agents, 1e-2, 1e-2);
    tournament.play();
    tournament.leaderboard();

    vs.save("model.safetensors").unwrap();
}
