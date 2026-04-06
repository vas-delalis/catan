use rand::seq::IteratorRandom;
use tch::{
    Tensor,
    nn::{self, OptimizerConfig},
};

use crate::{
    Agent, GameState, Tournament,
    agents::{ConstantEvaluator, Evaluator, Search},
    games::{Cell, TicTacToe, TicTacToePlayer},
    ml::{self, Model, data::Dataset},
};

#[derive(Clone)]
struct ModelEvaluator<'a> {
    net: &'a Model<'a>,
}

impl<'a> Evaluator<TicTacToe> for ModelEvaluator<'a> {
    fn evaluate(&self, game_state: TicTacToe) -> f64 {
        let image = batch(&game_state);
        (self.net)(&image).try_into().unwrap()
    }
}

fn batch(game_state: &TicTacToe) -> Tensor {
    use crate::GameState;
    let mut plane1: Vec<f32> = vec![];
    let mut plane2: Vec<f32> = vec![];
    let plane3: Vec<f32> = if game_state.current_player() == TicTacToePlayer::X {
        vec![1.0; 1]
    } else {
        vec![0.0; 1]
    };

    for tile in game_state.board {
        match tile {
            Some(p) => {
                if p == TicTacToePlayer::X {
                    plane1.push(1.0);
                    plane2.push(0.0);
                } else {
                    plane1.push(0.0);
                    plane2.push(1.0);
                }
            }
            None => {
                plane1.push(0.0);
                plane2.push(0.0);
            }
        }
    }
    let plane1 = Tensor::from_slice(&plane1);
    let plane2 = Tensor::from_slice(&plane2);
    let plane3 = Tensor::from_slice(&plane3); //.reshape([3, 3]);

    Tensor::cat(&[plane1, plane2, plane3], 0)
}

pub fn train() {
    let device = tch::Device::Cpu;
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    let model = ml::create_model(&root);
    let evaluator = ModelEvaluator { net: &model };
    let agent =
        Search::<TicTacToe, ModelEvaluator>::new(evaluator.clone(), 100, false, 1.41, 1.0, 0.01);
    let reference_agent = Search::<TicTacToe, ConstantEvaluator>::new(
        ConstantEvaluator {},
        100,
        false,
        1.41,
        1.0,
        0.01,
    );

    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    agents.push(Box::new(reference_agent.clone()));
    agents.push(Box::new(agent.clone()));
    let mut tournament = Tournament::new(agents);
    tournament.play();
    tournament.leaderboard();

    for epoch in 1..=10 {
        let mut dataset_train = Dataset::new();
        let mut dataset_test = Dataset::new();
        dataset_train.selfplay(&agent);
        dataset_test.selfplay(&agent);

        // Train
        for _ in 1..=1000 {
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
        for _ in 1..=100 {
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
    agents.push(Box::new(reference_agent));
    agents.push(Box::new(agent.clone()));
    let mut tournament = Tournament::new(agents);
    tournament.play();
    tournament.leaderboard();

    let mut game = TicTacToe::new();
    dbg!(evaluator.evaluate(game.clone()));

    game.apply_action(Cell(0));
    dbg!(evaluator.evaluate(game.clone()));

    game.apply_action(Cell(8));
    dbg!(evaluator.evaluate(game.clone()));

    game.apply_action(Cell(1));
    dbg!(evaluator.evaluate(game.clone()));

    game.apply_action(Cell(7));
    dbg!(evaluator.evaluate(game.clone()));
}
