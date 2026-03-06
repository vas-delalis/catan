#![recursion_limit = "256"]

use agent::{
    Action, Agent, GameState, Player, Tournament,
    agents::{ConstantEvaluator, Evaluator, Random, Search},
    games::{
        Cell, DotsAndBoxes, DotsAndBoxesAction as DnbEdge, DotsAndBoxesDir as DnbDir,
        ScoreEvaluator, TicTacToe, TopsideEvaluator,
    },
    ml::{Model, ModelConfig, ModelEvaluator, TicTacToeBatcher, TrainingConfig, train},
};
use burn::{
    backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    },
    data::dataloader::batcher::Batcher,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

// fn human_vs_human() {
//     let mut game = DotsAndBoxes::new();
//     let agent = Human {};
//     while !game.is_terminal() {
//         let scratch = game.clone();
//         game.apply_action(agent.get_action(scratch));
//         println!("Value: {}", ScoreEvaluator::evaluate(game.clone()));
//     }
//     for p in <DotsAndBoxes as GameState>::Player::list() {
//         println!("{:?} {:?}", p, game.outcome(p).unwrap())
//     }
// }

// fn tournament() {
//     let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
//     for value in [0.0] {
//         for evals in [1000, 1001] {
//             for alpha in [0.005] {
//                 agents.push(Box::new(Search::<TicTacToe, ConstantEvaluator>::new(
//                     evals, 1.41, 1.0, alpha, value,
//                 )));
//             }
//         }
//     }
//     // agents.push(Box::new(Random {}));
//     let mut tournament: Tournament<TicTacToe> = Tournament::new(agents);
//     tournament.play();
//     tournament.leaderboard();
// }

// fn dots() {
//     let mut agents: Vec<Box<dyn Agent<DotsAndBoxes>>> = Vec::new();
//     for _ in 0..3 {
//         agents.push(Box::new(Random {}));
//     }
//     agents.push(Box::new(Search::<DotsAndBoxes, ScoreEvaluator>::new(
//         1000, 1.41, 1.0, 0.01, 0.0,
//     )));
//     let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(agents);
//     tournament.play_multiplayer();
//     tournament.leaderboard();
// }

// fn dots2() {
//     let game = DotsAndBoxes::new();
//     let agent: Search<DotsAndBoxes, TopsideEvaluator> = Search::new(10000, 1.41, 1.0, 0.5, 0.5);
//     agent.run(game.clone());
// }

// fn wtf() {
//     let mut game = TicTacToe::new();
//     game.apply_action(Cell(0)); // X
//     game.apply_action(Cell(1));
//     game.apply_action(Cell(3)); // X
//     game.apply_action(Cell(4));
//     let agent = Search::<TicTacToe, ConstantEvaluator>::new(100, 1.41, 1.0, 0.1, 0.0);
//     agent.run(game);
// }

// fn wtf2() {
//     use DnbDir::*;
//     let mut game = DotsAndBoxes::new();
//     game.apply_action(DnbEdge(0, 0, N)); // A
//     game.apply_action(DnbEdge(1, 0, N));
//     game.apply_action(DnbEdge(2, 0, W));
//     game.apply_action(DnbEdge(3, 0, W));

//     game.apply_action(DnbEdge(1, 0, W)); // A
//     game.apply_action(DnbEdge(1, 0, N));
//     game.apply_action(DnbEdge(2, 0, W));
//     game.apply_action(DnbEdge(3, 0, W));

//     game.apply_action(DnbEdge(0, 1, N));
//     game.apply_action(DnbEdge(2, 0, W));
//     game.apply_action(DnbEdge(3, 0, W));
//     game.apply_action(DnbEdge(3, 0, W));

//     let agent = Search::<DotsAndBoxes, ScoreEvaluator>::new(1000, 1.41, 1.0, 0.01, 0.0);
//     agent.run(game);
// }

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let model = load_model(artifact_dir, device.clone());
    let evaluator = ModelEvaluator {
        model,
        batcher: TicTacToeBatcher {},
        device,
    };

    let agent: Box<dyn Agent<TicTacToe>> = Box::new(Search::<TicTacToe, ModelEvaluator<B>>::new(
        evaluator, 10, true, 1.41, 1.0, 0.01,
    ));
    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    agents.push(agent);
    agents.push(Box::new(Search::<TicTacToe, ConstantEvaluator>::new(
        ConstantEvaluator {},
        10,
        true,
        1.41,
        1.0,
        0.01,
    )));
    let mut tournament: Tournament<TicTacToe> = Tournament::new(agents);
    tournament.play();
    tournament.leaderboard();
}

pub fn load_model<B: Backend>(artifact_dir: &str, device: B::Device) -> Model<B> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    config.model.init::<B>(&device).load_record(record)
}

fn main() {
    type B = LibTorch<f32>;
    type AB = Autodiff<B>;

    let device = LibTorchDevice::Cpu;

    let artifact_dir = "./model";

    let model_config = ModelConfig::new(512);
    let model = model_config.init::<B>(&device);

    infer::<AB>(artifact_dir, device.clone());
}
