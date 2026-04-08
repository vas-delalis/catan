use agent::{
    Agent, Tournament,
    agents::{ConstantEvaluator, Random, Search},
    games::{NormalizedOddsEvaluator, OddsEvaluator, OddsGame, TicTacToe},
    ml::{ModelEvaluator, create_model},
};

// fn main() {
//     let mut agents: Vec<Box<dyn Agent<OddsGame>>> = Vec::new();
//     agents.push(Box::new(Random {}));
//     agents.push(Box::new(Search::<OddsGame, ConstantEvaluator>::new(
//         ConstantEvaluator {},
//         10,
//         true,
//         1.41,
//         1.0,
//         0.01,
//     )));
//     agents.push(Box::new(Search::<OddsGame, OddsEvaluator>::new(
//         OddsEvaluator {},
//         10,
//         true,
//         1.41,
//         1.0,
//         0.01,
//     )));
//     agents.push(Box::new(Search::<OddsGame, OddsEvaluator>::new(
//         OddsEvaluator {},
//         10,
//         true,
//         1.41,
//         1.0,
//         0.01,
//     )));

//     let mut tournament: Tournament<OddsGame> = Tournament::new(agents, 1e-2, 1e-2);
//     tournament.play();
//     tournament.leaderboard();
// }

fn main() {
    let _no_grad = tch::no_grad_guard();
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let model = create_model(&vs.root(), 4);
    vs.load("model.safetensors").unwrap();
    let evaluator = ModelEvaluator { model: &model };

    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    agents.push(Box::new(Search::new(
        evaluator.clone(),
        10000,
        true,
        1.41,
        1.0,
        0.01,
    )));
    agents.push(Box::new(Search::new(
        evaluator.clone(),
        10000,
        true,
        1.41,
        1.0,
        0.01,
    )));

    let mut tournament = Tournament::new(agents, 1e-2, 1e-2);
    tournament.play();
    tournament.leaderboard();
}
