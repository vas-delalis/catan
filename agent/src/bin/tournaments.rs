use agent::{
    Tournament,
    agents::{ConstantEvaluator, Random, Search},
    games::{DotsAndBoxes, TicTacToe},
    ml::Model,
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

// fn main() {
//     let _no_grad = tch::no_grad_guard();
//     let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
//     let model = create_model::<OddsGame>(&vs.root(), 16);
//     vs.load("model.safetensors").unwrap();
//     let evaluator = ModelEvaluator { model: &model };

//     let mut agents: Vec<Box<dyn Agent<OddsGame>>> = Vec::new();
//     agents.push(Box::new(Search::new(
//         OddsEvaluator {},
//         100,
//         true,
//         1.41,
//         1.0,
//         0.01,
//     )));
//     agents.push(Box::new(Search::new(
//         NormalizedOddsEvaluator {},
//         100,
//         true,
//         1.41,
//         1.0,
//         0.01,
//     )));
//     agents.push(Box::new(Search::new(
//         evaluator.clone(),
//         100,
//         true,
//         1.41,
//         1.0,
//         0.01,
//     )));

//     let mut tournament = Tournament::new(agents, 1e-2, 1e-2);
//     tournament.play();
//     tournament.leaderboard();
// }

// fn main() {
//     let _no_grad = tch::no_grad_guard();
//     let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
//     let model = create_model::<DotsAndBoxes>(&vs.root(), 32);
//     vs.load("model.safetensors").unwrap();
//     let evaluator = ModelEvaluator { model: &model };

//     let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(1e-2, 1e-2);
//     tournament.add(
//         Box::new(Search::new(evaluator, 100, true, 1.41, 1.0, 0.01)),
//         true,
//     );
//     tournament.add(
//         Box::new(Search::new(
//             ConstantEvaluator { c: 0.0 },
//             100,
//             true,
//             1.41,
//             1.0,
//             0.01,
//         )),
//         true,
//     );
//     tournament.add(Box::new(Random {}), true);
//     tournament.add(
//         Box::new(Search::new(
//             ConstantEvaluator { c: 0.0 },
//             100,
//             true,
//             1.41,
//             1.0,
//             0.01,
//         )),
//         false,
//     );

//     tournament.play();
//     tournament.leaderboard();
// }

fn main() {
    // let _no_grad = tch::no_grad_guard();
    // let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    // let model = create_model::<DotsAndBoxes>(&vs.root(), 8);
    // vs.load("dnb8.safetensors").unwrap();
    // let evaluator = ModelEvaluator { model: &model };

    // let mut vs2 = tch::nn::VarStore::new(tch::Device::Cpu);
    // let model2 = create_model::<DotsAndBoxes>(&vs.root(), 16);
    // vs2.load("dnb16.safetensors").unwrap();
    // let evaluator2 = ModelEvaluator { model: &model2 };

    // let mut vs3 = tch::nn::VarStore::new(tch::Device::Cpu);
    // let model3 = create_model::<DotsAndBoxes>(&vs.root(), 32);
    // vs3.load("dnb32.safetensors").unwrap();
    // let evaluator3 = ModelEvaluator { model: &model3 };

    // let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(1e-2, 1e-2);
    // tournament.add(
    //     Box::new(Search::new(evaluator, 100, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    // tournament.add(
    //     Box::new(Search::new(evaluator2, 100, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    // tournament.add(
    //     Box::new(Search::new(evaluator3, 100, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    // tournament.add(
    //     Box::new(Search::new(
    //         ConstantEvaluator { constant: 0.0 },
    //         100,
    //         true,
    //         1.41,
    //         1.0,
    //         0.01,
    //     )),
    //     true,
    // );

    // tournament.play();
    // tournament.leaderboard();
}
