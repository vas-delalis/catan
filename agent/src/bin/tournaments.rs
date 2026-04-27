use std::path::Path;

use agent::{
    Tournament,
    agents::{ConstantEvaluator, Random, Search},
    games::DotsAndBoxes,
    ml::{Model, vanilla},
};

type GAME = DotsAndBoxes;

fn main() {
    let _no_grad = tch::no_grad_guard();
    let arch = vanilla::<GAME>(2, 16);
    let mut model1 = Model::new::<GAME>(arch);
    model1
        .load(Path::new("./models/DotsAndBoxes/0.safetensors"))
        .unwrap();

    let arch = vanilla::<GAME>(3, 16);
    let mut model2 = Model::new::<GAME>(arch);
    model2
        .load(Path::new("./models/DotsAndBoxes/1.safetensors"))
        .unwrap();

    // let arch = vanilla::<GAME>(4, 32);
    // let mut model3 = Model::new::<GAME>(arch);
    // model3
    //     .load(Path::new("./models/DotsAndBoxes/8.safetensors"))
    //     .unwrap();

    // let arch = vanilla::<GAME>(5, 32);
    // let mut model4 = Model::new::<GAME>(arch);
    // model4
    //     .load(Path::new("./models/DotsAndBoxes/9.safetensors"))
    //     .unwrap();

    let mut tournament: Tournament<GAME> = Tournament::new(0.01, 0.01);
    // tournament.add(
    //     Box::new(Search::new(&model1, 200, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    // tournament.add(
    //     Box::new(Search::new(&model2, 1000, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    // tournament.add(
    //     Box::new(Search::new(&model1, 100, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    tournament.add(
        Box::new(Search::new(&model2, 10000, true, 1.41, 1.0, 0.01)),
        true,
    );
    tournament.add(
        Box::new(Search::new(
            ConstantEvaluator { constant: 0.0 },
            100,
            true,
            1.41,
            1.0,
            0.01,
        )),
        true,
    );
    tournament.add(Box::new(Random {}), false);
    tournament.add(Box::new(Random {}), false);
    // tournament.add(Box::new(Random {}), false);
    // tournament.add(Box::new(Random {}), false);
    // tournament.add(
    //     Box::new(Search::new(
    //         ConstantEvaluator { constant: 0.0 },
    //         10,
    //         true,
    //         1.41,
    //         1.0,
    //         0.01,
    //     )),
    //     false,
    // );

    tournament.play();
    tournament.leaderboard();
}
