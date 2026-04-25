use std::path::Path;

use agent::{
    Tournament,
    agents::{ConstantEvaluator, Search},
    games::DotsAndBoxes,
    ml::{Model, two_layers},
};

type GAME = DotsAndBoxes;

fn main() {
    let _no_grad = tch::no_grad_guard();
    let arch = two_layers::<GAME>(8);
    let mut model1 = Model::new::<GAME>(arch);
    model1
        .load(Path::new("./models/DotsAndBoxes/3.safetensors"))
        .unwrap();

    // let arch = two_layers::<GAME>(8);
    // let mut model2 = Model::new::<GAME>(arch);
    // model2
    //     .load(Path::new("./models/DotsAndBoxes/1.safetensors"))
    //     .unwrap();

    // let arch = two_layers::<GAME>(8);
    // let mut model3 = Model::new::<GAME>(arch);
    // model3
    //     .load(Path::new("./models/DotsAndBoxes/3.safetensors"))
    //     .unwrap();

    let mut tournament: Tournament<GAME> = Tournament::new(1e-2, 1e-2);
    tournament.add(
        Box::new(Search::new(&model1, 1000, true, 1.41, 1.0, 0.01)),
        true,
    );
    // tournament.add(
    //     Box::new(Search::new(&model2, 20, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    // tournament.add(
    //     Box::new(Search::new(&model3, 6, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    tournament.add(
        Box::new(Search::new(
            ConstantEvaluator { constant: 0.0 },
            10,
            true,
            1.41,
            1.0,
            0.01,
        )),
        true,
    );
    tournament.add(
        Box::new(Search::new(
            ConstantEvaluator { constant: 0.0 },
            10,
            true,
            1.41,
            1.0,
            0.01,
        )),
        false,
    );
    tournament.add(
        Box::new(Search::new(
            ConstantEvaluator { constant: 0.0 },
            10,
            true,
            1.41,
            1.0,
            0.01,
        )),
        false,
    );

    tournament.play();
    tournament.leaderboard();
}
