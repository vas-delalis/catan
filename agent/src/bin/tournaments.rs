use std::path::Path;

use agent::{
    Tournament,
    agents::*,
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

    // let arch = vanilla::<GAME>(8, 32);
    // let mut model3 = Model::new::<GAME>(arch);
    // model3
    //     .load(Path::new("./models/DotsAndBoxes/6.safetensors"))
    //     .unwrap();

    // let arch = vanilla::<GAME>(5, 16);
    // let mut model4 = Model::new::<GAME>(arch);
    // model4
    //     .load(Path::new("./models/DotsAndBoxes/5.safetensors"))
    //     .unwrap();

    let mut tournament: Tournament<GAME> = Tournament::new(0.25, 0.25);
    tournament.add(
        Box::new(Search::new(&model1, 10000, true, 1.41, 1.0, 0.01)),
        true,
    );
    tournament.add(
        Box::new(Search::new(&model2, 10000, true, 1.41, 1.0, 0.01)),
        true,
    );
    // tournament.add(
    //     Box::new(Search::new(&model3, 100, true, 1.41, 1.0, 0.01)),
    //     true,
    // );
    tournament.add(Box::new(Random {}), true);

    tournament.add(Box::new(Random {}), false);

    tournament.play();
    tournament.leaderboard();
}
