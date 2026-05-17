use std::path::Path;

use agent::{Tournament, agents::*, games::DotsAndBoxes, ml::Model};

type GAME = DotsAndBoxes;

fn main() {
    let _no_grad = tch::no_grad_guard();
    let (model1, _) =
        Model::load::<GAME>(Path::new("./models/DotsAndBoxes5x5/0.safetensors")).unwrap();

    let mut tournament: Tournament<GAME> = Tournament::new(0.25, 0.25);
    tournament.add(
        Box::new(Search::new(&model1, 100, true, 1.41, 1.0, 0.01)),
        true,
    );

    tournament.add(
        Box::new(Search::new(
            ConstantEvaluator::new(0.0),
            100,
            true,
            1.41,
            1.0,
            0.01,
        )),
        true,
    );
    tournament.add(Box::new(Random {}), true);

    tournament.add(Box::new(Random {}), false);

    tournament.play();
    tournament.leaderboard();
}
