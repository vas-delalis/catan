use std::path::PathBuf;

use agent::{GameState, Tournament, agents::*, ml::Model, with_game};

fn main() {
    let _no_grad = tch::no_grad_guard();
    let game_name = String::from("Pig");
    let model_ids = [23, 24, 25];
    with_game!(game_name.as_str() => G {
        let models: Vec<Model> = model_ids.into_iter().map(|id| {
            let mut path: PathBuf = ["./models", &G::name(), &id.to_string()].iter().collect();
            path.set_extension("safetensors");
            Model::load::<G>(&path).unwrap().0
        }).collect();

        // for t in models[0].var_store().trainable_variables() {
        //     println!("{}", t);
        // }

        let mut tournament: Tournament<G> = Tournament::new(0.05, 0.05);
        for (id, model) in model_ids.iter().zip(&models) {
            tournament.add(
                Box::new(Search::new(model, 100, true, 1.41, 1.0, 0.01)),
                &format!("model {}", id),
                true,
            );
        }

        // tournament.add(
        //     Box::new(Search::new(
        //         ConstantEvaluator::new(0.0),
        //         1000,
        //         true,
        //         1.41,
        //         1.0,
        //         0.01,
        //     )),
        //     true,
        // );
        tournament.add(Box::new(Random {}), "random", true);

        tournament.play();
        tournament.leaderboard();
    });
}
