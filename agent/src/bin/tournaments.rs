use agent::{
    Tournament,
    agents::*,
    games::DotsAndBoxes,
    ml::{Model, QuantizedEvaluator},
    with_game,
};
use common::GameState;

// alphas:
// DotsAndBoxes: 0.0001

fn main() {
    let _no_grad = tch::no_grad_guard();
    let game_name = String::from("DotsAndBoxes");
    let model_ids = [0, 11];
    let evals = [10000, 10000, 1000, 100];
    let alphas = [0.001, 0.001, 0.001];
    type G = DotsAndBoxes;
    // with_game!(game_name.as_str() => G {
    let models: Vec<Model<G>> = model_ids
        .into_iter()
        .map(|id: u32| Model::load(id).unwrap().0)
        .collect();

    // for t in models[0].var_store().trainable_variables() {
    //     println!("{}", t);
    // }

    let mut tournament: Tournament<G> = Tournament::new(0.05, 0.05);
    for (idx, (id, model)) in model_ids.iter().zip(&models).enumerate() {
        tournament.add(
            Box::new(Search::new(
                QuantizedEvaluator::new(model),
                evals[idx],
                true,
                1.41,
                1.0,
                alphas[idx],
            )),
            &format!("model {}x{} α={}", id, evals[idx], alphas[idx]),
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
    //     "constant",
    //     false,
    // );
    tournament.add(Box::new(Random {}), "random", false);
    tournament.add(Box::new(Random {}), "random", false);
    // tournament.add(Box::new(OptimalTicTacToe {}), "optimal", true);

    tournament.play();
    tournament.leaderboard();
    // });
}
