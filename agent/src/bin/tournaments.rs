use agent::{
    Tournament,
    agents::*,
    boxed,
    games::Pig,
    ml::{Model, QuantizedEvaluator},
    with_game,
};

// alphas:
// DotsAndBoxes: 0.0001

fn main() {
    let _no_grad = tch::no_grad_guard();
    let game_name = String::from("TicTacToe");
    let model_ids = [12, 15];
    let evals = [1000, 1000, 1000, 1000, 1000];
    let alphas = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001];
    type G = Pig;
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
        let eval = evals[idx];
        let alpha = alphas[idx];
        tournament.add(
            move || {
                boxed(Search::new(
                    QuantizedEvaluator::new(model),
                    eval,
                    true,
                    1.41,
                    1.0,
                    alpha,
                ))
            },
            &format!("model {}x{} α={}", id, evals[idx], alphas[idx]),
            true,
        );
    }

    // tournament.add(|| boxed(OptimalPig::new()), "OptimalPig", true);

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
    // tournament.add(Box::new(Random {}), "random", false);
    // tournament.add(Box::new(Random {}), "random", false);
    // tournament.add(Box::new(OptimalTicTacToe {}), "optimal", true);

    tournament.play();
    tournament.leaderboard();
    // });
}
