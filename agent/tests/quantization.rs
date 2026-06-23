use std::path::PathBuf;

use agent::{
    Agent,
    agents::{Evaluator, Random},
    games::{TicTacToe, TicTacToePlayer},
    ml::{Model, QuantizedEvaluator},
};
use common::GameState;

#[test]
fn error_is_within_margin() {
    let (model, _) =
        Model::load::<TicTacToe>(&PathBuf::from("./models/TicTacToe/3.safetensors")).unwrap();
    let quant = QuantizedEvaluator::new(&model);

    let luck = Random {};
    let mut max_error = 0.0;
    for _ in 0..100 {
        let mut game = TicTacToe::new();
        while !game.is_terminal() {
            let action = luck.get_action(game.clone());
            game.apply_action(action);

            let error = (quant.evaluate(&game, TicTacToePlayer::X)
                - model.evaluate(&game, TicTacToePlayer::X))
            .abs();

            if error > max_error {
                max_error = error;
            }
        }
    }
    assert!(max_error < 0.1, "quantization error too high");
}
