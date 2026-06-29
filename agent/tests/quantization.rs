use agent::{
    Agent,
    agents::{Evaluator, Random},
    games::{
        DotsAndBoxes,
        tic_tac_toe::{self, TicTacToe},
    },
    ml::{Model, QuantizedEvaluator},
};
use common::{GameState, Player};

#[test]
fn error_is_within_margin() {
    let (model, _) = Model::load(7).unwrap();
    let quant = QuantizedEvaluator::new(&model);

    let luck = Random {};
    let mut errors = vec![];
    let mut max_error = 0.0;
    for _ in 0..100 {
        let mut game = TicTacToe::new();
        while !game.is_terminal() {
            let action = luck.get_action(game.clone());
            game.apply_action(action);

            let error = quant.evaluate(&game, tic_tac_toe::Player::X)
                - model.evaluate(&game, tic_tac_toe::Player::X);

            errors.push(error);
            if error.abs() > max_error {
                max_error = error.abs();
            }
        }
    }
    let sum: f32 = errors.iter().sum();
    dbg!(sum / errors.len() as f32);
    assert!(max_error < 0.1, "quantization error too high");
}

#[test]
fn error_is_within_margin2() {
    let (model, _) = Model::load(11).unwrap();
    let quant = QuantizedEvaluator::new(&model);
    let player = <DotsAndBoxes as GameState>::Player::list()[0];
    let luck = Random {};
    let mut max_error = 0.0;
    let mut errors = vec![];
    for _ in 0..1000 {
        let mut game = DotsAndBoxes::new();
        while !game.is_terminal() {
            let action = luck.get_action(game.clone());
            game.apply_action(action);

            let error = quant.evaluate(&game, player) - model.evaluate(&game, player);

            errors.push(error);
            if error.abs() > max_error {
                max_error = error.abs();
            }
        }
    }
    let sum: f32 = errors.iter().sum();
    dbg!(sum / errors.len() as f32);
    assert!(max_error < 0.15, "quantization error too high");
}
