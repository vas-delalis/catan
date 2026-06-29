use agent::{
    Agent,
    agents::{Evaluator, Random},
    games::{DotsAndBoxes, tic_tac_toe::TicTacToe},
    ml::{Model, QuantizedEvaluator},
};
use common::{GameState, Image, Player};

fn error_is_within_margin<G: GameState + Image>(name: &str) {
    let (model, _) = Model::load(name).expect(&format!("A model named {name} should be available"));
    let quant = QuantizedEvaluator::new(&model);
    let player = <G as GameState>::Player::list()[0];
    let luck = Random {};

    let mut errors = vec![];
    for _ in 0..1000 {
        let mut game = G::new();
        while !game.is_terminal() {
            let action = luck.get_action(game.clone());
            game.apply_action(action);

            let error = quant.evaluate(&game, player) - model.evaluate(&game, player);

            errors.push(error);
        }
    }
    let abs_sum: f32 = errors.iter().map(|e| e.abs()).sum();
    let abs_avg = abs_sum / errors.len() as f32;
    assert!(
        abs_avg < 0.05,
        "Quantization error too high. Average: {}",
        abs_avg
    );

    let sum: f32 = errors.iter().sum();
    let avg = sum / errors.len() as f32;
    assert!(
        avg.abs() < 0.05,
        "Quantization error has bias. Average: {}",
        avg
    );
}

#[test]
fn tic_tac_toe_quant_error() {
    error_is_within_margin::<TicTacToe>("test2x32");
}

#[test]
fn dots_and_boxes_quant_error() {
    error_is_within_margin::<DotsAndBoxes>("test2x32");
}
