use agent::games::{DotsAndBoxes, OddsGame, Pig, TicTacToe};
use agent::ml::{TrainingConfig, train};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_json = fs::read_to_string("training_config.json")?;
    let config: TrainingConfig = serde_json::from_str(&config_json)?;

    match config.model_config.game.as_str() {
        "DotsAndBoxes" => train::<DotsAndBoxes>(config),
        "OddsGame" => train::<OddsGame>(config),
        "TicTacToe" => train::<TicTacToe>(config),
        "Pig" => train::<Pig>(config),
        other => return Err(format!("Unknown game: {}", other).into()),
    }

    Ok(())
}
