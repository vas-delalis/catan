use agent::ml::{TrainingConfig, train};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = common::PROJECT_DIRS.config_dir().to_path_buf();
    path.push("training_config.json");
    let config_json = fs::read_to_string(&path).expect(&format!(
        "A training_config.json file should exist at {}",
        path.parent().unwrap().to_string_lossy()
    ));
    let config: TrainingConfig = serde_json::from_str(&config_json)?;

    agent::with_game!(config.model_config.game.as_str() => G {
        train::<G>(config)
    });

    Ok(())
}
