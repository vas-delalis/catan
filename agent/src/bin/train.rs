use agent::ml::{TrainingConfig, train};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_json = fs::read_to_string("training_config.json")?;
    let config: TrainingConfig = serde_json::from_str(&config_json)?;
    train(config);
    Ok(())
}
