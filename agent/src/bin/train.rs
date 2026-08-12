use agent::ml::{Trainer, TrainingConfig};
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
        let mut trainer: Trainer<G> = Trainer::new(config);

        trainer.run_tournament();

        println!("Training {} parameters...", trainer.model.parameter_count());

        while trainer.epoch < trainer.params.max_epochs {
            trainer.generate_data();
            trainer.run_epoch();
        }
        println!("Training complete. Elapsed: {:.1}s", trainer.metadata().training_data.duration_secs);

        trainer.run_tournament();

        let (checkpoint_path, _) = trainer.save_model()?;
        println!(
            "Saved as {}",
            checkpoint_path.file_name().unwrap().to_string_lossy()
        );
    });

    Ok(())
}
