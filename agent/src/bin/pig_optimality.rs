use agent::{
    INTERRUPTED, Tournament,
    agents::{Evaluator, Search},
    boxed,
    games::{Pig, pig::OptimalPig},
    ml::{QuantizedEvaluator, Trainer, TrainingConfig},
};
use std::{fs, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = common::PROJECT_DIRS.config_dir().to_path_buf();
    path.push("training_config.json");
    let config_json = fs::read_to_string(&path).expect(&format!(
        "A training_config.json file should exist at {}",
        path.parent().unwrap().to_string_lossy()
    ));
    let config: TrainingConfig = serde_json::from_str(&config_json)?;

    type G = Pig;
    let mut trainer: Trainer<G> = Trainer::new(config);

    trainer.run_tournament();
    let params = trainer.params.clone();
    let optimal_evaluator = OptimalPig::new();

    println!("Training {} parameters...", trainer.model.parameter_count());

    while trainer.epoch < params.max_epochs && !INTERRUPTED.read() {
        trainer.generate_data();

        let mut optimal_outputs = Vec::with_capacity(trainer.dataset_train.replay_buffer.len());
        for batch in trainer
            .dataset_train
            .replay_buffer
            .chunks(params.batch_size)
        {
            let mut batch_loss = 0.0;
            for (state, values) in batch {
                let output = optimal_evaluator.evaluate(state);
                batch_loss += (output[0] - values[0]).powi(2);
                batch_loss += (output[1] - values[1]).powi(2);
                optimal_outputs.push(vec![output[0], output[1]]);
            }
            batch_loss /= 2.0;
            batch_loss /= batch.len() as f32;
            println!("Optimal loss: {}", batch_loss);
        }

        trainer.run_epoch();

        let mse: f32 = optimal_outputs
            .iter()
            .zip(&trainer.prev_values)
            .map(|(o, m)| o.iter().zip(m).map(|(a, b)| (a - b).powi(2)).sum::<f32>())
            .sum::<f32>()
            / (optimal_outputs.len() * 2) as f32;
        println!("Optimal vs model MSE: {}", mse);

        if trainer.epoch % 10 == 0 {
            let mut tournament = Tournament::new(0.01, 0.01).max_time(Duration::from_secs(10));
            tournament.add(
                || {
                    boxed(Search::new(
                        QuantizedEvaluator::new(&trainer.model),
                        100,
                        true,
                        1.41,
                        1.0,
                        0.01,
                    ))
                },
                &format!("model {}x{} α={}", trainer.model_id, 1000, 0.00001),
                true,
            );
            tournament.add(
                || boxed(Search::new(&optimal_evaluator, 100, true, 1.41, 1.0, 0.01)),
                "Optimal",
                true,
            );
            tournament.play();
            tournament.leaderboard();
        }
    }
    println!(
        "Training complete. Elapsed: {:.1}s",
        trainer.metadata().training_data.duration_secs
    );

    trainer.run_tournament();

    let (checkpoint_path, _) = trainer.save_model()?;
    println!(
        "Saved as {}",
        checkpoint_path.file_name().unwrap().to_string_lossy()
    );

    Ok(())
}
