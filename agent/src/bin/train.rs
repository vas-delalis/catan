use agent::ml::{TrainingConfig, train};

fn main() {
    train(TrainingConfig {
        epochs: 50,
        train_replays: 2000,
        test_replays: 200,
        batch_size: 10,
        learning_rate: 1e-3,
        search_evals: 100,
    });
}
