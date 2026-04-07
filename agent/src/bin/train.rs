use agent::ml::{TrainingConfig, train};

fn main() {
    train(TrainingConfig {
        epochs: 10,
        train_iters: 1000,
        test_iters: 100,
        replay_count: 1_000,
        learning_rate: 1e-3,
    });
}
