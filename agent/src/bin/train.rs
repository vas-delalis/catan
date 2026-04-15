use agent::ml::{TrainingConfig, train};

fn main() {
    train(TrainingConfig {
        epochs: 20,
        train_iters: 1000,
        test_iters: 100,
        replay_count: 1000,
        learning_rate: 1e-3,
    });
}
