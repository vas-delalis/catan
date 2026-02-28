use agent::{
    Agent, GameState,
    agents::{Evaluator, Random},
    games::TicTacToe,
    ml::{Model, TrainingConfig},
};
use burn::{
    backend::{Autodiff, LibTorch, libtorch::LibTorchDevice},
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::random_range;
use std::hint::black_box;

fn load_model<B: Backend>(artifact_dir: &str, device: B::Device) -> Model<B> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    config.model.init::<B>(&device).load_record(record)
}

fn random_state() -> TicTacToe {
    let agent = Random {};
    let mut game = TicTacToe::new();
    for _ in 0..random_range(1..=3) {
        game.apply_action(agent.get_action(game.clone()));
    }
    game
}

fn criterion_benchmark(c: &mut Criterion) {
    type B = LibTorch<f32>;
    type AB = Autodiff<B>;
    let device = LibTorchDevice::Cuda(0);
    let artifact_dir = "./model";
    let model = load_model::<AB>(artifact_dir, device);
    c.bench_function("fib 20", |b| {
        b.iter_batched(
            random_state,
            |state| model.evaluate(state),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
