use catan::*;
use common::GameState;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::seq::IndexedRandom;

struct Agent {}

impl Agent {
    // pub fn new(&self, initial_obs: InitialObservation) {}

    pub fn get_action(&self, obs: Observation) -> Action {
        *obs.actions.choose(&mut rand::rng()).unwrap()
    }
}

fn play(mut state: State) {
    let agent = Agent {};
    while !state.is_terminal() {
        let action = agent.get_action(state.observe(state.current_player()));
        // dbg!(state.current_player(), state.phase, action);
        state.apply_action(action);
        // agent.update();
        // sleep(Duration::from_millis(400));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full game");
    group.throughput(criterion::Throughput::Elements(1));

    let root = State::default();

    group.bench_function("W/ longest road", |b| b.iter(|| play(root.clone())));
    group.finish();

    // let mut group = c.benchmark_group("");
    // group.throughput(criterion::Throughput::Elements(1));
    // group.bench_function("No longest road", |b| b.iter(|| play()));
    // group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
