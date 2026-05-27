#[allow(unused)]
use agent::{
    Agent, GameState,
    agents::{ConstantEvaluator, Random, Search},
    games::DotsAndBoxes,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn dots_and_boxes(c: &mut Criterion) {
    let mut group = c.benchmark_group("DotsAndBoxes");
    let agent = Random {};
    group.bench_function("DotsAndBoxes5x5", |b| {
        b.iter(|| {
            let mut game: DotsAndBoxes = DotsAndBoxes::new();
            while !game.is_terminal() {
                game.apply_action(agent.get_action(game.clone()));
            }
        })
    });
    group.finish();
}

fn mcts(c: &mut Criterion) {
    let mut group = c.benchmark_group("dnb_mcts");
    for evals in [100, 1000, 5000] {
        let agent = Search::new(ConstantEvaluator::new(0.0), evals, true, 1.41, 1.0, 0.01);
        let luck = Random {};
        group.bench_with_input(BenchmarkId::from_parameter(evals), &evals, |b, _| {
            b.iter(|| {
                let mut game: DotsAndBoxes = DotsAndBoxes::new();
                while !game.is_terminal() {
                    let action = if game.is_random() {
                        luck.get_action(game.clone())
                    } else {
                        agent.get_action(game.clone())
                    };
                    game.apply_action(action);
                    agent.inform(action);
                }
                agent.reset();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, dots_and_boxes, mcts);
criterion_main!(benches);
