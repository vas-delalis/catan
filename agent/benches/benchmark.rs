use std::{hint::black_box, path::PathBuf};

#[allow(unused)]
use agent::{
    Agent,
    agents::{ConstantEvaluator, Random, Search},
    games::DotsAndBoxes,
};
use agent::{
    agents::Evaluator,
    games::{Cell, TicTacToe, TicTacToePlayer},
    ml::{Model, QuantizedEvaluator},
};
use common::GameState;
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

fn tic_tac_toe_game() -> TicTacToe {
    let mut game = TicTacToe::new();
    game.apply_action(Cell(0)); // X O X
    game.apply_action(Cell(1)); // X O X
    game.apply_action(Cell(2)); // O X -
    game.apply_action(Cell(4));
    game.apply_action(Cell(3));
    game.apply_action(Cell(6));
    game.apply_action(Cell(5));
    game.apply_action(Cell(7));
    game
}

fn inference(c: &mut Criterion) {
    let game = tic_tac_toe_game();

    let (model, _) =
        Model::load::<TicTacToe>(&PathBuf::from("./models/TicTacToe/3.safetensors")).unwrap();
    let quantized_evaluator = QuantizedEvaluator::new(&model);

    let mut group = c.benchmark_group("inference");
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("float", |b| {
        b.iter(|| model.evaluate(black_box(&game), TicTacToePlayer::X))
    });
    group.bench_function("quantized", |b| {
        b.iter(|| quantized_evaluator.evaluate(black_box(&game), TicTacToePlayer::X))
    });
}

criterion_group!(benches, dots_and_boxes, mcts, inference,);
criterion_main!(benches);
