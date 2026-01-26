use agent::{
    Action, Agent, Evaluator, GameState, Human, Player, Random, Search, Tournament,
    games::{
        Cell, DotsAndBoxes, DotsAndBoxesAction as DnbEdge, DotsAndBoxesDir as DnbDir,
        ScoreEvaluator, TicTacToe, TopsideEvaluator,
    },
};

fn human_vs_human() {
    let mut game = DotsAndBoxes::new();
    let agent = Human {};
    while !game.is_terminal() {
        let scratch = game.clone();
        game.apply_action(agent.get_action(scratch));
        println!("Value: {}", ScoreEvaluator::evaluate(game.clone()));
    }
    for p in <DotsAndBoxes as GameState>::Player::list() {
        println!("{:?} {:?}", p, game.outcome(p).unwrap())
    }
}

struct ConstantEvaluator {}
impl<G: GameState> Evaluator<G> for ConstantEvaluator {
    fn evaluate(_: G) -> f64 {
        0.0
    }
}

fn tournament() {
    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    for value in [0.0] {
        for evals in [1000] {
            for alpha in [0.0025, 0.0050, 0.0075, 0.0100] {
                agents.push(Box::new(Search::<TicTacToe, ConstantEvaluator>::new(
                    evals, 1.41, 1.0, alpha, value,
                )));
            }
        }
    }
    // agents.push(Box::new(Random {}));
    let mut tournament: Tournament<TicTacToe> = Tournament::new(agents);
    tournament.play();
    tournament.leaderboard();
}

fn dots() {
    let mut agents: Vec<Box<dyn Agent<DotsAndBoxes>>> = Vec::new();
    for _ in 0..3 {
        agents.push(Box::new(Random {}));
    }
    agents.push(Box::new(Search::<DotsAndBoxes, ScoreEvaluator>::new(
        1000, 1.41, 1.0, 0.01, 0.0,
    )));
    let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(agents);
    tournament.play_multiplayer();
    tournament.leaderboard();
}

fn dots2() {
    let game = DotsAndBoxes::new();
    let agent: Search<DotsAndBoxes, TopsideEvaluator> = Search::new(10000, 1.41, 1.0, 0.5, 0.5);
    agent.run(game.clone());
}

fn wtf() {
    let mut game = TicTacToe::new();
    game.apply_action(Cell(0)); // X
    game.apply_action(Cell(1));
    game.apply_action(Cell(3)); // X
    game.apply_action(Cell(4));
    let agent = Search::<TicTacToe, ConstantEvaluator>::new(100, 1.41, 1.0, 0.1, 0.0);
    agent.run(game);
}

fn wtf2() {
    use DnbDir::*;
    let mut game = DotsAndBoxes::new();
    game.apply_action(DnbEdge(0, 0, N)); // A
    game.apply_action(DnbEdge(1, 0, N));
    game.apply_action(DnbEdge(2, 0, W));
    game.apply_action(DnbEdge(3, 0, W));

    game.apply_action(DnbEdge(1, 0, W)); // A
    game.apply_action(DnbEdge(1, 0, N));
    game.apply_action(DnbEdge(2, 0, W));
    game.apply_action(DnbEdge(3, 0, W));

    game.apply_action(DnbEdge(0, 1, N));
    game.apply_action(DnbEdge(2, 0, W));
    game.apply_action(DnbEdge(3, 0, W));
    game.apply_action(DnbEdge(3, 0, W));

    let agent = Search::<DotsAndBoxes, ScoreEvaluator>::new(1000, 1.41, 1.0, 0.01, 0.0);
    agent.run(game);
}

fn main() {
    tournament();
}
