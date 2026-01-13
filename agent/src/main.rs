use agent::{
    Agent, GameState, Human, Player, Search, Tournament,
    games::{DotsAndBoxes, DotsAndBoxesPlayer, TicTacToe, TicTacToeAction, TicTacToePlayer},
};

fn human_vs_human() {
    let mut game = DotsAndBoxes::new();
    let agent = Human {};
    while !game.is_terminal() {
        let scratch = game.clone();
        game.apply_action(agent.get_action(scratch));
    }
    for p in DotsAndBoxesPlayer::list() {
        println!("{:?} {:?}", p, game.terminal_value(p).unwrap())
    }
}

fn tournament() {
    let mut agents: Vec<Box<dyn Agent<TicTacToeAction, TicTacToePlayer, TicTacToe>>> = Vec::new();
    for value in [0.0, 0.5, 1.0] {
        for evals in [10, 100, 1000] {
            for alpha in [0.5] {
                agents.push(Box::new(Search::new(evals, 1.41, 1.0, alpha, value)));
            }
        }
    }
    // agents.push(Box::new(Random {}));
    let mut tournament: Tournament<TicTacToeAction, TicTacToePlayer, TicTacToe> =
        Tournament::new(agents);
    tournament.play(100000);
    tournament.leaderboard();
}

fn main() {
    human_vs_human();
}
