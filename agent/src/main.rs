use agent::{GameState, Search};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    X,
    O,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Move(u8); // board position 0-8

impl Hash for Move {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[derive(Clone)]
struct TicTacToe {
    board: [Option<Player>; 9],
    current_player: Player,
}

impl TicTacToe {
    fn new() -> Self {
        TicTacToe {
            board: [None; 9],
            current_player: Player::X,
        }
    }

    fn check_winner(&self) -> Option<Player> {
        let lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for [a, b, c] in &lines {
            if self.board[*a].is_some()
                && self.board[*a] == self.board[*b]
                && self.board[*b] == self.board[*c]
            {
                return self.board[*a];
            }
        }
        None
    }
}

impl agent::GameState<Move, Player> for TicTacToe {
    fn get_actions(&self, _player: Player) -> Vec<Move> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| cell.is_none().then_some(Move(i as u8)))
            .collect()
    }

    fn current_player(&self) -> Player {
        self.current_player
    }

    fn apply_action(&mut self, mv: Move) {
        self.board[mv.0 as usize] = Some(self.current_player);
        self.current_player = if self.current_player == Player::X {
            Player::O
        } else {
            Player::X
        };
    }

    fn terminal_value(&self, player: Player) -> Option<f64> {
        if self.check_winner().is_some() {
            return Some(1.0);
        }
        if self.get_actions(player).is_empty() {
            return Some(0.5);
        }
        None
    }
    fn is_terminal(&self) -> bool {
        self.check_winner().is_some() || self.get_actions(self.current_player).is_empty()
    }
}

fn main() {
    let search1: Search<Move, Player> = Search::new(1, 1.41, 1.0, 1.0);
    let search2: Search<Move, Player> = Search::new(1000, 1.41, 1.0, 1.0);

    let game_count = 400;
    let mut x_victories = 0;
    let mut draws = 0;

    for _ in 0..game_count {
        let mut game = TicTacToe::new();
        while !game.is_terminal() {
            let scratch = game.clone();
            let action = if game.current_player == Player::X {
                search1.run(scratch)
            } else {
                search2.run(scratch)
            };
            game.apply_action(action);
        }
        match game.check_winner() {
            Some(Player::X) => x_victories += 1,
            None => draws += 1,
            _ => {}
        }
    }

    println!("X wins: {:.1}%", 100 * x_victories / game_count);
    println!("Draws: {:.1}%", 100 * draws / game_count);
    println!(
        "O wins: {:.1}%",
        100 * (game_count - x_victories - draws) / game_count
    );
}
