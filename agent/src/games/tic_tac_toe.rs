use std::hash::{Hash, Hasher};

use crate::{GameState, Outcome, Player};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TicTacToePlayer {
    X,
    O,
}

impl Player for TicTacToePlayer {
    fn list() -> Vec<TicTacToePlayer> {
        vec![TicTacToePlayer::X, TicTacToePlayer::O]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TicTacToeAction(u8); // board position 0-8

impl Hash for TicTacToeAction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[derive(Clone)]
pub struct TicTacToe {
    board: [Option<TicTacToePlayer>; 9],
    current_player: TicTacToePlayer,
}

impl TicTacToe {
    fn check_winner(&self) -> Option<TicTacToePlayer> {
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

impl GameState<TicTacToeAction, TicTacToePlayer> for TicTacToe {
    fn new() -> Self {
        TicTacToe {
            board: [None; 9],
            current_player: TicTacToePlayer::X,
        }
    }

    fn get_actions(&self, _player: TicTacToePlayer) -> Vec<TicTacToeAction> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| cell.is_none().then_some(TicTacToeAction(i as u8)))
            .collect()
    }

    fn current_player(&self) -> TicTacToePlayer {
        self.current_player
    }

    fn apply_action(&mut self, mv: TicTacToeAction) {
        self.board[mv.0 as usize] = Some(self.current_player);
        self.current_player = if self.current_player == TicTacToePlayer::X {
            TicTacToePlayer::O
        } else {
            TicTacToePlayer::X
        };
    }

    fn terminal_value(&self, player: TicTacToePlayer) -> Option<(Outcome, f64)> {
        if self.is_terminal() {
            if let Some(winner) = self.check_winner() {
                if winner == player {
                    Some((Outcome::Win, 1.0))
                } else {
                    Some((Outcome::Loss, 0.0))
                }
            } else {
                Some((Outcome::Draw, 0.5))
            }
        } else {
            None
        }
    }
    fn is_terminal(&self) -> bool {
        self.check_winner().is_some() || self.get_actions(self.current_player).is_empty()
    }
}
