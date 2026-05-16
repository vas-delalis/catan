

use tch::Tensor;

use crate::{GameState, Outcome, Player, ml::Image};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TicTacToePlayer {
    X,
    O,
}

impl Player for TicTacToePlayer {
    const LEN: usize = 2;
    fn list() -> Vec<TicTacToePlayer> {
        vec![TicTacToePlayer::X, TicTacToePlayer::O]
    }
}

impl Into<usize> for TicTacToePlayer {
    fn into(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell(pub u8); // board position 0-8

impl Into<usize> for Cell {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for Cell {
    fn from(val: usize) -> Self {
        Cell(val as u8)
    }
}


#[derive(Debug, Clone)]
pub struct TicTacToe {
    pub board: [Option<TicTacToePlayer>; 9],
    current_player: TicTacToePlayer,
}

impl TicTacToe {
    pub fn check_winner(&self) -> Option<TicTacToePlayer> {
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

impl GameState for TicTacToe {
    type Action = Cell;
    type Player = TicTacToePlayer;

    fn name() -> String {
        String::from("TicTacToe")
    }

    fn new() -> Self {
        TicTacToe {
            board: [None; 9],
            current_player: TicTacToePlayer::X,
        }
    }

    fn get_actions(&self, _player: TicTacToePlayer) -> Vec<Cell> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| cell.is_none().then_some(Cell(i as u8)))
            .collect()
    }

    fn current_player(&self) -> TicTacToePlayer {
        self.current_player
    }

    fn apply_action(&mut self, mv: Cell) {
        self.board[mv.0 as usize] = Some(self.current_player);
        self.current_player = if self.current_player == TicTacToePlayer::X {
            TicTacToePlayer::O
        } else {
            TicTacToePlayer::X
        };
    }

    fn outcome(&self, player: TicTacToePlayer) -> Option<(Outcome, f32)> {
        if self.is_terminal() {
            if let Some(winner) = self.check_winner() {
                if winner == player {
                    Some((Outcome::Win, 1.0))
                } else {
                    Some((Outcome::Loss, -1.0))
                }
            } else {
                Some((Outcome::Draw, 0.0))
            }
        } else {
            None
        }
    }

    fn pairwise_outcome(&self, player1: Self::Player, _: Self::Player) -> Option<(Outcome, f32)> {
        self.outcome(player1)
    }

    fn is_terminal(&self) -> bool {
        self.check_winner().is_some() || self.get_actions(self.current_player).is_empty()
    }
}

impl Image for TicTacToe {
    const IMAGE_SIZE: i64 = 19;

    fn image(&self) -> tch::Tensor {
        let mut plane1: Vec<f32> = vec![];
        let mut plane2: Vec<f32> = vec![];
        let plane3: Vec<f32> = if self.current_player() == TicTacToePlayer::X {
            vec![1.0; 1]
        } else {
            vec![0.0; 1]
        };

        for tile in self.board {
            match tile {
                Some(p) => {
                    if p == TicTacToePlayer::X {
                        plane1.push(1.0);
                        plane2.push(0.0);
                    } else {
                        plane1.push(0.0);
                        plane2.push(1.0);
                    }
                }
                None => {
                    plane1.push(0.0);
                    plane2.push(0.0);
                }
            }
        }
        let plane1 = Tensor::from_slice(&plane1);
        let plane2 = Tensor::from_slice(&plane2);
        let plane3 = Tensor::from_slice(&plane3); //.reshape([3, 3]);

        Tensor::cat(&[plane1, plane2, plane3], 0)
    }
}
