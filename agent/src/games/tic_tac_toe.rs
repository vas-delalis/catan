use std::fmt;

use common::{Image, Outcome, Player};
use tch::Tensor;

use crate::{Agent, GameState};

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

    fn get_actions(&self, _player: TicTacToePlayer) -> (Vec<Cell>, Option<Vec<f64>>) {
        (
            self.board
                .iter()
                .enumerate()
                .filter_map(|(i, cell)| cell.is_none().then_some(Cell(i as u8)))
                .collect(),
            None,
        )
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
        self.check_winner().is_some() || self.get_actions(self.current_player).0.is_empty()
    }

    fn is_random(&self) -> bool {
        false
    }
}

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, cell) in self.board.iter().enumerate() {
            let c = match cell {
                Some(TicTacToePlayer::X) => 'X',
                Some(TicTacToePlayer::O) => 'O',
                None => '.',
            };
            write!(f, "{}", c)?;
            if i % 3 == 2 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Image for TicTacToe {
    const IMAGE_SIZE: i64 = 20;

    fn image(&self, arbiter: Self::Player) -> tch::Tensor {
        let mut exes: Vec<f32> = vec![];
        let mut ohs: Vec<f32> = vec![];

        for tile in self.board {
            match tile {
                Some(p) => {
                    if p == TicTacToePlayer::X {
                        exes.push(1.0);
                        ohs.push(0.0);
                    } else {
                        exes.push(0.0);
                        ohs.push(1.0);
                    }
                }
                None => {
                    exes.push(0.0);
                    ohs.push(0.0);
                }
            }
        }

        let curr: Vec<f32> = if self.current_player() == TicTacToePlayer::X {
            vec![1.0]
        } else {
            vec![0.0]
        };
        let arb: Vec<f32> = if arbiter == TicTacToePlayer::X {
            vec![1.0]
        } else {
            vec![0.0]
        };

        Tensor::from_slice(&[exes, ohs, curr, arb].concat())
    }
}

pub struct Optimal;

impl Optimal {
    const LINES: [[usize; 3]; 8] = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ];
    const CORNERS: [usize; 4] = [0, 2, 6, 8];
    const SIDES: [usize; 4] = [1, 3, 5, 7];

    fn other(player: TicTacToePlayer) -> TicTacToePlayer {
        match player {
            TicTacToePlayer::X => TicTacToePlayer::O,
            TicTacToePlayer::O => TicTacToePlayer::X,
        }
    }

    // Returns distinct cells that would complete a win for `player` in the given board.
    fn threat_cells(board: &[Option<TicTacToePlayer>; 9], player: TicTacToePlayer) -> Vec<usize> {
        let mut cells: Vec<usize> = vec![];
        for line in &Self::LINES {
            let marks = line.iter().filter(|&&i| board[i] == Some(player)).count();
            let empty: Vec<usize> = line
                .iter()
                .filter(|&&i| board[i].is_none())
                .cloned()
                .collect();
            if marks == 2 && empty.len() == 1 {
                let cell = empty[0];
                if !cells.contains(&cell) {
                    cells.push(cell);
                }
            }
        }
        cells
    }

    // Returns cells where placing `player` would create two simultaneous threats (a fork).
    fn fork_moves(board: &[Option<TicTacToePlayer>; 9], player: TicTacToePlayer) -> Vec<usize> {
        (0..9)
            .filter(|&cell| {
                if board[cell].is_some() {
                    return false;
                }
                let mut sim = *board;
                sim[cell] = Some(player);
                Self::threat_cells(&sim, player).len() >= 2
            })
            .collect()
    }
}

impl Agent<TicTacToe> for Optimal {
    fn get_action(&self, state: TicTacToe) -> Cell {
        let board = &state.board;
        let player = state.current_player();
        let opponent = Self::other(player);

        // 1. Win
        if let Some(&cell) = Self::threat_cells(board, player).first() {
            return Cell(cell as u8);
        }

        // 2. Block opponent's win
        if let Some(&cell) = Self::threat_cells(board, opponent).first() {
            return Cell(cell as u8);
        }

        // 3. Fork
        if let Some(&cell) = Self::fork_moves(board, player).first() {
            return Cell(cell as u8);
        }

        // 4. Block opponent's fork
        let opp_forks = Self::fork_moves(board, opponent);
        if !opp_forks.is_empty() {
            if opp_forks.len() == 1 {
                return Cell(opp_forks[0] as u8);
            }

            // Multiple opponent forks: prefer a two-in-a-row that forces the opponent
            // to block their own fork cell (and leaves them with no fork after blocking).
            for cell in 0..9usize {
                if board[cell].is_some() {
                    continue;
                }
                let mut sim = *board;
                sim[cell] = Some(player);
                let threats = Self::threat_cells(&sim, player);
                if threats.len() == 1 {
                    let forced = threats[0];
                    if opp_forks.contains(&forced) {
                        let mut sim2 = sim;
                        sim2[forced] = Some(opponent);
                        if Self::fork_moves(&sim2, opponent).is_empty() {
                            return Cell(cell as u8);
                        }
                    }
                }
            }

            // Otherwise, any two-in-a-row whose forced response leaves opponent fork-free.
            for cell in 0..9usize {
                if board[cell].is_some() {
                    continue;
                }
                let mut sim = *board;
                sim[cell] = Some(player);
                let threats = Self::threat_cells(&sim, player);
                if threats.len() == 1 {
                    let forced = threats[0];
                    let mut sim2 = sim;
                    sim2[forced] = Some(opponent);
                    if Self::fork_moves(&sim2, opponent).is_empty() {
                        return Cell(cell as u8);
                    }
                }
            }

            // Fallback: block any fork.
            return Cell(opp_forks[0] as u8);
        }

        // 5. Center
        if board[4].is_none() {
            return Cell(4);
        }

        // 6. Opposite corner
        for &(a, b) in &[(0, 8), (2, 6)] {
            if board[a] == Some(opponent) && board[b].is_none() {
                return Cell(b as u8);
            }
            if board[b] == Some(opponent) && board[a].is_none() {
                return Cell(a as u8);
            }
        }

        // 7. Empty corner
        for &corner in &Self::CORNERS {
            if board[corner].is_none() {
                return Cell(corner as u8);
            }
        }

        // 8. Empty side
        for &side in &Self::SIDES {
            if board[side].is_none() {
                return Cell(side as u8);
            }
        }

        unreachable!("no valid move")
    }
}
