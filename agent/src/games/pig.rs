use generic_array::typenum;
use std::fmt::Display;
use tch::Tensor;

use crate::GameState;
use common::{Image, Outcome, Player as PlayerTrait};

const WIN_SCORE: u32 = 50;

#[derive(Debug, Clone)]
pub struct Pig {
    scores: [u32; 2],
    turn_total: u32,
    to_play: Player,
    pending_roll: bool,
    winner: Option<Player>,
}

impl GameState for Pig {
    type Action = Action;
    type Player = Player;

    fn name() -> String {
        "Pig".to_string()
    }

    fn new() -> Self {
        Pig {
            scores: [0; 2],
            turn_total: 0,
            to_play: Player::P1,
            pending_roll: false,
            winner: None,
        }
    }

    fn apply_action(&mut self, action: Self::Action) {
        match action {
            Action::Roll => {
                self.pending_roll = true;
            }
            Action::Bank => {
                let idx: usize = self.to_play.into();
                self.scores[idx] += self.turn_total;
                self.turn_total = 0;
                if self.scores[idx] >= WIN_SCORE {
                    self.winner = Some(self.to_play);
                } else {
                    self.to_play = self.to_play.other();
                }
            }
            Action::Die(face) => {
                self.pending_roll = false;
                if face == 1 {
                    self.turn_total = 0;
                    self.to_play = self.to_play.other();
                } else {
                    self.turn_total += face as u32;
                }
            }
        }
    }

    fn current_player(&self) -> Self::Player {
        self.to_play
    }

    fn get_actions(&self, _: Self::Player) -> (Vec<Self::Action>, Option<Vec<f64>>) {
        if self.pending_roll {
            ((1..=6).map(Action::Die).collect(), Some(vec![1.0 / 6.0; 6]))
        } else if self.turn_total == 0 {
            (vec![Action::Roll], None)
        } else {
            (vec![Action::Roll, Action::Bank], None)
        }
    }

    fn is_random(&self) -> bool {
        self.pending_roll
    }

    fn is_terminal(&self) -> bool {
        self.winner.is_some()
    }

    fn outcome(&self, player: Self::Player) -> Option<(Outcome, f32)> {
        match self.winner {
            None => None,
            Some(winner) if winner == player => Some((Outcome::Win, 1.0)),
            Some(_) => Some((Outcome::Loss, -1.0)),
        }
    }

    fn pairwise_outcome(&self, player1: Self::Player, _: Self::Player) -> Option<(Outcome, f32)> {
        self.outcome(player1)
    }
}

impl Display for Pig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "P1: {} | P2: {} | Turn: {}",
            self.scores[0], self.scores[1], self.turn_total
        )
    }
}

impl Image for Pig {
    const IMAGE_SIZE: usize = 4;

    fn tensor_image(&self) -> Tensor {
        Tensor::from_slice(&[
            self.scores[0] as f32 / WIN_SCORE as f32,
            self.scores[1] as f32 / WIN_SCORE as f32,
            self.turn_total as f32 / WIN_SCORE as f32,
            if self.to_play == Player::P1 { 1.0 } else { 0.0 },
        ])
    }

    fn quantized_image(&self, _buffer: *mut i8) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Roll,
    Bank,
    Die(u8),
}

impl From<Action> for usize {
    fn from(val: Action) -> Self {
        match val {
            Action::Roll => 0,
            Action::Bank => 1,
            Action::Die(n) => n as usize + 1,
        }
    }
}

impl From<usize> for Action {
    fn from(val: usize) -> Self {
        match val {
            0 => Action::Roll,
            1 => Action::Bank,
            n => Action::Die((n - 1) as u8),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    fn other(self) -> Self {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        }
    }
}

impl PlayerTrait for Player {
    const LEN: usize = 2;
    type Len = typenum::U2;
    fn list() -> Vec<Self> {
        vec![Player::P1, Player::P2]
    }
}

impl From<Player> for usize {
    fn from(val: Player) -> Self {
        val as Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;

    #[test]
    fn bank_accumulates_and_switches_player() {
        let mut g = Pig::new();
        g.apply_action(Action::Roll);
        g.apply_action(Action::Die(5));
        g.apply_action(Action::Bank);
        assert_eq!(g.scores[0], 5);
        assert_eq!(g.to_play, Player::P2);
    }

    #[test]
    fn pig_on_one_loses_turn_total() {
        let mut g = Pig::new();
        g.apply_action(Action::Roll);
        g.apply_action(Action::Die(4));
        g.apply_action(Action::Roll);
        g.apply_action(Action::Die(1));
        assert_eq!(g.turn_total, 0);
        assert_eq!(g.scores[0], 0);
        assert_eq!(g.to_play, Player::P2);
    }

    #[test]
    fn first_to_100_wins() {
        let mut g = Pig::new();
        g.scores[0] = 94;
        g.apply_action(Action::Roll);
        g.apply_action(Action::Die(6));
        assert!(!g.is_terminal());
        g.apply_action(Action::Bank);
        assert!(g.is_terminal());
        assert!(matches!(g.outcome(Player::P1), Some((Outcome::Win, _))));
        assert!(matches!(g.outcome(Player::P2), Some((Outcome::Loss, _))));
    }

    #[test]
    fn image_size_is_accurate() {
        assert_eq!(
            Pig::new().tensor_image().size()[0] as usize,
            Pig::IMAGE_SIZE
        )
    }
}
