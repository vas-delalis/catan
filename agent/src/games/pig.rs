use std::fmt::Display;
use tch::Tensor;

use crate::GameState;
use common::{Image, Outcome, Player as PlayerTrait};

const WIN_SCORE: u32 = 50;

#[derive(Debug, Clone)]
pub struct Pig {
    scores: [u32; 2],
    turn_total: u32,
    to_play: PigPlayer,
    pending_roll: bool,
    winner: Option<PigPlayer>,
}

impl GameState for Pig {
    type Action = PigAction;
    type Player = PigPlayer;

    fn name() -> String {
        "Pig".to_string()
    }

    fn new() -> Self {
        Pig {
            scores: [0; 2],
            turn_total: 0,
            to_play: PigPlayer::P1,
            pending_roll: false,
            winner: None,
        }
    }

    fn apply_action(&mut self, action: Self::Action) {
        match action {
            PigAction::Roll => {
                self.pending_roll = true;
            }
            PigAction::Bank => {
                let idx: usize = self.to_play.into();
                self.scores[idx] += self.turn_total;
                self.turn_total = 0;
                if self.scores[idx] >= WIN_SCORE {
                    self.winner = Some(self.to_play);
                } else {
                    self.to_play = self.to_play.other();
                }
            }
            PigAction::Die(face) => {
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
            (
                (1..=6).map(PigAction::Die).collect(),
                Some(vec![1.0 / 6.0; 6]),
            )
        } else if self.turn_total == 0 {
            (vec![PigAction::Roll], None)
        } else {
            (vec![PigAction::Roll, PigAction::Bank], None)
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
    const IMAGE_SIZE: i64 = 5;

    fn image(&self, arbiter: PigPlayer) -> Tensor {
        Tensor::from_slice(&[
            self.scores[0] as f32 / WIN_SCORE as f32,
            self.scores[1] as f32 / WIN_SCORE as f32,
            self.turn_total as f32 / WIN_SCORE as f32,
            if self.to_play == PigPlayer::P1 {
                1.0
            } else {
                0.0
            },
            if arbiter == PigPlayer::P1 { 1.0 } else { 0.0 },
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PigAction {
    Roll,
    Bank,
    Die(u8),
}

impl Into<usize> for PigAction {
    fn into(self) -> usize {
        match self {
            PigAction::Roll => 0,
            PigAction::Bank => 1,
            PigAction::Die(n) => n as usize + 1,
        }
    }
}

impl From<usize> for PigAction {
    fn from(val: usize) -> Self {
        match val {
            0 => PigAction::Roll,
            1 => PigAction::Bank,
            n => PigAction::Die((n - 1) as u8),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PigPlayer {
    P1,
    P2,
}

impl PigPlayer {
    fn other(self) -> Self {
        match self {
            PigPlayer::P1 => PigPlayer::P2,
            PigPlayer::P2 => PigPlayer::P1,
        }
    }
}

impl PlayerTrait for PigPlayer {
    const LEN: usize = 2;
    fn list() -> Vec<Self> {
        vec![PigPlayer::P1, PigPlayer::P2]
    }
}

impl Into<usize> for PigPlayer {
    fn into(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;

    #[test]
    fn bank_accumulates_and_switches_player() {
        let mut g = Pig::new();
        g.apply_action(PigAction::Roll);
        g.apply_action(PigAction::Die(5));
        g.apply_action(PigAction::Bank);
        assert_eq!(g.scores[0], 5);
        assert_eq!(g.to_play, PigPlayer::P2);
    }

    #[test]
    fn pig_on_one_loses_turn_total() {
        let mut g = Pig::new();
        g.apply_action(PigAction::Roll);
        g.apply_action(PigAction::Die(4));
        g.apply_action(PigAction::Roll);
        g.apply_action(PigAction::Die(1));
        assert_eq!(g.turn_total, 0);
        assert_eq!(g.scores[0], 0);
        assert_eq!(g.to_play, PigPlayer::P2);
    }

    #[test]
    fn first_to_100_wins() {
        let mut g = Pig::new();
        g.scores[0] = 94;
        g.apply_action(PigAction::Roll);
        g.apply_action(PigAction::Die(6));
        assert!(!g.is_terminal());
        g.apply_action(PigAction::Bank);
        assert!(g.is_terminal());
        assert!(matches!(g.outcome(PigPlayer::P1), Some((Outcome::Win, _))));
        assert!(matches!(g.outcome(PigPlayer::P2), Some((Outcome::Loss, _))));
    }

    #[test]
    fn image_size_is_accurate() {
        assert_eq!(Pig::new().image(PigPlayer::P1).size()[0], Pig::IMAGE_SIZE)
    }
}
