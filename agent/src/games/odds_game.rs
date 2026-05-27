use rand::random_range;
use std::{
    cmp::{max, min},
    fmt::Display,
};
use tch::Tensor;

use crate::{GameState, Outcome, Player, agents::Evaluator, ml::Image};

const DENOMINATOR: u32 = 100;

#[derive(Debug, Clone)]
pub struct OddsGame {
    w: u32,
    l: u32,
    to_play: OddsGamePlayer,
    roll: bool,
    winner: Option<OddsGamePlayer>,
}

fn denom(w: u32, l: u32) -> u32 {
    max(w + l, DENOMINATOR)
}

impl GameState for OddsGame {
    type Action = OddsGameAction;
    type Player = OddsGamePlayer;

    fn name() -> String {
        "OddsGame".to_string()
    }

    fn new() -> Self {
        OddsGame {
            w: 1,
            l: 1,
            to_play: A,
            roll: false,
            winner: None,
        }
    }

    fn apply_action(&mut self, action: Self::Action) {
        match action {
            OddsGameAction::Choose(w, l) => {
                self.w = w;
                self.l = l;
                self.roll = true;
                self.to_play = if self.to_play == A { B } else { A };
            }
            OddsGameAction::Roll(opt) => match opt {
                Some(A) => self.winner = Some(A),
                Some(B) => self.winner = Some(B),
                _ => self.roll = false,
            },
        }
    }

    fn current_player(&self) -> Self::Player {
        self.to_play
    }

    fn get_actions(&self, _: Self::Player) -> (Vec<Self::Action>, Option<Vec<f64>>) {
        if self.roll {
            let denom = denom(self.w, self.l) as f64;
            let w = self.w as f64;
            let l = self.l as f64;
            (
                vec![
                    OddsGameAction::Roll(Some(A)),
                    OddsGameAction::Roll(Some(B)),
                    OddsGameAction::Roll(None),
                ],
                Some(vec![w / denom, l / denom, (denom - w - l) / denom]),
            )
        } else {
            let mut actions = vec![];
            for _ in 0..8 {
                actions.push(OddsGameAction::Choose(
                    min(random_range(0..=100), DENOMINATOR),
                    min(random_range(0..=100), DENOMINATOR),
                ));
            }
            (actions, None)
        }
    }

    fn is_random(&self) -> bool {
        self.roll
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

impl Display for OddsGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.w, self.l)
    }
}

impl Image for OddsGame {
    const IMAGE_SIZE: i64 = 4;

    fn image(&self, arbiter: OddsGamePlayer) -> tch::Tensor {
        Tensor::from_slice(&[
            self.w as f32 / denom(self.w, self.l) as f32,
            self.l as f32 / denom(self.w, self.l) as f32,
            if self.current_player() == A { 1.0 } else { 0.0 },
            if arbiter == A { 1.0 } else { 0.0 },
        ])
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OddsGameAction {
    Choose(u32, u32),
    Roll(Option<OddsGamePlayer>),
}

impl From<usize> for OddsGameAction {
    fn from(val: usize) -> Self {
        if val > 2 {
            OddsGameAction::Choose((val >> 32) as u32, val as u32)
        } else if val == 2 {
            OddsGameAction::Roll(Some(B))
        } else if val == 1 {
            OddsGameAction::Roll(Some(A))
        } else {
            OddsGameAction::Roll(None)
        }
    }
}

impl Into<usize> for OddsGameAction {
    fn into(self) -> usize {
        match self {
            OddsGameAction::Choose(w, l) => ((w as usize) << 32) | (l as usize),
            OddsGameAction::Roll(opt) => match opt {
                Some(A) => 1,
                Some(B) => 2,
                None => 0,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OddsGamePlayer {
    A,
    B,
}
use OddsGamePlayer::*;

impl Player for OddsGamePlayer {
    const LEN: usize = 2;
    fn list() -> Vec<Self> {
        vec![A, B]
    }
}

impl Into<usize> for OddsGamePlayer {
    fn into(self) -> usize {
        self as usize
    }
}

pub struct OddsEvaluator {}
impl Evaluator<OddsGame> for OddsEvaluator {
    fn evaluate(&self, game_state: &OddsGame, _: OddsGamePlayer) -> f32 {
        let v = game_state.w as f32 / (game_state.w + game_state.l) as f32;
        let v = v * (game_state.w + game_state.l) as f32 / DENOMINATOR as f32;
        let v = v * 2.0 - 1.0;
        if game_state.current_player() == A {
            -v
        } else {
            v
        }
    }
}

pub struct NormalizedOddsEvaluator {}
impl Evaluator<OddsGame> for NormalizedOddsEvaluator {
    fn evaluate(&self, game_state: &OddsGame, _: OddsGamePlayer) -> f32 {
        let v = game_state.w as f32 / (game_state.w + game_state.l) as f32;
        let v = v * 2.0 - 1.0;
        let v = v * (game_state.w + game_state.l) as f32 / DENOMINATOR as f32;
        if game_state.current_player() == A {
            -v
        } else {
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t2() {
        let mut g = OddsGame::new();
        let (a, _) = g.get_actions(g.current_player());
        dbg!(&a);
        g.apply_action(a[0]);
        dbg!(g.w, g.l);
    }
}
