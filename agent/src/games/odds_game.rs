use rand::{random_range, random_ratio};
use std::cmp::{max, min};
use tch::Tensor;

use crate::{GameState, Outcome, Player, agents::Evaluator, ml::Image};

const DENOMINATOR: u32 = 100;

#[derive(Debug, Clone)]
pub struct OddsGame {
    w: u32,
    l: u32,
    to_play: OddsGamePlayer,
    winner: Option<OddsGamePlayer>,
}

#[derive(Debug, Clone, Copy)]
pub struct OddsGameAction(u32, u32);

impl From<usize> for OddsGameAction {
    fn from(val: usize) -> Self {
        OddsGameAction((val >> 32) as u32, val as u32)
    }
}

impl Into<usize> for OddsGameAction {
    fn into(self) -> usize {
        ((self.0 as usize) << 32) | (self.1 as usize)
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
            winner: None,
        }
    }

    fn apply_action(&mut self, action: Self::Action) {
        if random_ratio(self.w + self.l, max(DENOMINATOR, self.w + self.l)) {
            if random_ratio(self.w, self.w + self.l) {
                self.winner = Some(A);
            } else {
                self.winner = Some(B);
            }
        }

        self.w = action.0;
        self.l = action.1;
        self.to_play = if self.to_play == A { B } else { A };
    }

    fn current_player(&self) -> Self::Player {
        self.to_play
    }

    fn get_actions(&self, player: Self::Player) -> Vec<Self::Action> {
        assert!(player == self.to_play);
        let mut actions = vec![];
        for _ in 0..3 {
            actions.push(OddsGameAction(
                min(self.w + random_range(0..=10), DENOMINATOR),
                min(self.l + random_range(0..=10), DENOMINATOR),
            ));
        }
        actions
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

impl Image for OddsGame {
    const IMAGE_SIZE: i64 = 3;
    fn image(&self) -> tch::Tensor {
        Tensor::from_slice(&[
            self.w as f32,
            self.l as f32,
            if self.current_player() == A { 1.0 } else { 0.0 },
        ])
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
        let a = g.get_actions(g.current_player());
        dbg!(&a);
        g.apply_action(a[0]);
        dbg!(g.w, g.l);
    }
}
