use rand::{random_range, random_ratio};
use std::cmp::{max, min};

use crate::{GameState, Outcome, Player, agents::Evaluator};

const DENOMINATOR: u32 = 100;

#[derive(Debug, Clone)]
pub struct OddsGame {
    w: u32,
    l: u32,
    to_play: OddsGamePlayer,
    winner: Option<OddsGamePlayer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OddsGameAction(u32, u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OddsGamePlayer {
    A,
    B,
}
use OddsGamePlayer::*;

impl Player for OddsGamePlayer {
    fn list() -> Vec<Self> {
        vec![A, B]
    }
}

impl GameState for OddsGame {
    type Action = OddsGameAction;
    type Player = OddsGamePlayer;

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

    fn outcome(&self, player: Self::Player) -> Option<(Outcome, f64)> {
        match self.winner {
            None => None,
            Some(winner) if winner == player => Some((Outcome::Win, 1.0)),
            Some(_) => Some((Outcome::Loss, -1.0)),
        }
    }
}

pub struct OddsEvaluator {}
impl Evaluator<OddsGame> for OddsEvaluator {
    fn evaluate(&self, game_state: OddsGame) -> f64 {
        let v = game_state.w as f64 / (game_state.w + game_state.l) as f64;
        let v = v * (game_state.w + game_state.l) as f64 / DENOMINATOR as f64;
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
    fn evaluate(&self, game_state: OddsGame) -> f64 {
        let v = game_state.w as f64 / (game_state.w + game_state.l) as f64;
        let v = v * 2.0 - 1.0;
        let v = v * (game_state.w + game_state.l) as f64 / DENOMINATOR as f64;
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
    fn t1() {
        let mut g = OddsGame::new();
        g.w = 3;
        g.l = 1;
        let e1 = OddsEvaluator {};
        let e2 = NormalizedOddsEvaluator {};
        dbg!(e1.evaluate(g.clone()));
        dbg!(e2.evaluate(g));
    }

    #[test]
    fn t2() {
        let mut g = OddsGame::new();
        let a = g.get_actions(g.current_player());
        dbg!(&a);
        g.apply_action(a[0]);
        dbg!(g.w, g.l);
    }
}
