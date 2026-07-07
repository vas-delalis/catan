use common::{Evaluation, Image, Outcome, Player as PlayerTrait};
use generic_array::{GenericArray, typenum};
use rand::random_range;
use std::{
    cmp::{max, min},
    fmt::Display,
};
use tch::Tensor;

use crate::GameState;

const DENOMINATOR: u32 = 100;

#[derive(Debug, Clone)]
pub struct OddsGame {
    w: u32,
    l: u32,
    to_play: Player,
    roll: bool,
    winner: Option<Player>,
}

fn denom(w: u32, l: u32) -> u32 {
    max(w + l, DENOMINATOR)
}

impl GameState for OddsGame {
    type Action = Action;
    type Player = Player;

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
            Action::Choose(w, l) => {
                self.w = w;
                self.l = l;
                self.roll = true;
                self.to_play = if self.to_play == A { B } else { A };
            }
            Action::Roll(opt) => match opt {
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
                    Action::Roll(Some(A)),
                    Action::Roll(Some(B)),
                    Action::Roll(None),
                ],
                Some(vec![w / denom, l / denom, (denom - w - l) / denom]),
            )
        } else {
            let mut actions = vec![];
            for _ in 0..8 {
                actions.push(Action::Choose(
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

    fn scores(&self) -> Option<Evaluation<Self>> {
        self.winner.map(|winner| {
            GenericArray::from_iter(
                Player::list()
                    .into_iter()
                    .map(|p| if p == winner { 1.0 } else { -1.0 }),
            )
        })
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
    const IMAGE_SIZE: usize = 3;

    fn tensor_image(&self) -> tch::Tensor {
        Tensor::from_slice(&[
            self.w as f32 / denom(self.w, self.l) as f32,
            self.l as f32 / denom(self.w, self.l) as f32,
            if self.current_player() == A { 1.0 } else { 0.0 },
        ])
    }

    fn quantized_image(&self, _buffer: *mut i8) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Choose(u32, u32),
    Roll(Option<Player>),
}

impl From<usize> for Action {
    fn from(val: usize) -> Self {
        if val > 2 {
            Action::Choose((val >> 32) as u32, val as u32)
        } else if val == 2 {
            Action::Roll(Some(B))
        } else if val == 1 {
            Action::Roll(Some(A))
        } else {
            Action::Roll(None)
        }
    }
}

impl From<Action> for usize {
    fn from(val: Action) -> Self {
        match val {
            Action::Choose(w, l) => ((w as usize) << 32) | (l as usize),
            Action::Roll(opt) => match opt {
                Some(A) => 1,
                Some(B) => 2,
                None => 0,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Player {
    A,
    B,
}
use Player::*;

impl PlayerTrait for Player {
    const LEN: usize = 2;
    type Len = typenum::U2;
    fn list() -> Vec<Self> {
        vec![A, B]
    }
}

impl From<Player> for usize {
    fn from(val: Player) -> Self {
        val as usize
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
