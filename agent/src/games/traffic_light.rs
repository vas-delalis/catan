use tch::Tensor;

use crate::{GameState, Outcome, Player, ml::Image};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum A {
    Roll,
    Roll2,
}

impl Player for P {
    fn list() -> Vec<Self> {
        vec![P::A, P::B]
    }
}

#[derive(Clone)]
pub struct TrafficLight {
    points1: i32,
    points2: i32,
    to_play: P,
    prev: P,
}

impl GameState for TrafficLight {
    const NAME: &str = "TrafficLight";
    type Player = P;
    type Action = A;

    fn new() -> Self {
        TrafficLight {
            points1: 0,
            points2: 0,
            to_play: P::A,
            prev: P::B,
        }
    }

    fn current_player(&self) -> Self::Player {
        self.to_play
    }

    fn prev_player(&self) -> Self::Player {
        self.prev
    }

    fn get_actions(&self, _: Self::Player) -> Vec<Self::Action> {
        vec![A::Roll, A::Roll2]
    }

    fn apply_action(&mut self, _: Self::Action) {
        let points = 1;
        if self.to_play == P::A {
            self.points1 += points;
        } else {
            self.points2 += points;
        }
        self.prev = self.to_play;
        self.to_play = if rand::random_bool(0.5) { P::A } else { P::B };
    }

    fn is_terminal(&self) -> bool {
        self.points1 >= 10 || self.points2 >= 10
    }

    fn outcome(&self, player: Self::Player) -> Option<(crate::Outcome, f64)> {
        if !self.is_terminal() {
            return None;
        }
        if self.points1 == self.points2 {
            Some((Outcome::Draw, 0.0))
        } else if self.points1 > self.points2 {
            if player == P::A {
                Some((Outcome::Win, 1.0))
            } else {
                Some((Outcome::Loss, -1.0))
            }
        } else {
            if player == P::B {
                Some((Outcome::Win, 1.0))
            } else {
                Some((Outcome::Loss, -1.0))
            }
        }
    }

    fn pairwise_outcome(&self, player1: Self::Player, _: Self::Player) -> Option<(Outcome, f64)> {
        self.outcome(player1)
    }
}

impl Image for TrafficLight {
    const IMAGE_SIZE: i64 = 6;

    fn image(&self) -> tch::Tensor {
        let mut t = [
            self.points1 as f32 / 10.0,
            self.points2 as f32 / 10.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        if self.prev == P::A {
            t[2] = 1.0;
        } else {
            t[3] = 1.0;
        };
        if self.to_play == P::A {
            t[4] = 1.0;
        } else {
            t[5] = 1.0;
        };
        Tensor::from_slice(&t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game() {
        let mut g = TrafficLight::new();
        while !g.is_terminal() {
            g.apply_action(g.get_actions(P::A)[0]);
        }
    }
}
