use std::{collections::HashSet, fmt::Display, hash::Hash};

use crate::{GameState, Outcome, Player as PlayerTrait, agents::Evaluator, ml::Image};

#[derive(Clone)]
pub struct DotsAndBoxes<const R: usize, const C: usize> {
    pub board: HashSet<Edge>,
    current_player: Player,
    pub score: [usize; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    A,
    B,
    C,
    D,
}

impl PlayerTrait for Player {
    const LEN: usize = 4;
    fn list() -> Vec<Player> {
        use Player::*;
        vec![A, B, C, D]
    }
}

impl Into<usize> for Player {
    fn into(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir {
    N,
    W,
}

use Dir::*;
use tch::Tensor;

pub struct Box(i8, i8);

impl Box {
    fn edges(&self) -> Vec<Edge> {
        let Box(x, y) = *self;
        vec![
            Edge(x, y, N),
            Edge(x, y, W),
            Edge(x, y + 1, N),
            Edge(x + 1, y, W),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge(pub i8, pub i8, pub Dir);

impl Edge {
    fn boxes(&self) -> [Box; 2] {
        let Edge(x, y, _) = *self;
        match self.2 {
            N => [Box(x, y - 1), Box(x, y)],
            W => [Box(x - 1, y), Box(x, y)],
        }
    }
}

impl<const R: usize, const C: usize> DotsAndBoxes<R, C> {
    const N_EDGES: usize = 2 * R * C + R + C;

    fn in_bounds(&self, e: &Edge) -> bool {
        let Edge(x, y, dir) = *e;
        let w = C as i8;
        let h = R as i8;
        let bound = match dir {
            Dir::N => x < w && y <= h,
            Dir::W => x <= w && y < h,
        };
        bound && 0 <= x && 0 <= y
    }

    fn winners(&self) -> Option<Vec<Player>> {
        if self.is_terminal() {
            let max = *self.score.iter().max().unwrap();
            let players = Player::list();
            let winners = (0..players.len())
                .filter(|&i| self.score[i] == max)
                .map(|i| players[i]);
            return Some(winners.collect());
        }
        None
    }
}

impl<const R: usize, const C: usize> GameState for DotsAndBoxes<R, C> {
    type Action = Edge;
    type Player = Player;

    fn name() -> String {
        format!("DotsAndBoxes{}x{}", R, C)
    }

    fn new() -> Self {
        DotsAndBoxes {
            board: HashSet::with_capacity(Self::N_EDGES),
            current_player: Player::A,
            score: [0, 0, 0, 0],
        }
    }

    fn get_actions(&self, _player: Player) -> Vec<Edge> {
        let mut edges: Vec<Edge> = vec![];
        for x in -1..=(C as i8 + 1) {
            for y in -1..=(R as i8 + 1) {
                for dir in [N, W] {
                    let e = Edge(x as i8, y as i8, dir);
                    if self.in_bounds(&e) && !self.board.contains(&e) {
                        edges.push(e);
                    }
                }
            }
        }
        edges
    }

    fn current_player(&self) -> Player {
        self.current_player
    }

    fn apply_action(&mut self, e: Edge) {
        self.board.insert(e.clone());
        let [box1, box2] = e.boxes();
        let boxes = [box1.edges(), box2.edges()];
        let mut end_turn = true;
        for b in boxes {
            if b.into_iter()
                .map(|e| self.board.get(&e))
                .all(|o| o.is_some())
            {
                self.score[self.current_player as usize] += 1;
                end_turn = false;
            }
        }
        if end_turn {
            let players = Player::list();
            self.current_player = players[(self.current_player as usize + 1) % players.len()];
        }
    }

    fn outcome(&self, player: Player) -> Option<(Outcome, f32)> {
        use Outcome::*;
        if !self.is_terminal() {
            return None;
        }
        let winners = self.winners().unwrap();
        if winners.contains(&player) {
            if winners.len() > 1 {
                return Some((Draw, 1.0 / winners.len() as f32));
            }
            return Some((Win, 1.0));
        } else {
            return Some((Loss, -0.3333));
        }
    }

    fn pairwise_outcome(&self, player1: Player, player2: Player) -> Option<(Outcome, f32)> {
        use Outcome::*;
        if self.is_terminal() {
            let winners = self.winners().unwrap();
            let p1_win = winners.contains(&player1);
            let p2_win = winners.contains(&player2);
            if p1_win == p2_win {
                return Some((Draw, 0.0));
            }
            if p1_win {
                return Some((Win, 1.0));
            }
            return Some((Loss, -1.0));
        }
        None
    }

    fn is_terminal(&self) -> bool {
        self.board.len() == Self::N_EDGES
    }
}

impl<const R: usize, const C: usize> Display for DotsAndBoxes<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut grid = String::new();
        let h = R as i8;
        let w = C as i8;

        for y in 0..=h {
            // Dot row with horizontal edges
            for x in 0..w {
                grid.push('.');
                let e = Edge(x, y, N);
                if let Some(_) = self.board.get(&e) {
                    grid.push('-');
                    grid.push('-');
                    grid.push('-');
                } else {
                    grid.push_str("   ");
                }
            }
            grid.push('.');
            grid.push('\n');

            // Vertical edges and box interiors (except after last dot row)
            if y < h {
                for x in 0..=w {
                    let e = Edge(x, y, W);
                    if let Some(_) = self.board.get(&e) {
                        grid.push('|');
                    } else {
                        grid.push(' ');
                    }

                    if x < w {
                        grid.push_str("   ");
                    }
                }
                grid.push('\n');
            }
        }

        write!(f, "{}", grid)
    }
}

impl<const R: usize, const C: usize> Image for DotsAndBoxes<R, C> {
    const IMAGE_SIZE: i64 = Self::N_EDGES as i64 + 8;

    fn image(&self) -> tch::Tensor {
        let mut planes = vec![0f32; Self::N_EDGES];

        let mut count = 0;
        for x in -1..=(C as i8 + 1) {
            for y in -1..=(R as i8 + 1) {
                for dir in [N, W] {
                    let e = Edge(x as i8, y as i8, dir);
                    if self.in_bounds(&e) {
                        if self.board.contains(&e) {
                            planes[count] = 1.0;
                        }
                        count += 1;
                    }
                }
            }
        }
        let mut score = vec![0f32; 4];
        let max = (R * C) as f32 / 2 as f32;
        for i in 0..4 {
            score[i] = self.score[i] as f32 - max;
        }

        let mut to_play = vec![0f32; 4];
        to_play[self.current_player as usize] = 1.0;

        Tensor::from_slice(&[planes, score, to_play].concat())
    }
}

// pub struct TopsideEvaluator {}
// impl Evaluator<DotsAndBoxes> for TopsideEvaluator {
//     fn evaluate(&self, game_state: DotsAndBoxes) -> f64 {
//         let limit = ((WIDTH + 1) / 2) as i8;
//         let all_topside = game_state
//             .board
//             .iter()
//             .all(|(&e, &p)| p != game_state.current_player() || e.1 < limit);
//         if all_topside { 1.0 } else { 0.0 }
//     }
// }

pub struct ScoreEvaluator {}
impl<const R: usize, const C: usize> Evaluator<DotsAndBoxes<R, C>> for ScoreEvaluator {
    fn evaluate(&self, game_state: &DotsAndBoxes<R, C>, _: Player) -> f32 {
        let sum: usize = game_state.score.iter().sum();
        if sum == 0 {
            return 0.0;
        }
        let idx = Player::list()
            .iter()
            .position(|&p| game_state.current_player() == p)
            .unwrap();
        let share = game_state.score[idx] as f32 / sum as f32;
        share * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn edge_out_of_bounds() {
    //     use Dir::*;
    //     assert!(!Edge(-1, 0, N).in_bounds());
    //     assert!(!Edge(0, -1, N).in_bounds());
    //     assert!(Edge(0, 0, N).in_bounds());
    //     assert!(Edge(WIDTH as i8, 0, W).in_bounds());
    //     assert!(Edge(0, HEIGHT as i8, N).in_bounds());
    // }

    #[test]
    fn neighbors() {
        let mut game: DotsAndBoxes<3, 2> = DotsAndBoxes::new();
        dbg!(game.get_actions(Player::A));
        // game.apply_action(Edge(0, 0, N));
        // game.apply_action(Edge(2, 0, N));
        // game.apply_action(Edge(0, 2, N));
        // game.apply_action(Edge(2, 2, N));

        // game.apply_action(Edge(0, 0, W));
        // game.apply_action(Edge(2, 0, W));
        // game.apply_action(Edge(0, 2, W));
        // game.apply_action(Edge(2, 2, W));

        // game.apply_action(Edge(1, 0, W));
        // game.apply_action(Edge(3, 0, W));
        // game.apply_action(Edge(1, 2, W));
        // game.apply_action(Edge(3, 2, W));

        // game.apply_action(Edge(0, 1, N));

        dbg!(game.current_player);
        dbg!(game.score);

        while !game.is_terminal() {
            game.apply_action(game.get_actions(game.current_player())[0]);
        }

        dbg!(game.score);
        dbg!(game.outcome(Player::A));
    }

    #[test]
    fn display_dimensions() {
        let game: DotsAndBoxes<3, 2> = DotsAndBoxes::new();
        let rendered = format!("{}", game);

        let mut expected = String::new();
        let w = 2;
        let h = 3;

        for y in 0..=h {
            for _x in 0..w {
                expected.push('.');
                expected.push_str("   ");
            }
            expected.push('.');
            expected.push('\n');

            if y < h {
                for _x in 0..=w {
                    expected.push(' ');
                    if _x < w {
                        expected.push_str("   ");
                    }
                }
                expected.push('\n');
            }
        }

        assert_eq!(rendered, expected);
    }

    #[test]
    fn image() {
        let mut game: DotsAndBoxes<3, 2> = DotsAndBoxes::new();
        game.apply_action(Edge(0, 2, N));
        let b: Vec<f32> = game.image().try_into().unwrap();
        dbg!(b);
        game.apply_action(Edge(0, 0, N));
        let b: Vec<f32> = game.image().try_into().unwrap();
        dbg!(b);
    }
}
