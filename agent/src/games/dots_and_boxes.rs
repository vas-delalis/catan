use std::{collections::HashSet, fmt::Display, hash::Hash};

use crate::{GameState, Outcome, Player as PlayerTrait, agents::Evaluator, ml::Batch};

const WIDTH: usize = 2;
const HEIGHT: usize = 3;
const N_EDGES: usize = 2 * WIDTH * HEIGHT + WIDTH + HEIGHT;

#[derive(Clone)]
pub struct DotsAndBoxes {
    pub board: HashSet<Edge>,
    current_player: Player,
    prev_player: Player,
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
    fn list() -> Vec<Player> {
        vec![A, B, C, D]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir {
    N,
    W,
}

use Dir::*;
use Player::*;
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
    fn in_bounds(&self) -> bool {
        let Edge(x, y, dir) = *self;
        let w = WIDTH as i8;
        let h = HEIGHT as i8;
        let bound = match dir {
            Dir::N => x < w && y <= h,
            Dir::W => x <= w && y < h,
        };
        bound && 0 <= x && 0 <= y
    }

    fn boxes(&self) -> [Box; 2] {
        let Edge(x, y, _) = *self;
        match self.2 {
            N => [Box(x, y - 1), Box(x, y)],
            W => [Box(x - 1, y), Box(x, y)],
        }
    }

    // fn neighbors(&self) -> Vec<Edge> {
    //     let Edge(x, y, dir) = *self;
    //     match dir {
    //         N => [
    //             (0, -1, N),
    //             (0, -1, W),
    //             (1, -1, W),
    //             (0, 0, W),
    //             (0, 1, N),
    //             (1, 0, W),
    //         ],
    //         W => [
    //             (-1, 1, N),
    //             (-1, 0, N),
    //             (-1, 0, W),
    //             (0, 0, N),
    //             (0, 1, N),
    //             (1, 0, W),
    //         ],
    //     }
    //     .into_iter()
    //     .map(|(i, j, d)| Edge(x + i, y + j, d))
    //     .filter(|e: &Edge| e.in_bounds())
    //     .collect()
    // }
}

impl DotsAndBoxes {
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

impl GameState for DotsAndBoxes {
    const NAME: &str = "DotsAndBoxes";
    type Action = Edge;
    type Player = Player;

    fn new() -> Self {
        DotsAndBoxes {
            board: HashSet::with_capacity(N_EDGES),
            prev_player: A,
            current_player: A,
            score: [0, 0, 0, 0],
        }
    }

    fn get_actions(&self, _player: Player) -> Vec<Edge> {
        let mut edges: Vec<Edge> = vec![];
        for x in -1..=(WIDTH as i8 + 1) {
            for y in -1..=(HEIGHT as i8 + 1) {
                for dir in [N, W] {
                    let e = Edge(x as i8, y as i8, dir);
                    if e.in_bounds() && !self.board.contains(&e) {
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

    fn prev_player(&self) -> Self::Player {
        self.prev_player
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
        self.prev_player = self.current_player;
        if end_turn {
            let players = Player::list();
            self.current_player = players[(self.current_player as usize + 1) % players.len()];
        }
    }

    fn outcome(&self, player: Player) -> Option<(Outcome, f64)> {
        use Outcome::*;
        if !self.is_terminal() {
            return None;
        }
        let winners = self.winners().unwrap();
        if winners.contains(&player) {
            if winners.len() > 1 {
                return Some((Draw, 1.0 / winners.len() as f64));
            }
            return Some((Win, 1.0));
        } else {
            return Some((Loss, -1.0));
        }
    }

    fn pairwise_outcome(&self, player1: Player, player2: Player) -> Option<(Outcome, f64)> {
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
        self.board.len() == N_EDGES
    }
}

impl Display for DotsAndBoxes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut grid = String::new();
        let w = WIDTH as i8;
        let h = HEIGHT as i8;

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

impl Batch for DotsAndBoxes {
    const BATCH_DIM: i64 = N_EDGES as i64 + 8;

    fn batch(&self) -> tch::Tensor {
        let mut planes = vec![0f32; N_EDGES];

        let mut count = 0;
        for x in -1..=(WIDTH as i8 + 1) {
            for y in -1..=(HEIGHT as i8 + 1) {
                for dir in [N, W] {
                    let e = Edge(x as i8, y as i8, dir);
                    if e.in_bounds() {
                        if self.board.contains(&e) {
                            planes[count] = 1.0;
                        }
                        count += 1;
                    }
                }
            }
        }
        let mut played = vec![0f32; 4];
        played[self.prev_player() as usize] = 1.0;
        let mut to_play = vec![0f32; 4];
        to_play[self.current_player as usize] = 1.0;

        Tensor::from_slice(&[planes, played, to_play].concat())
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
impl Evaluator<DotsAndBoxes> for ScoreEvaluator {
    fn evaluate(&self, game_state: DotsAndBoxes) -> f64 {
        let sum: usize = game_state.score.iter().sum();
        if sum == 0 {
            return 0.0;
        }
        let idx = Player::list()
            .iter()
            .position(|&p| game_state.current_player() == p)
            .unwrap();
        let share = game_state.score[idx] as f64 / sum as f64;
        share * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_out_of_bounds() {
        use Dir::*;
        assert!(!Edge(-1, 0, N).in_bounds());
        assert!(!Edge(0, -1, N).in_bounds());
        assert!(Edge(0, 0, N).in_bounds());
        assert!(Edge(WIDTH as i8, 0, W).in_bounds());
        assert!(Edge(0, HEIGHT as i8, N).in_bounds());
    }

    #[test]
    fn neighbors() {
        let mut game = DotsAndBoxes::new();
        dbg!(game.get_actions(A));
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
        dbg!(game.outcome(A));
    }

    #[test]
    fn display_dimensions() {
        let game = DotsAndBoxes::new();
        let rendered = format!("{}", game);

        let mut expected = String::new();
        let w = WIDTH as i8;
        let h = HEIGHT as i8;

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
    fn score_evaluator() {
        let mut game = DotsAndBoxes::new();
        game.apply_action(Edge(0, 0, N));
        game.current_player = A;
        game.apply_action(Edge(0, 0, W));
        game.current_player = A;
        game.apply_action(Edge(0, 1, N));
        game.current_player = A;
        game.apply_action(Edge(1, 0, W));
        game.current_player = A;
        let e = ScoreEvaluator {};
        assert!(e.evaluate(game) == 0.1);
    }

    #[test]
    fn batch() {
        let mut game = DotsAndBoxes::new();
        game.apply_action(Edge(0, 2, N));
        let b: Vec<f32> = game.batch().try_into().unwrap();
        dbg!(b);
        game.apply_action(Edge(0, 0, N));
        let b: Vec<f32> = game.batch().try_into().unwrap();
        dbg!(b);
    }
}
