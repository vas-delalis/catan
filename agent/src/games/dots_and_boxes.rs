use std::{fmt::Display, sync::LazyLock};

use crate::{GameState, agents::Evaluator};
use common::{Image, Outcome, Player as PlayerTrait};

#[derive(Clone)]
pub struct DotsAndBoxes {
    pub board: u64,
    current_player: Player,
    pub score: [u8; 4],
}

impl DotsAndBoxes {
    fn index(&self, e: &Edge) -> usize {
        let Edge(x, y, dir) = *e;
        let x: usize = x.into();
        let y: usize = y.into();
        let i = y * C + x;
        match dir {
            N => i,
            W => C * (R + 1) + i + y,
        }
    }

    fn contains(&self, e: &Edge) -> bool {
        let bit: u64 = 1 << self.index(e);
        bit & self.board > 0
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

impl GameState for DotsAndBoxes {
    type Action = usize;
    type Player = Player;

    fn name() -> String {
        format!("DotsAndBoxes{}x{}", R, C)
    }

    fn new() -> Self {
        DotsAndBoxes {
            board: 0,
            current_player: Player::A,
            score: [0, 0, 0, 0],
        }
    }

    fn get_actions(&self, _player: Player) -> (Vec<usize>, Option<Vec<f64>>) {
        (
            (0..N_EDGES)
                .filter(|i| self.board & (1 << i) == 0)
                .collect(),
            None,
        )
    }

    fn current_player(&self) -> Player {
        self.current_player
    }

    fn apply_action(&mut self, e: usize) {
        let idx = e;
        self.board |= 1 << idx;

        let (b1, b2) = BOXES[idx];
        let mut end_turn = true;

        if (b1 & self.board).count_ones() == 4 {
            self.score[self.current_player as usize] += 1;
            end_turn = false;
        }
        if b1 != b2 && (b2 & self.board).count_ones() == 4 {
            self.score[self.current_player as usize] += 1;
            end_turn = false;
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
        self.board.count_ones() == N_EDGES as u32
    }

    fn is_random(&self) -> bool {
        false
    }
}

impl Display for DotsAndBoxes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut grid = String::new();
        let h = R as u8;
        let w = C as u8;

        for y in 0..=h {
            // Dot row with horizontal edges
            for x in 0..w {
                grid.push('.');
                let e = Edge(x, y, N);
                if self.contains(&e) {
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
                    if self.contains(&e) {
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

impl Image for DotsAndBoxes {
    const IMAGE_SIZE: i64 = N_EDGES as i64 + 12;

    fn image(&self, arbiter: Player) -> tch::Tensor {
        use tch::Tensor;

        let mut board_image = vec![0f32; N_EDGES];
        let mut board = self.board;
        for i in 0..N_EDGES {
            board_image[i] = (board & 1) as f32;
            board >>= 1;
        }

        let mut score = vec![0f32; 4];
        let max = (R * C) as f32 / 2 as f32;
        for i in 0..4 {
            score[i] = self.score[i] as f32 - max;
        }

        let mut to_play = vec![0f32; 4];
        to_play[self.current_player as usize] = 1.0;

        let mut arbiter_vec = vec![0f32; 4];
        arbiter_vec[arbiter as usize] = 1.0;

        Tensor::from_slice(&[board_image, score, to_play, arbiter_vec].concat())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dir {
    N,
    W,
}

use Dir::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge(pub u8, pub u8, pub Dir);

impl Edge {
    fn from_index(i: usize) -> Self {
        if i < C * (R + 1) {
            let x = i % C;
            let y = i / C;
            Edge(x as u8, y as u8, N)
        } else {
            let i = i - (C * (R + 1));
            let x = i % (C + 1);
            let y = i / (C + 1);
            Edge(x as u8, y as u8, W)
        }
    }

    fn boxes(&self) -> [Vec<Self>; 2] {
        let Edge(x, y, dir) = *self;
        match dir {
            N => [self.box_edges(x, y.saturating_sub(1)), self.box_edges(x, y)],
            W => [self.box_edges(x.saturating_sub(1), y), self.box_edges(x, y)],
        }
    }

    fn box_edges(&self, x: u8, y: u8) -> Vec<Self> {
        vec![
            Edge(x, y, N),
            Edge(x, y, W),
            Edge(x, y + 1, N),
            Edge(x + 1, y, W),
        ]
    }

    fn index(&self) -> usize {
        let Edge(x, y, dir) = *self;
        let x: usize = x.into();
        let y: usize = y.into();
        let i = y * C + x;
        match dir {
            N => i,
            W => C * (R + 1) + i + y,
        }
    }

    fn in_bounds(&self) -> bool {
        let Edge(x, y, dir) = *self;
        let w = C as u8;
        let h = R as u8;
        let bound = match dir {
            Dir::N => x < w && y <= h,
            Dir::W => x <= w && y < h,
        };
        bound
    }
}

const R: usize = 5;
const C: usize = 5;
const N_EDGES: usize = 2 * R * C + R + C;

static BOXES: LazyLock<[(u64, u64); N_EDGES]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let e = Edge::from_index(i);
        assert_eq!(e.index(), i);
        let [box1, box2] = e.boxes();
        let mut mask1: u64 = 0;
        let mut mask2: u64 = 0;

        for e in box1 {
            if e.in_bounds() {
                mask1 |= 1 << e.index();
            }
        }
        for e in box2 {
            if e.in_bounds() {
                mask2 |= 1 << e.index();
            }
        }
        (mask1, mask2)
    })
});

pub struct ScoreEvaluator {}
impl Evaluator<DotsAndBoxes> for ScoreEvaluator {
    fn evaluate(&self, game_state: &DotsAndBoxes, _: Player) -> f32 {
        let sum: u8 = game_state.score.iter().sum();
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

    #[test]
    fn display_dimensions() {
        let game: DotsAndBoxes = DotsAndBoxes::new();
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
    fn get_actions() {
        type GAME = DotsAndBoxes;
        let mut game: GAME = DotsAndBoxes::new();
        let (actions, _) = game.get_actions(Player::A);
        assert_eq!(actions.len(), N_EDGES);
        game.apply_action(actions[1]);
        let (actions, _) = game.get_actions(Player::B);
        assert_eq!(actions.len(), N_EDGES - 1);
    }
}
