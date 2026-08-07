use std::{fmt::Display, sync::LazyLock};

use crate::{GameState, ml::ACTIVATION_SCALE};
use common::{Evaluation, Image, Outcome, Player as PlayerTrait};
use generic_array::GenericArray;

const R: usize = 5;
const C: usize = 5;
const N_EDGES: usize = 2 * R * C + R + C;

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
        String::from("DotsAndBoxes")
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
        let winners = self.winners()?;
        if winners.contains(&player) {
            if winners.len() > 1 {
                return Some((Draw, 1.0 / winners.len() as f32));
            }
            Some((Win, 1.0))
        } else {
            Some((Loss, -0.3333))
        }
    }

    fn scores(&self) -> Option<Evaluation<Self>> {
        let winners = self.winners()?;
        Some(GenericArray::from_iter(Player::list().into_iter().map(
            |p| {
                if winners.contains(&p) {
                    if winners.len() > 1 {
                        1.0 / winners.len() as f32
                    } else {
                        1.0
                    }
                } else {
                    -0.3333
                }
            },
        )))
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
    const IMAGE_SIZE: usize = N_EDGES + 8;

    fn tensor_image(&self) -> tch::Tensor {
        use tch::Tensor;

        let mut board_image = vec![0f32; N_EDGES];
        let mut board = self.board;
        for i in 0..N_EDGES {
            board_image[i] = (board & 1) as f32;
            board >>= 1;
        }

        let mut score = vec![0f32; 4];
        let max = (R * C) as f32 / 2.0;
        for i in 0..4 {
            score[i] = self.score[i] as f32 / max;
        }

        let mut to_play = vec![0f32; 4];
        to_play[self.current_player as usize] = 1.0;

        Tensor::from_slice(&[board_image, score, to_play].concat())
    }

    fn quantized_image(&self, buffer: *mut i8) {
        let scale = ACTIVATION_SCALE as i8;

        let mut idx = buffer;
        let mut board = self.board;
        for _ in 0..N_EDGES {
            unsafe {
                idx.write((board & 1) as i8 * scale);
                idx = idx.add(1);
            }
            board >>= 1;
        }

        let max = (R * C) as f32 / 2.0;
        for i in 0..4 {
            let normalized = self.score[i] as f32 / max;
            unsafe {
                idx.write((scale as f32 * normalized) as i8);
                idx = idx.add(1);
            }
        }
        let current = self.current_player() as usize;
        unsafe {
            idx = idx.add(current);
            idx.write(scale);
        }
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
    type Len = typenum::U4;
    fn list() -> Vec<Player> {
        use Player::*;
        vec![A, B, C, D]
    }
}

impl From<Player> for usize {
    fn from(val: Player) -> Self {
        val as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dir {
    N,
    W,
}

use Dir::*;
use generic_array::typenum;

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
        match dir {
            Dir::N => x < w && y <= h,
            Dir::W => x <= w && y < h,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use tch::IndexOp;

    use super::*;
    use crate::{Agent, agents::Random, ml::allocate_aligned_slice};

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

    #[test]
    fn quantized_image_matches_normal() {
        let luck = Random {};
        let image_size = DotsAndBoxes::IMAGE_SIZE;
        let quantized_img: *mut [i8] = allocate_aligned_slice(image_size.next_multiple_of(32));
        for _ in 0..1000 {
            let mut game = DotsAndBoxes::new();
            while !game.is_terminal() {
                let action = luck.get_action(game.clone());
                game.apply_action(action);

                let normal_img = Image::tensor_image(&game);
                game.quantized_image(quantized_img as *mut i8);

                for i in 0..image_size {
                    let a: i8 = (normal_img.i(i as i64) * 64).try_into().unwrap();
                    assert_eq!(
                        a,
                        unsafe { (quantized_img as *mut i8).add(i as usize).read() },
                        "index {} is different",
                        i
                    )
                }
                unsafe {
                    (quantized_img as *mut i8).write_bytes(0, image_size);
                }
            }
        }
    }

    #[test]
    fn quantized_image_non_negative() {
        let luck = Random {};
        let image_size = DotsAndBoxes::IMAGE_SIZE;
        let img: *mut [i8] = allocate_aligned_slice(image_size.next_multiple_of(32));
        for _ in 0..0100 {
            let mut game = DotsAndBoxes::new();
            while !game.is_terminal() {
                let action = luck.get_action(game.clone());
                game.apply_action(action);

                game.quantized_image(img as *mut i8);

                for i in 0..image_size {
                    let value = unsafe { (img as *mut i8).add(i as usize).read() };
                    assert!(value >= 0)
                }

                unsafe {
                    (img as *mut i8).write_bytes(0, image_size);
                }
            }
        }
    }
}
