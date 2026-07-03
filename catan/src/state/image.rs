use common::{Image, Player};
use tch::Tensor;

use crate::State;

impl Image for State {
    const IMAGE_SIZE: usize = 16;
    fn tensor_image(&self) -> Tensor {
        let mut resources = vec![0f32; 4];
        for p in Self::Player::list() {
            resources[p as usize] = self.player_resources[p].reduce_sum() as f32
        }

        let mut production = vec![0f32; 4];
        for p in Self::Player::list() {
            production[p as usize] = self.board.production(p).iter().sum()
        }

        let mut score = vec![0f32; 4];
        for p in Self::Player::list() {
            score[p as usize] = self.victory_points(p) as f32 / 10.0;
        }

        let mut turn = vec![0f32; 4];
        turn[self.whose_turn as usize] = 1.0;

        Tensor::from_slice(&[resources, production, score, turn].concat())
    }

    fn quantized_image(&self, _buffer: *mut i8) {
        todo!()
    }
}
