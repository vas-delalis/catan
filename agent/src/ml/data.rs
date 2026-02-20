use burn::{
    Tensor,
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::TensorData,
};
use rand::random_ratio;

use crate::{
    Agent, GameState,
    agents::Random,
    games::{TicTacToe, TicTacToePlayer},
};

#[derive(Debug, Clone)]
pub struct TicTacToeBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1>,
}

type Player = <TicTacToe as GameState>::Player;
type TicTacToeSnapshot = (TicTacToe, Option<Player>);

#[derive(Clone, Default)]
pub struct TicTacToeBatcher {}

impl<B: Backend> Batcher<B, TicTacToeSnapshot, TicTacToeBatch<B>> for TicTacToeBatcher {
    fn batch(&self, items: Vec<TicTacToeSnapshot>, device: &B::Device) -> TicTacToeBatch<B> {
        // items.iter().map(|replay| replay.choose(&mut rng).unwrap()).map(|(p, a)| )
        let mut images = vec![];
        let mut targets = vec![];
        for (state, winner) in items {
            let mut plane1 = vec![];
            let mut plane2 = vec![];
            let plane3 = if state.current_player() == TicTacToePlayer::X {
                [1.0; 9]
            } else {
                [0.0; 9]
            };

            for tile in state.board {
                match tile {
                    Some(p) => {
                        if p == TicTacToePlayer::X {
                            plane1.push(1.0);
                            plane2.push(0.0);
                        } else {
                            plane1.push(0.0);
                            plane2.push(1.0);
                        }
                    }
                    None => {
                        plane1.push(0.0);
                        plane2.push(0.0);
                    }
                }
            }
            let data1 = TensorData::new(plane1, [3, 3]);
            let plane1 = Tensor::<B, 2>::from_data(data1, &device);

            let data2 = TensorData::new(plane2, [3, 3]);
            let plane2 = Tensor::<B, 2>::from_data(data2, &device);

            let plane3 = Tensor::<B, 1>::from_floats(plane3, &device).reshape([3, 3]);
            let image = Tensor::stack::<3>(vec![plane1, plane2, plane3], 0);
            images.push(image);
            targets.push(winner);
        }

        let images = Tensor::stack(images, 0);

        TicTacToeBatch {
            images,
            targets: Tensor::<B, 1>::from_floats([1.0], device),
        }
    }
}

pub struct TicTacToeDataset {
    pub replay_buffer: Vec<(TicTacToe, Option<Player>)>,
}

impl TicTacToeDataset {
    pub fn new() -> Self {
        TicTacToeDataset {
            replay_buffer: vec![],
        }
    }
}

impl Dataset<TicTacToeSnapshot> for TicTacToeDataset {
    fn get(&self, index: usize) -> Option<TicTacToeSnapshot> {
        Some(self.replay_buffer[index].clone())
    }
    fn len(&self) -> usize {
        100_000
    }
}

pub fn selfplay(dataset: &mut TicTacToeDataset) {
    let agent = Random {};
    while dataset.replay_buffer.len() < dataset.len() {
        let mut game = TicTacToe::new();
        let mut buffer = vec![];
        while !game.is_terminal() {
            game.apply_action(agent.get_action(game.clone()));
            if random_ratio(1, 5) {
                buffer.push(game.clone());
            }
        }

        assert!(game.is_terminal());
        for state in buffer {
            let winner = state.check_winner();
            dataset.replay_buffer.push((state, winner));
        }
    }
}
