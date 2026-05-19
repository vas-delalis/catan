use rand::seq::SliceRandom;
use tch::no_grad_guard;

use crate::{Agent, GameState, Player};

type Snapshot<G> = (G, [f32; 4]);

pub struct Dataset<G: GameState> {
    replay_buffer: Vec<Snapshot<G>>,
    replay_count: usize,
}

impl<G: GameState> Dataset<G> {
    pub fn new(replay_count: usize) -> Self {
        Dataset {
            replay_buffer: Vec::with_capacity(replay_count),
            replay_count,
        }
    }

    pub fn len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn selfplay(&mut self, agent: &dyn Agent<G>) {
        let _guard = no_grad_guard();
        while self.len() < self.replay_count {
            let mut game = G::new();
            let mut buffer = vec![];
            while !game.is_terminal() {
                let action = agent.get_action(game.clone());
                game.apply_action(action);
                agent.inform(action);
                if rand::random_ratio(1, 5) {
                    buffer.push(game.clone());
                }
            }
            agent.reset();

            for state in buffer {
                let players = G::Player::list();
                let values: [f32; 4] = std::array::from_fn(|i| game.outcome(players[i]).unwrap().1);
                self.replay_buffer.push((state, values));
            }
        }
        self.replay_buffer.shuffle(&mut rand::rng());
    }
}

impl<G: GameState> IntoIterator for Dataset<G> {
    type Item = Snapshot<G>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.replay_buffer.into_iter()
    }
}
