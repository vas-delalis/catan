use rand::seq::SliceRandom;

use crate::{Agent, GameState};

type Snapshot<G> = (G, f32);

pub struct Dataset<G: GameState> {
    replay_buffer: Vec<Snapshot<G>>,
    replay_count: usize,
}

impl<G: GameState> Dataset<G> {
    pub fn new(replay_count: usize) -> Self {
        Dataset {
            replay_buffer: vec![],
            replay_count,
        }
    }

    pub fn len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn selfplay(&mut self, agent: &dyn Agent<G>) {
        while self.len() < self.replay_count {
            let mut game = G::new();
            let mut buffer = vec![];
            while !game.is_terminal() {
                game.apply_action(agent.get_action(game.clone()));
                if rand::random_ratio(1, 5) {
                    buffer.push(game.clone());
                }
            }

            for state in buffer {
                let (_, value) = game.outcome(state.prev_player()).unwrap();
                self.replay_buffer.push((state, value));
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
