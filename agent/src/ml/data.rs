use common::{GameState, Image, Player};
use rand::seq::SliceRandom;
use tch::no_grad_guard;

use crate::{
    Agent,
    agents::{Random, Search},
    ml::{Hyperparameters, Model, QuantizedEvaluator},
};

type Snapshot<G> = (G, Vec<f32>);

pub struct Dataset<G: GameState> {
    replay_buffer: Vec<Snapshot<G>>,
    replay_count: usize,
}

impl<G: GameState + Image> Dataset<G> {
    pub fn new(replay_count: usize) -> Self {
        Dataset {
            replay_buffer: Vec::with_capacity(replay_count),
            replay_count,
        }
    }

    pub fn len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, (G, Vec<f32>)> {
        self.replay_buffer.drain(..)
    }

    pub fn self_play(&mut self, model: &Model<G>, params: &Hyperparameters, threads: usize) {
        let _guard = no_grad_guard();
        let replays_per_thread = self.replay_count / threads;
        let remainder = self.replay_count % threads;

        let results = std::thread::scope(|s| {
            let mut handles = vec![];
            for i in 0..threads {
                // Spread out the remainder
                let count = replays_per_thread + if i < remainder { 1 } else { 0 };
                if count == 0 {
                    continue;
                }

                handles.push(s.spawn(move || {
                    let quantized_agent = Search::new(
                        QuantizedEvaluator::new(model),
                        params.search_evals,
                        false,
                        1.41,
                        1.0,
                        params.dirichlet_alpha,
                    );
                    Self::generate(
                        quantized_agent,
                        count,
                        params.self_play_sampling_rate,
                        params.self_play_random_action_chance,
                    )
                }));
            }

            let mut thread_results = Vec::new();
            for handle in handles {
                thread_results.push(handle.join().unwrap());
            }
            thread_results
        });

        for res in results {
            self.replay_buffer.extend(res);
        }
        self.replay_buffer.shuffle(&mut rand::rng());
    }

    fn generate<A: Agent<G>>(
        agent: A,
        count: usize,
        sampling_rate: f64,
        random_action_chance: f64,
    ) -> Vec<Snapshot<G>> {
        let _guard = no_grad_guard();
        let mut thread_buffer = Vec::with_capacity(count);
        let luck = Random {};
        while thread_buffer.len() < count {
            let mut game = G::new();
            let mut buffer = vec![];
            while !game.is_terminal() {
                let action = if game.is_random() || rand::random_bool(random_action_chance) {
                    luck.get_action(game.clone())
                } else {
                    agent.get_action(game.clone())
                };
                game.apply_action(action);
                agent.inform(action);
                // Sampling gives better training data than adding every state.
                // This may be because it allows the model to see a greater variety of situations.
                if rand::random_bool(sampling_rate) {
                    buffer.push(game.clone());
                }
            }
            agent.reset();

            for state in buffer {
                let players = G::Player::list();
                let values: Vec<f32> = players
                    .iter()
                    .map(|&p| game.outcome(p).unwrap().1)
                    .collect();
                thread_buffer.push((state, values));
            }
        }
        thread_buffer
    }
}

impl<G: GameState> IntoIterator for Dataset<G> {
    type Item = Snapshot<G>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.replay_buffer.into_iter()
    }
}
