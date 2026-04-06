use crate::{Agent, GameState};

type Snapshot<G> = (G, f64);

pub struct Dataset<G: GameState> {
    replay_buffer: Vec<Snapshot<G>>,
}

impl<G: GameState> Dataset<G> {
    pub fn new() -> Self {
        Dataset {
            replay_buffer: vec![],
        }
    }

    pub fn get(&self, index: usize) -> Snapshot<G> {
        self.replay_buffer[index].clone()
    }

    pub fn len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn selfplay(&mut self, agent: &dyn Agent<G>) {
        while self.len() < 1000 {
            let mut game = G::new();
            let mut buffer = vec![];
            while !game.is_terminal() {
                game.apply_action(agent.get_action(game.clone()));
                if rand::random_ratio(1, 3) {
                    buffer.push(game.clone());
                }
            }

            for state in buffer {
                // TODO: prev player, not current
                let (_, value) = game.outcome(state.current_player()).unwrap();
                self.replay_buffer.push((state, -value));
            }
        }
    }
}
