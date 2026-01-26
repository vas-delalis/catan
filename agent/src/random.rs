use crate::{Agent, GameState};
use rand::{rng, seq::IndexedRandom};

#[derive(Debug, Clone)]
pub struct Random {}

impl<G: GameState> Agent<G> for Random {
    fn get_action(&self, game_state: G) -> G::Action {
        *game_state
            .get_actions(game_state.current_player())
            .choose(&mut rng())
            .unwrap()
    }
}
