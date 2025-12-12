use crate::{Action, Agent, GameState, Player};
use rand::{rng, seq::IndexedRandom};

pub struct Random {}

impl<A: Action, P: Player, G: GameState<A, P>> Agent<A, P, G> for Random {
    fn get_action(&self, game_state: G) -> A {
        *game_state
            .get_actions(game_state.current_player())
            .choose(&mut rng())
            .unwrap()
    }
}
