use crate::{Agent, GameState};
use rand::{rng, seq::IndexedRandom};

#[derive(Debug, Clone)]
pub struct Random {}

impl<G: GameState> Agent<G> for Random {
    fn get_action(&self, state: G) -> G::Action {
        let (actions, probs) = state.get_actions(state.current_player());
        let probs = probs.unwrap_or(vec![1.0; actions.len()]);
        let zipped: Vec<(G::Action, f64)> = actions.into_iter().zip(probs).collect();
        zipped.choose_weighted(&mut rng(), |item| item.1).unwrap().0
    }
}
