use std::{fmt::Display, io};

use crate::{Action, Agent, GameState, Player};

pub struct Human {}

impl<A: Action, P: Player, G: GameState<A, P> + Display> Agent<A, P, G> for Human {
    fn get_action(&self, game_state: G) -> A {
        let actions = game_state.get_actions(game_state.current_player());

        println!("{}", game_state);
        println!("{:?}", game_state.current_player());
        println!("{:?}", actions);

        let mut action_idx = String::new();
        io::stdin()
            .read_line(&mut action_idx)
            .expect("failed to read line");

        let action_idx: usize = action_idx.trim().parse().expect("failed to parse number");
        actions[action_idx]
    }
}
