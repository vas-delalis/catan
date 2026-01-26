use std::{
    fmt::Display,
    io::{self, Write},
};

use crate::{Agent, GameState};

pub struct Human {}

impl<G: GameState + Display> Agent<G> for Human {
    fn get_action(&self, game_state: G) -> G::Action {
        let actions = game_state.get_actions(game_state.current_player());

        println!("{}", game_state);
        println!("{:?}", game_state.current_player());
        println!("{:?}", actions);

        let mut action_idx = String::new();
        print!("Action: ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut action_idx)
            .expect("failed to read line");

        let action_idx: usize = action_idx.trim().parse().expect("failed to parse number");
        actions[action_idx]
    }
}
