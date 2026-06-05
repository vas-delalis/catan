use std::fmt::Display;

use crate::State;
use common::{GameState, Player};

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        for p in <Self as GameState>::Player::list() {
            let buildings = self.board.buildings(p);
            output.push_str(&format!("{:?} {:?}\n", p, buildings));
        }
        write!(f, "{}", output)
    }
}
