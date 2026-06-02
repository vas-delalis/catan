pub mod agents;
pub mod games;
pub mod ml;
mod tournament;

use common::GameState;
pub use tournament::Tournament;

pub trait Agent<G: GameState> {
    fn get_action(&self, game_state: G) -> G::Action;
    fn inform(&self, _action: G::Action) {}
    fn reset(&self) {}
}

#[macro_export]
macro_rules! with_game {
    ($name:expr => $G:ident { $($body:tt)* }) => {
        match $name {
            "DotsAndBoxes" => { type $G = $crate::games::DotsAndBoxes; $($body)* }
            "TicTacToe"    => { type $G = $crate::games::TicTacToe;    $($body)* }
            "OddsGame"     => { type $G = $crate::games::OddsGame;     $($body)* }
            "Pig"          => { type $G = $crate::games::Pig;          $($body)* }
            other => panic!("Unknown game: {other}"),
        }
    }
}
