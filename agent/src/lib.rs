use std::fmt::Debug;

pub mod agents;
pub mod games;
pub mod ml;
mod tournament;

pub use tournament::Tournament;

pub trait GameState: Clone + Send {
    type Action: Action;
    type Player: Player;

    fn name() -> String;
    fn new() -> Self;
    fn is_random(&self) -> bool;
    fn get_actions(&self, player: Self::Player) -> (Vec<Self::Action>, Option<Vec<f64>>);
    fn apply_action(&mut self, action: Self::Action);
    fn current_player(&self) -> Self::Player;
    fn is_terminal(&self) -> bool;
    fn outcome(&self, player: Self::Player) -> Option<(Outcome, f32)>;
    fn pairwise_outcome(
        &self,
        player1: Self::Player,
        player2: Self::Player,
    ) -> Option<(Outcome, f32)>;
}

pub trait Agent<G: GameState> {
    fn get_action(&self, game_state: G) -> G::Action;
    fn inform(&self, _action: G::Action) {}
    fn reset(&self) {}
}

pub trait Action: Copy + Debug + Into<usize> + From<usize> + Send {}
impl<T: Copy + Debug + Into<usize> + From<usize> + Send> Action for T {}

pub trait Player: Copy + PartialEq + Debug + Into<usize> + Send {
    const LEN: usize;
    fn list() -> Vec<Self>;
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
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
