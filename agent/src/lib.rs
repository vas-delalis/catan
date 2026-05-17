use std::fmt::Debug;

pub mod agents;
pub mod games;
pub mod ml;
mod tournament;

pub use tournament::Tournament;

pub trait Agent<G: GameState> {
    fn get_action(&self, game_state: G) -> G::Action;
    fn inform(&self, _action: G::Action) {}
    fn reset(&self) {}
}

pub trait Action: Copy + Debug + Into<usize> + From<usize> {}
impl<T: Copy + Debug + Into<usize> + From<usize>> Action for T {}

pub trait Player: Copy + PartialEq + Debug + Into<usize> {
    const LEN: usize;
    fn list() -> Vec<Self>;
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

pub trait GameState: Clone {
    type Action: Action;
    type Player: Player;

    fn name() -> String;
    fn new() -> Self;
    fn get_actions(&self, player: Self::Player) -> Vec<Self::Action>;
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
