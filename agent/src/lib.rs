use std::fmt::Debug;
use std::hash::Hash;

mod human;
mod mcts;
mod random;

pub use self::mcts::Search;
pub use self::random::Random;

pub trait Agent<A: Action, P: Player, G: GameState<A, P>> {
    fn get_action(&self, game_state: G) -> A;
}

pub trait Action: Hash + Eq + Copy + Debug {}
impl<T: Hash + Eq + Copy + Debug> Action for T {}

pub trait Player: Copy + Eq + Debug {
    fn list() -> Vec<Self>;
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

pub trait GameState<A: Action, P: Player>: Clone {
    fn new() -> Self;
    fn get_actions(&self, player: P) -> Vec<A>;
    fn apply_action(&mut self, action: A);
    fn current_player(&self) -> P;
    fn is_terminal(&self) -> bool;
    fn terminal_value(&self, player: P) -> Option<(Outcome, f64)>;
}
