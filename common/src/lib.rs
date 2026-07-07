use std::{fmt::Debug, sync::LazyLock};

use directories::ProjectDirs;
use generic_array::{ArrayLength, GenericArray};
use tch::Tensor;

pub static PROJECT_DIRS: LazyLock<ProjectDirs> =
    LazyLock::new(|| ProjectDirs::from("", "vas-delalis", "catan").unwrap());

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
    fn scores(&self) -> Option<Evaluation<Self>>;
    fn pairwise_outcome(
        &self,
        player1: Self::Player,
        player2: Self::Player,
    ) -> Option<(Outcome, f32)>;
}

pub type Evaluation<G> = GenericArray<f32, <<G as GameState>::Player as Player>::Len>;

pub trait Image: GameState {
    const IMAGE_SIZE: usize;
    fn tensor_image(&self) -> Tensor;
    fn quantized_image(&self, buffer: *mut i8);
}

pub trait Action: Copy + Debug + Into<usize> + From<usize> + Send {}
impl<T: Copy + Debug + Into<usize> + From<usize> + Send> Action for T {}

pub trait Player: Copy + PartialEq + Debug + Into<usize> + Send {
    const LEN: usize;
    type Len: ArrayLength;
    fn list() -> Vec<Self>;
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}
