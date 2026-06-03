use common::GameState;

use crate::{Action, Player, State};

impl GameState for State {
    type Action = Action;
    type Player = Player;
    fn apply_action(&mut self, _action: Self::Action) {
        todo!()
    }

    fn current_player(&self) -> Self::Player {
        todo!()
    }

    fn get_actions(&self, _player: Self::Player) -> (Vec<Self::Action>, Option<Vec<f64>>) {
        todo!()
    }

    fn is_random(&self) -> bool {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        todo!()
    }

    fn name() -> String {
        String::from("Catan")
    }

    fn new() -> Self {
        todo!()
    }

    fn outcome(&self, _player: Self::Player) -> Option<(common::Outcome, f32)> {
        todo!()
    }

    fn pairwise_outcome(
        &self,
        _player1: Self::Player,
        _player2: Self::Player,
    ) -> Option<(common::Outcome, f32)> {
        todo!()
    }
}
