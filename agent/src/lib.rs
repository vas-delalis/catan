pub mod agents;
pub mod games;
pub mod ml;
mod tournament;

use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

use common::GameState;
pub use tournament::Tournament;

pub(crate) struct Interrupt {
    ready: AtomicBool,
    interrupted: AtomicBool,
}

impl Interrupt {
    fn setup_if_needed(&self) {
        if !self.ready.load(SeqCst) {
            self.ready.store(true, SeqCst);
            ctrlc::set_handler(|| INTERRUPTED.interrupted.store(true, SeqCst))
                .expect("Error setting CTRL+C handler");
        }
    }

    pub fn read(&self) -> bool {
        self.setup_if_needed();
        self.interrupted.load(SeqCst)
    }

    pub fn reset(&self) {
        self.setup_if_needed();
        self.interrupted.store(false, SeqCst);
    }
}

pub(crate) static INTERRUPTED: Interrupt = Interrupt {
    ready: AtomicBool::new(false),
    interrupted: AtomicBool::new(false),
};

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
            "Catan"        => { type $G = $crate::games::Catan;        $($body)* }
            other => panic!("Unknown game: {other}"),
        }
    }
}
