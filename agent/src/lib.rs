pub mod agents;
pub mod games;
pub mod ml;
mod tournament;

use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

use common::GameState;
pub use tournament::Tournament;

pub struct Interrupt {
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

pub static INTERRUPTED: Interrupt = Interrupt {
    ready: AtomicBool::new(false),
    interrupted: AtomicBool::new(false),
};

pub trait Agent<G: GameState> {
    fn get_action(&self, game_state: G) -> G::Action;
    fn inform(&self, _action: G::Action) {}
    fn reset(&self) {}
}

pub fn boxed<'a, G: GameState>(agent: impl Agent<G> + 'a) -> Box<dyn Agent<G> + 'a> {
    Box::new(agent)
}

#[macro_export]
macro_rules! with_game {
    ($name:expr => $G:ident { $($body:tt)* }) => {
        $crate::with_game!(@dispatch $name, $G, [$($body)*],
            $crate::games::DotsAndBoxes,
            $crate::games::TicTacToe,
            $crate::games::OddsGame,
            $crate::games::Pig,
            $crate::games::Catan,
        )
    };
    (@dispatch $name:expr, $G:ident, [$($body:tt)*], $head:ty, $($tail:ty,)*) => {
        if $name == <$head as $crate::__deps::GameState>::name() {
            type $G = $head;
            $($body)*
        } else {
            $crate::with_game!(@dispatch $name, $G, [$($body)*], $($tail,)*)
        }
    };
    (@dispatch $name:expr, $G:ident, [$($body:tt)*], ) => {
        panic!("Unknown game: {}", $name)
    };
}

#[doc(hidden)]
pub mod __deps {
    pub use common::GameState;
}
