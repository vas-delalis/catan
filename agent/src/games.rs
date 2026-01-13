mod dots_and_boxes;
mod tic_tac_toe;

pub use dots_and_boxes::{DotsAndBoxes, Edge as DotsAndBoxesAction, Player as DotsAndBoxesPlayer};
pub use tic_tac_toe::{TicTacToe, TicTacToeAction, TicTacToePlayer};
