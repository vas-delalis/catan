mod dots_and_boxes;
mod tic_tac_toe;

pub use dots_and_boxes::{
    Dir as DotsAndBoxesDir, DotsAndBoxes, Edge as DotsAndBoxesAction, Player as DotsAndBoxesPlayer,
    ScoreEvaluator, TopsideEvaluator,
};
pub use tic_tac_toe::{Cell, TicTacToe, TicTacToePlayer};
