use tch::{Tensor, nn};

use crate::{
    agents::Evaluator,
    games::{TicTacToe, TicTacToePlayer},
};

pub type Model<'a> = Box<dyn Fn(&Tensor) -> Tensor + 'a>;

pub fn create_model<'a>(vs: &nn::Path, hidden: i64) -> Model<'a> {
    let seq = nn::seq()
        .add(nn::linear(vs / "layer1", 19, hidden, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, hidden, 1, Default::default()));
    Box::new(move |xs| xs.apply(&seq))
}

#[derive(Clone)]
pub struct ModelEvaluator<'a> {
    pub model: &'a Model<'a>,
}

impl<'a> Evaluator<TicTacToe> for ModelEvaluator<'a> {
    fn evaluate(&self, game_state: TicTacToe) -> f64 {
        let image = batch(&game_state);
        (self.model)(&image).try_into().unwrap()
    }
}

pub fn batch(game_state: &TicTacToe) -> Tensor {
    use crate::GameState;
    let mut plane1: Vec<f32> = vec![];
    let mut plane2: Vec<f32> = vec![];
    let plane3: Vec<f32> = if game_state.current_player() == TicTacToePlayer::X {
        vec![1.0; 1]
    } else {
        vec![0.0; 1]
    };

    for tile in game_state.board {
        match tile {
            Some(p) => {
                if p == TicTacToePlayer::X {
                    plane1.push(1.0);
                    plane2.push(0.0);
                } else {
                    plane1.push(0.0);
                    plane2.push(1.0);
                }
            }
            None => {
                plane1.push(0.0);
                plane2.push(0.0);
            }
        }
    }
    let plane1 = Tensor::from_slice(&plane1);
    let plane2 = Tensor::from_slice(&plane2);
    let plane3 = Tensor::from_slice(&plane3); //.reshape([3, 3]);

    Tensor::cat(&[plane1, plane2, plane3], 0)
}
