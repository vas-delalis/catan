use tch::{Tensor, nn};

use crate::{agents::Evaluator, ml::Batch};

pub type Model<'a> = Box<dyn Fn(&Tensor) -> Tensor + 'a>;

pub fn create_model<'a, G: Batch>(vs: &nn::Path, hidden: i64) -> Model<'a> {
    let seq = nn::seq()
        .add(nn::linear(
            vs / "layer1",
            G::BATCH_DIM,
            hidden,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer2",
            hidden,
            hidden,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, hidden, 1, Default::default()));
    Box::new(move |xs| xs.apply(&seq))
}

#[derive(Clone)]
pub struct ModelEvaluator<'a> {
    pub model: &'a Model<'a>,
}

impl<'a> ModelEvaluator<'a> {
    pub fn new(model: &'a Model<'a>) -> Self {
        ModelEvaluator { model }
    }
}

impl<'a, G: Batch> Evaluator<G> for ModelEvaluator<'a> {
    fn evaluate(&self, game_state: G) -> f64 {
        let image = game_state.batch();
        (self.model)(&image).try_into().unwrap()
    }
}
