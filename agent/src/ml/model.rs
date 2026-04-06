use tch::{Tensor, nn};

pub type Model<'a> = Box<dyn Fn(&Tensor) -> Tensor + 'a>;

pub fn create_model<'a>(vs: &nn::Path) -> Model<'a> {
    let hidden = 32;
    let seq = nn::seq()
        .add(nn::linear(vs / "layer1", 19, hidden, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, hidden, 1, Default::default()));
    Box::new(move |xs| xs.apply(&seq))
}
