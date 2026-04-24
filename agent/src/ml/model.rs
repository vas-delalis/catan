use tch::{
    Tensor,
    nn::{self, Path, Sequential, VarStore},
};

use crate::{agents::Evaluator, ml::Batch};

pub type Architecture = Box<dyn FnOnce(&Path) -> Sequential>;

pub fn two_layers<G: Batch>(hidden: i64) -> Architecture {
    Box::new(move |root: &Path<'_>| {
        nn::seq()
            .add(nn::linear(
                root.clone() / "layer1",
                G::BATCH_DIM,
                hidden,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                root.clone() / "layer2",
                hidden,
                hidden,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(root.clone(), hidden, 1, Default::default()))
    })
}

pub struct Model {
    layers: Sequential,
    vs: VarStore,
}

impl Model {
    pub fn new<G: Batch>(config: Architecture) -> Self {
        let device = tch::Device::Cpu;
        let vs = nn::VarStore::new(device);
        let root = vs.root();
        Model {
            layers: config(&root),
            vs,
        }
    }

    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }

    pub fn infer(&self, xs: Tensor) -> Tensor {
        xs.apply(&self.layers)
    }

    pub fn parameter_count(&self) -> usize {
        for (n, v) in self.vs.variables() {
            println!("{} {:?}", n, v);
        }
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel())
            .sum()
    }

    pub fn save(&self, path: &str) -> Result<(), tch::TchError> {
        self.vs.save(path)
    }

    pub fn load(&mut self, path: &str) -> Result<(), tch::TchError> {
        self.vs.load(path)
    }
}

impl<G: Batch> Evaluator<G> for Model {
    fn evaluate(&self, game_state: G) -> f64 {
        let image = game_state.batch();
        self.infer(image).try_into().unwrap()
    }
}
