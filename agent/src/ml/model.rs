use tch::{
    Tensor,
    nn::{self, Path, Sequential, VarStore},
};

use crate::{agents::Evaluator, ml::Batch};

pub type Architecture = Box<dyn FnOnce(&Path) -> Sequential>;

pub fn vanilla<G: Batch>(layers: usize, hidden: i64) -> Architecture {
    assert!(layers > 1);
    Box::new(move |root: &Path<'_>| {
        let mut seq = nn::seq()
            .add(nn::linear(
                root.clone() / "layer1",
                G::BATCH_DIM,
                hidden,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());

        for i in 2..layers + 1 {
            seq = seq
                .add(nn::linear(
                    root.clone() / format!("layer{}", i),
                    hidden,
                    hidden,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu());
        }

        seq = seq.add(nn::linear(
            root.clone() / "output",
            hidden,
            1,
            Default::default(),
        ));
        seq
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
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel())
            .sum()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.vs.save(path)
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.vs.load(path)
    }
}

impl<G: Batch> Evaluator<G> for Model {
    fn evaluate(&self, game_state: G) -> f64 {
        let image = game_state.batch();
        self.infer(image).try_into().unwrap()
    }
}
