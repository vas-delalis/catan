use tch::{
    IndexOp, Tensor,
    nn::{self, Path, Sequential, VarStore},
};

use crate::{GameState, Player, agents::Evaluator, ml::Image};

pub type Architecture = Box<dyn FnOnce(&Path) -> Sequential>;

pub fn vanilla<G: GameState + Image>(layers: usize, hidden: i64) -> Architecture {
    assert!(layers > 1);
    Box::new(move |root: &Path<'_>| {
        let mut seq = nn::seq()
            .add(nn::linear(
                root.clone() / "layer1",
                G::IMAGE_SIZE,
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
            <G as GameState>::Player::LEN as i64,
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
    pub fn new<G: Image>(config: Architecture) -> Self {
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

impl<G: GameState + Image> Evaluator<G> for Model {
    fn evaluate(&self, game_state: G, arbiter: G::Player) -> f32 {
        let image = game_state.image();
        let p: usize = arbiter.into();
        self.infer(image).i(p as i64).try_into().unwrap()
    }
}
