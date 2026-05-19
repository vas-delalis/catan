use std::{error::Error, path::Path as FSPath};

use tch::{
    Tensor,
    nn::{self, Path as VSPath, Sequential, VarStore},
};

use crate::{
    GameState,
    agents::Evaluator,
    ml::{Image, TrainingConfig},
};

pub struct Model {
    layers: Sequential,
    vs: VarStore,
}

pub type CreateLayers = Box<dyn FnOnce(&VSPath) -> Sequential>;

pub fn vanilla<G: GameState + Image>(layers: usize, hidden: i64) -> CreateLayers {
    assert!(layers > 1);
    Box::new(move |root: &VSPath<'_>| {
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
            1,
            Default::default(),
        ));
        seq
    })
}

impl Model {
    pub fn new<G: Image>(create_layers: CreateLayers) -> Self {
        let device = tch::Device::Cpu;
        let vs = VarStore::new(device);
        let root = vs.root();
        Model {
            layers: create_layers(&root),
            vs,
        }
    }

    pub fn load<G: GameState + Image>(
        path: &FSPath,
    ) -> Result<(Self, TrainingConfig), Box<dyn Error>> {
        let config_path = path.with_extension("json");
        let config_json = std::fs::read_to_string(config_path)?;
        let config: TrainingConfig = serde_json::from_str(&config_json)?;

        let vs = VarStore::new(tch::Device::Cpu);
        let root = vs.root();

        let create_layers =
            vanilla::<G>(config.model_config.layers, config.model_config.hidden_dim);

        let mut model = Model {
            layers: create_layers(&root),
            vs,
        };
        model.vs.load(path)?;

        Ok((model, config))
    }

    pub fn save_with_config(
        &self,
        path: &FSPath,
        config: &TrainingConfig,
    ) -> Result<(), Box<dyn Error>> {
        self.vs.save(path)?;
        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(config)?;
        std::fs::write(config_path, config_json)?;
        Ok(())
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
}

impl<G: GameState + Image> Evaluator<G> for Model {
    fn evaluate(&self, game_state: &G, arbiter: G::Player) -> f32 {
        let image = game_state.image(arbiter);
        self.infer(image).try_into().unwrap()
    }
}
