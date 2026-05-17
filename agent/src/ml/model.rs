use std::{error::Error, path::Path as FSPath};

use serde::{Deserialize, Serialize};
use tch::{
    IndexOp, Tensor,
    nn::{self, Path as VSPath, Sequential, VarStore},
};

use crate::{
    GameState, Player,
    agents::Evaluator,
    ml::{Image, TrainingConfig},
};

pub struct Model {
    layers: Sequential,
    vs: VarStore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub game: String,
    pub layers: usize,
    pub hidden_dim: i64,
}

#[derive(Serialize, Deserialize)]
pub struct Metadata {
    pub model_config: ModelConfig,
    pub training_config: TrainingConfig,
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
            <G as GameState>::Player::LEN as i64,
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

    pub fn load<G: GameState + Image>(path: &FSPath) -> Result<(Self, Metadata), Box<dyn Error>> {
        let metadata_path = path.with_extension("json");
        let metadata_json = std::fs::read_to_string(metadata_path)?;
        let metadata: Metadata = serde_json::from_str(&metadata_json)?;

        let vs = VarStore::new(tch::Device::Cpu);
        let root = vs.root();

        let create_layers = vanilla::<G>(
            metadata.model_config.layers,
            metadata.model_config.hidden_dim,
        );

        let mut model = Model {
            layers: create_layers(&root),
            vs,
        };
        model.vs.load(path)?;

        Ok((model, metadata))
    }

    pub fn save_with_metadata(
        &self,
        path: &FSPath,
        metadata: &Metadata,
    ) -> Result<(), Box<dyn Error>> {
        self.vs.save(path)?;
        let metadata_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(metadata_path, config_json)?;
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
        let image = game_state.image();
        let p: usize = arbiter.into();
        self.infer(image).i(p as i64).try_into().unwrap()
    }
}
