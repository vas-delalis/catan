use std::{error::Error, fs, marker::PhantomData, path::PathBuf};

use tch::{
    Tensor,
    nn::{self, Path as VSPath, Sequential, VarStore},
};

use crate::{
    GameState,
    agents::Evaluator,
    ml::{ModelMetadata, quantization::CLAMP_LIMIT},
};
use common::{Evaluation, Image, Player};

pub struct Model<G: GameState + Image> {
    layers: Sequential,
    vs: VarStore,
    _game: PhantomData<G>,
}

pub type CreateLayers = Box<dyn FnOnce(&VSPath) -> Sequential>;

pub fn vanilla<G: GameState + Image>(layers: usize, hidden: i64) -> CreateLayers {
    assert!(layers > 1);
    Box::new(move |root: &VSPath<'_>| {
        let mut seq = nn::seq()
            .add(nn::linear(
                root.clone() / "layer0",
                G::IMAGE_SIZE as i64,
                hidden,
                Default::default(),
            ))
            .add_fn(|xs| xs.clamp(0.0, CLAMP_LIMIT));

        for i in 1..layers {
            seq = seq
                .add(nn::linear(
                    root.clone() / format!("layer{}", i),
                    hidden,
                    hidden,
                    Default::default(),
                ))
                .add_fn(|xs| xs.clamp(0.0, CLAMP_LIMIT));
        }

        seq = seq.add(nn::linear(
            root.clone() / "output",
            hidden,
            G::Player::LEN as i64,
            Default::default(),
        ));
        seq
    })
}

impl<G: GameState + Image> Model<G> {
    pub fn new(create_layers: CreateLayers) -> Self {
        let device = tch::Device::Cpu;
        let vs = VarStore::new(device);
        let root = vs.root();
        Model {
            layers: create_layers(&root),
            vs,
            _game: PhantomData,
        }
    }

    pub fn load<T: ToString>(name: T) -> Result<(Self, ModelMetadata), Box<dyn Error>> {
        let mut path = Self::get_dir();
        path.push(name.to_string());

        let checkpoint_path = path.with_extension("safetensors");
        let metadata_path = path.with_extension("json");

        let metadata_json = fs::read_to_string(metadata_path)?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;

        let vs = VarStore::new(tch::Device::Cpu);
        let root = vs.root();

        let create_layers = vanilla::<G>(
            metadata.model_config.layers,
            metadata.model_config.hidden_dim,
        );

        let mut model = Model {
            layers: create_layers(&root),
            vs,
            _game: PhantomData,
        };
        model.vs.load(checkpoint_path)?;

        Ok((model, metadata))
    }

    /// Saves the model as a checkpoint–metadata file pair and returns their paths.
    pub fn save_with_metadata(
        &self,
        name: &str,
        metadata: ModelMetadata,
    ) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
        let mut path = Self::get_dir();
        path.push(name);

        let checkpoint_path = path.with_extension("safetensors");
        self.vs.save(&checkpoint_path)?;

        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_json)?;
        Ok((checkpoint_path, metadata_path))
    }

    /// Saves the model without its corresponding metadata. See also `save_with_metadata()`.
    pub fn save(&self, name: &str) -> Result<PathBuf, Box<dyn Error>> {
        let mut path = Self::get_dir();
        path.push(name);

        let checkpoint_path = path.with_extension("safetensors");
        self.vs.save(&checkpoint_path)?;

        Ok(checkpoint_path)
    }

    pub fn get_next_id() -> usize {
        let dir = Self::get_dir();

        // Get highest id in directory
        let prev = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                e.unwrap()
                    .path()
                    .file_prefix()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse::<usize>()
                    .ok()
            })
            .max();
        match prev {
            Some(x) => x + 1,
            None => 0,
        }
    }

    fn get_dir() -> PathBuf {
        let mut path = model_dir();
        path.push(&G::name());
        fs::create_dir_all(&path).unwrap();
        path
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

// We force Model to be Sync in order to use multiple Search agents concurrently.
// Ideally, tch would offer a way to get an immutable reference for evaluation
// (with no gradients) or a mutable reference for training (with gradients enabled).
// Less ideally, we'd find a way to safely implement this ourselves.
// For now, `unsafe` works.
unsafe impl<G: GameState + Image> Sync for Model<G> {}

impl<G: GameState + Image> Evaluator<G> for Model<G> {
    fn evaluate(&self, game_state: &G) -> Evaluation<G> {
        let image = game_state.tensor_image();
        let vec: Vec<f32> = self.infer(image).try_into().unwrap();
        vec.try_into().unwrap()
    }
}

fn model_dir() -> PathBuf {
    let dirs = &common::PROJECT_DIRS;
    let data_dir = dirs.data_dir();
    [data_dir, &PathBuf::from("models")].iter().collect()
}
