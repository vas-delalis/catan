use agent::{
    Agent, Tournament,
    agents::{ConstantEvaluator, Search},
    games::TicTacToe,
    ml::{Model, ModelEvaluator, TicTacToeBatcher, TrainingConfig},
};
use burn::{
    backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    },
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let model = load_model(artifact_dir, device.clone());
    let evaluator = ModelEvaluator {
        model,
        batcher: TicTacToeBatcher {},
        device,
    };

    let agent: Box<dyn Agent<TicTacToe>> = Box::new(Search::<TicTacToe, ModelEvaluator<B>>::new(
        evaluator, 10, true, 1.41, 1.0, 0.01,
    ));
    let mut agents: Vec<Box<dyn Agent<TicTacToe>>> = Vec::new();
    agents.push(agent);
    agents.push(Box::new(Search::<TicTacToe, ConstantEvaluator>::new(
        ConstantEvaluator {},
        10,
        true,
        1.41,
        1.0,
        0.01,
    )));
    let mut tournament: Tournament<TicTacToe> = Tournament::new(agents);
    tournament.play();
    tournament.leaderboard();
}

pub fn load_model<B: Backend>(artifact_dir: &str, device: B::Device) -> Model<B> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    config.model.init::<B>(&device).load_record(record)
}

fn main() {
    type B = LibTorch<f32>;
    type AB = Autodiff<B>;

    let device = LibTorchDevice::Cpu;
    let artifact_dir = "./model";

    infer::<AB>(artifact_dir, device.clone());
}
