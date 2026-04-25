use std::path::Path;

use agent::{
    Agent, GameState, Player,
    agents::{Evaluator, Human, Random, Search},
    games::{DotsAndBoxes, DotsAndBoxesPlayer},
    ml::{Model, two_layers},
};

fn main() {
    let _no_grad = tch::no_grad_guard();
    let arch = two_layers::<DotsAndBoxes>(8);
    let mut model = Model::new::<DotsAndBoxes>(arch);
    model
        .load(Path::new("./models/DotsAndBoxes/3.safetensors"))
        .unwrap();

    let mut agents: Vec<Box<dyn Agent<DotsAndBoxes>>> = Vec::new();
    agents.push(Box::new(Human {}));
    agents.push(Box::new(Search::new(&model, 10000, true, 1.41, 1.0, 0.01)));
    agents.push(Box::new(Random {}));
    agents.push(Box::new(Random {}));

    let mut game = DotsAndBoxes::new();
    while !game.is_terminal() {
        use DotsAndBoxesPlayer::*;
        let agent = match game.current_player() {
            A => &agents[0],
            B => &agents[1],
            C => &agents[2],
            D => &agents[3],
        };
        let action = agent.get_action(game.clone());
        println!("{:?} {:?}", game.current_player(), action);
        game.apply_action(action);
        println!("Value: {:.2}", model.evaluate(game.clone()));
    }
    for (i, &p) in <DotsAndBoxes as GameState>::Player::list()
        .iter()
        .enumerate()
    {
        println!("{:?} {} {:?}", p, game.score[i], game.outcome(p).unwrap())
    }
}
