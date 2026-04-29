use std::path::Path;

use agent::{
    Agent, GameState, Player,
    agents::{Evaluator, Human, Random, Search},
    games::{DotsAndBoxes, DotsAndBoxesPlayer},
    ml::{Model, vanilla},
};

fn main() {
    let _no_grad = tch::no_grad_guard();
    let arch = vanilla::<DotsAndBoxes>(5, 16);
    let mut model = Model::new::<DotsAndBoxes>(arch);
    model
        .load(Path::new("./models/DotsAndBoxes/5.safetensors"))
        .unwrap();

    let mut agents: Vec<Box<dyn Agent<DotsAndBoxes>>> = Vec::new();
    agents.push(Box::new(Human {}));
    agents.push(Box::new(Search::new(&model, 1000, true, 1.41, 1.0, 0.01)));
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
        print!("Values: ");
        for p in DotsAndBoxesPlayer::list() {
            print!("{:.2} ", model.evaluate(&game, p));
        }
        println!();
    }
    for (i, &p) in <DotsAndBoxes as GameState>::Player::list()
        .iter()
        .enumerate()
    {
        println!("{:?} {} {:?}", p, game.score[i], game.outcome(p).unwrap())
    }
}
