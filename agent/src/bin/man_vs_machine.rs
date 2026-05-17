use std::path::Path;

use agent::{
    Agent, GameState, Player,
    agents::{Evaluator, Human, Random, Search},
    games::{DotsAndBoxes, DotsAndBoxesPlayer},
    ml::Model,
};

type GAME = DotsAndBoxes;

fn main() {
    let _no_grad = tch::no_grad_guard();
    let (model, _) =
        Model::load::<GAME>(Path::new("./models/DotsAndBoxes5x5/2.safetensors")).unwrap();

    let mut agents: Vec<Box<dyn Agent<GAME>>> = Vec::new();
    agents.push(Box::new(Human {}));
    agents.push(Box::new(Search::new(&model, 0, true, 1.41, 1.0, 0.01)));
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
        for a in agents.iter() {
            a.inform(action);
        }
        print!("Values: ");
        for p in DotsAndBoxesPlayer::list() {
            print!("{:.2} ", model.evaluate(&game, p));
        }
        println!();
    }
    for (i, &p) in <GAME as GameState>::Player::list().iter().enumerate() {
        println!("{:?} {} {:?}", p, game.score[i], game.outcome(p).unwrap())
    }
}
