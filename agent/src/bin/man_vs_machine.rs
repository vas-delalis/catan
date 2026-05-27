use std::path::Path;

use agent::{
    Agent, GameState, Player,
    agents::{Evaluator, Human, Random, Search},
    games::Pig,
    ml::Model,
};

type GAME = Pig;

fn main() {
    let _no_grad = tch::no_grad_guard();
    let (model, _) = Model::load::<GAME>(Path::new("./models/Pig/4.safetensors")).unwrap();

    let luck: Box<dyn Agent<GAME>> = Box::new(Random {});
    let mut agents: Vec<Box<dyn Agent<GAME>>> = Vec::new();
    agents.push(Box::new(Human {}));
    agents.push(Box::new(Search::new(&model, 1000, true, 1.41, 1.0, 0.01)));
    // agents.push(Box::new(Random {}));
    // agents.push(Box::new(Random {}));

    let mut game = GAME::new();
    while !game.is_terminal() {
        let agent = if game.is_random() {
            &luck
        } else {
            &agents[game.current_player() as usize]
        };
        let action = agent.get_action(game.clone());
        if game.is_random() {
            println!("[🎲]: {:?}", action);
        } else {
            println!("[{:?}]: {:?}", game.current_player(), action);
        }
        game.apply_action(action);
        for a in agents.iter() {
            a.inform(action);
        }
        for p in <GAME as GameState>::Player::list() {
            print!("{:.2} ", model.evaluate(&game, p));
        }
        println!();
        println!();
    }
    // for (i, &p) in <GAME as GameState>::Player::list().iter().enumerate() {
    //     println!("{:?} {} {:?}", p, game.score[i], game.outcome(p).unwrap())
    // }
}
