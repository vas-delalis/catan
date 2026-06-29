use std::env;

use agent::{
    Agent,
    agents::{Evaluator, Human, Random, Search},
    ml::Model,
};
use common::{GameState, Player};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let game_name = args
        .next()
        .expect("Usage: man_vs_machine <game> <model id>");
    let model_id: usize = args
        .next()
        .expect("Usage: man_vs_machine <game> <model id>")
        .parse()?;

    let _no_grad = tch::no_grad_guard();

    agent::with_game!(game_name.as_str() => G {
        let (model, _) = Model::load(model_id).unwrap();

        let luck: Box<dyn Agent<G>> = Box::new(Random {});
        let mut agents: Vec<Box<dyn Agent<G>>> = Vec::new();
        agents.push(Box::new(Search::new(&model, 100, true, 1.41, 1.0, 0.01)));
        agents.push(Box::new(Human {}));
        // Fill remaining slots
        for _ in 2..<G as GameState>::Player::LEN {
            agents.push(Box::new(Random {}));
        }

        let mut game = G::new();
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
            for p in <G as GameState>::Player::list() {
                print!("{:.2} ", model.evaluate(&game, p));
            }
            println!();
            println!();
        }

        for (_, &p) in <G as GameState>::Player::list().iter().enumerate() {
            println!("{:?} {:?}", p, game.outcome(p).unwrap())
        }
    });

    Ok(())
}
