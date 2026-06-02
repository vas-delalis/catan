use agent::{
    Agent,
    agents::{ConstantEvaluator, Random, Search},
    games::DotsAndBoxes,
};
use common::GameState;

fn main() {
    let agent = Search::new(ConstantEvaluator::new(0.0), 1000, true, 1.41, 1.0, 0.01);
    let luck = Random {};
    for _ in 0..100 {
        let mut game = DotsAndBoxes::new();
        while !game.is_terminal() {
            let action = if game.is_random() {
                luck.get_action(game.clone())
            } else {
                agent.get_action(game.clone())
            };
            game.apply_action(action);
            agent.inform(action);
        }
        agent.reset();
    }
}
