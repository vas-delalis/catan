use agent::{
    Agent, GameState,
    agents::{ConstantEvaluator, Search},
    games::DotsAndBoxes,
};

fn main() {
    let agent = Search::new(ConstantEvaluator::new(0.0), 1000, true, 1.41, 1.0, 0.01);
    for _ in 0..100 {
        let mut game = DotsAndBoxes::new();
        while !game.is_terminal() {
            let action = agent.get_action(game.clone());
            game.apply_action(action);
            agent.inform(action);
        }
    }
}
