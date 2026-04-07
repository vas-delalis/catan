use agent::{
    Agent, Tournament,
    agents::{ConstantEvaluator, Random, Search},
    games::{NormalizedOddsEvaluator, OddsEvaluator, OddsGame},
};

fn main() {
    let mut agents: Vec<Box<dyn Agent<OddsGame>>> = Vec::new();
    agents.push(Box::new(Random {}));
    agents.push(Box::new(Search::<OddsGame, ConstantEvaluator>::new(
        ConstantEvaluator {},
        10,
        true,
        1.41,
        1.0,
        0.01,
    )));
    agents.push(Box::new(Search::<OddsGame, OddsEvaluator>::new(
        OddsEvaluator {},
        10,
        true,
        1.41,
        1.0,
        0.01,
    )));
    agents.push(Box::new(Search::<OddsGame, OddsEvaluator>::new(
        OddsEvaluator {},
        10,
        true,
        1.41,
        1.0,
        0.01,
    )));

    let mut tournament: Tournament<OddsGame> = Tournament::new(agents, 1e-2, 1e-2);
    tournament.play();
    tournament.leaderboard();
}
