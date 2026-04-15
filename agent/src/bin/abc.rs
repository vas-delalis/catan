use agent::{
    Tournament,
    agents::{ConstantEvaluator, Random, Search},
    games::DotsAndBoxes,
};

fn main() {
    let mut tournament: Tournament<DotsAndBoxes> = Tournament::new(0.05, 0.05);
    tournament.add(Box::new(Random {}), false);
    tournament.add(Box::new(Random {}), true);
    tournament.add(Box::new(Random {}), true);
    tournament.add(
        Box::new(Search::new(
            ConstantEvaluator {},
            100,
            true,
            1.41,
            1.0,
            0.01,
        )),
        true,
    );
    tournament.play();
    tournament.leaderboard();
}
