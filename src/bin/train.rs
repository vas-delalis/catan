use catan::*;
use rand::seq::IndexedRandom;
use std::time::Instant;

struct Env {}

impl Env {}

struct Agent {}

impl Agent {
    pub fn new(&self, initial_obs: InitialObservation) {}

    pub fn get_action(&self, obs: Observation) -> Action {
        *obs.actions.choose(&mut rand::rng()).unwrap()
    }

    pub fn update(&self) {}
}

fn main() {
    let mut count = 0;
    let start = Instant::now();

    for _ in 0..100 {
        let mut state = State::default();
        let agent = Agent {};
        while !state.is_terminal() {
            count += 1;
            // if state.get_actions(state.current_player()).len() == 0 {
            //     dbg!(state.phase);
            // }
            let action = agent.get_action(state.observe(state.current_player()));
            // dbg!(state.current_player(), state.phase, action);
            state.apply_action(action);
            agent.update();
            // sleep(Duration::from_millis(400));
        }
        // for p in PLAYERS {
        //     dbg!(state.victory_points(p));
        // }
    }

    dbg!(start.elapsed());
    dbg!(count);
}
