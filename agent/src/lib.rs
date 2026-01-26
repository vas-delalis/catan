use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

pub mod games;
mod human;
mod mcts;
mod random;

use rand::rng;
use rand::seq::SliceRandom;

pub use self::human::Human;
pub use self::mcts::{Evaluator, Search};
pub use self::random::Random;

pub trait Agent<G: GameState> {
    fn get_action(&self, game_state: G) -> G::Action;
}

pub trait Action: Hash + Eq + Copy + Debug {}
impl<T: Hash + Eq + Copy + Debug> Action for T {}

pub trait Player: Copy + Eq + Debug {
    fn list() -> Vec<Self>;
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

pub trait GameState: Clone {
    type Action: Action;
    type Player: Player;

    fn new() -> Self;
    fn get_actions(&self, player: Self::Player) -> Vec<Self::Action>;
    fn apply_action(&mut self, action: Self::Action);
    fn current_player(&self) -> Self::Player;
    fn is_terminal(&self) -> bool;
    fn outcome(&self, player: Self::Player) -> Option<(Outcome, f64)>;
}

pub trait MultiplayerGameState: GameState {
    fn pairwise_outcome(
        &self,
        player1: Self::Player,
        player2: Self::Player,
    ) -> Option<(Outcome, f64)>;
}

pub struct Tournament<G> {
    roster: Vec<Participant<G>>,
    scorelines: Vec<Vec<Scoreline>>,
    test_results: Vec<Vec<Option<TestResult>>>,
}

#[derive(Debug, Clone)]
struct Scoreline {
    wins: usize,
    draws: usize,
    losses: usize,
}

struct Participant<G> {
    id: usize,
    agent: Box<dyn Agent<G>>,
    rating: f64,
    wins: usize,
    draws: usize,
    losses: usize,
}

fn expected_score(x: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-x / 400.0))
}

fn log_likelihood_ratio(wins: usize, draws: usize, losses: usize) -> f64 {
    if wins == 0 || draws == 0 || losses == 0 {
        return 0.0;
    }
    let elo0 = 0.0;
    let elo1 = 5.0;
    let n = (wins + draws + losses) as f64;
    let (w, d) = (wins as f64 / n, draws as f64 / n);
    let s = w + d / 2.0;
    let m2 = w + d / 4.0;
    let var = m2 - s * s;
    let var_s = var / n;
    let s0 = expected_score(elo0);
    let s1 = expected_score(elo1);
    (s1 - s0) * (2.0 * s - s0 - s1) / var_s / 2.0
}

#[derive(Debug, Clone, Copy)]
enum TestResult {
    H0,
    H1,
}

impl<G: GameState> Tournament<G> {
    pub fn new(agents: Vec<Box<dyn Agent<G>>>) -> Self {
        Tournament {
            scorelines: vec![
                vec![
                    Scoreline {
                        wins: 0,
                        draws: 0,
                        losses: 0
                    };
                    agents.len()
                ];
                agents.len()
            ],
            test_results: vec![vec![None; agents.len()]; agents.len()],
            roster: agents
                .into_iter()
                .enumerate()
                .map(move |(id, agent)| Participant {
                    id,
                    agent,
                    rating: 1000.0,
                    wins: 0,
                    draws: 0,
                    losses: 0,
                })
                .collect(),
        }
    }

    /// Runs a sequential probability ratio test for a two-agent match-up with the given scoreline.
    fn termination_test(&self, wins: usize, draws: usize, losses: usize) -> Option<TestResult> {
        let alpha = 0.01;
        let beta = 0.01;
        let upper = f64::ln((1.0 - beta) / alpha);
        let lower = f64::ln(beta / (1.0 - alpha));
        let ratio = log_likelihood_ratio(wins, draws, losses);
        if ratio > upper {
            return Some(TestResult::H1);
        }
        if ratio < lower {
            return Some(TestResult::H0);
        }
        None
    }

    pub fn play(&mut self) {
        let mut matchups = HashSet::new();
        for i in 0..self.roster.len() {
            for j in 0..self.roster.len() {
                if i == j {
                    continue;
                };
                matchups.insert((i, j));
            }
        }
        while matchups.len() > 0 {
            let (i, j) = matchups.iter().next().cloned().unwrap();
            let mut players = G::Player::list();
            players.shuffle(&mut rng());

            let mut game = G::new();
            while !game.is_terminal() {
                let scratch = game.clone();
                let id = if game.current_player() == players[0] {
                    i
                } else {
                    j
                };

                let agent = &self.roster[id].agent;
                let action = agent.get_action(scratch);
                game.apply_action(action);
            }

            let (outcome, score) = game.outcome(players[0]).unwrap();
            // dbg!(outcome, participants[0].0);

            match outcome {
                Outcome::Win => {
                    self.scorelines[i][j].wins += 1;
                    self.scorelines[j][i].losses += 1;
                    self.roster[i].wins += 1;
                    self.roster[j].losses += 1;
                }
                Outcome::Loss => {
                    self.scorelines[i][j].losses += 1;
                    self.scorelines[j][i].wins += 1;
                    self.roster[i].losses += 1;
                    self.roster[j].wins += 1;
                }
                Outcome::Draw => {
                    self.scorelines[i][j].draws += 1;
                    self.scorelines[j][i].draws += 1;
                    self.roster[i].draws += 1;
                    self.roster[j].draws += 1;
                }
            }

            let expected_score = expected_score(self.roster[i].rating - self.roster[j].rating);

            let delta = 16.0 * ((score + 1.0) / 2.0 - expected_score);
            self.roster[i].rating += delta;
            self.roster[j].rating -= delta;

            if self.roster.len() > 0 {
                let scoreline = &self.scorelines[i][j];
                if let Some(result) =
                    self.termination_test(scoreline.wins, scoreline.draws, scoreline.losses)
                {
                    self.test_results[i][j] = Some(result);
                    matchups.remove(&(i, j));
                    match result {
                        TestResult::H0 => {
                            println!("Agent {} is no better than agent {}", i, j);
                        }
                        TestResult::H1 => {
                            println!("Agent {} is better than agent {}", i, j);
                            println!(
                                "{} {} {}",
                                scoreline.wins, scoreline.draws, scoreline.losses,
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn leaderboard(&self) {
        for i in 0..self.scorelines.len() {
            for j in 0..self.scorelines.len() {
                if i == j {
                    continue;
                }
                let line = &self.scorelines[i][j];
                println!("{} {} {}/{}/{}", i, j, line.wins, line.draws, line.losses)
            }
        }
        for agent in &self.roster {
            println!(
                "{} {:.0} {}/{}/{}",
                agent.id, agent.rating, agent.wins, agent.draws, agent.losses
            )
        }
    }
}

impl<G: MultiplayerGameState> Tournament<G> {
    pub fn play_multiplayer(&mut self) {
        let mut game_count: usize = 0;
        let mut result_count = 0;
        let matchup_count = 12;
        while result_count < matchup_count {
            game_count += 1;
            if game_count % 10000 == 0 {
                dbg!(&self.scorelines[3][0]);
                dbg!(&self.scorelines[3][1]);
                dbg!(&self.scorelines[3][2]);
            }
            let mut players = G::Player::list();
            players.shuffle(&mut rng());

            let mut game = G::new();
            while !game.is_terminal() {
                let scratch = game.clone();
                let id = players
                    .iter()
                    .position(|&p| p == game.current_player())
                    .unwrap();

                let agent = &self.roster[id].agent;
                let action = agent.get_action(scratch);
                game.apply_action(action);
            }

            for i in 0..4 {
                for j in 0..4 {
                    if i == j {
                        continue;
                    }

                    if i < j {
                        let (outcome, _) = game.pairwise_outcome(players[i], players[j]).unwrap();
                        match outcome {
                            Outcome::Win => {
                                self.scorelines[i][j].wins += 1;
                                self.scorelines[j][i].losses += 1;
                            }
                            Outcome::Loss => {
                                self.scorelines[j][i].wins += 1;
                                self.scorelines[i][j].losses += 1;
                            }
                            Outcome::Draw => {
                                self.scorelines[i][j].draws += 1;
                                self.scorelines[j][i].draws += 1;
                            }
                        }
                    }

                    if self.test_results[i][j].is_some() {
                        continue;
                    }

                    // dbg!(&self.scorelines[i][j]);
                    if let Some(result) = self.termination_test(
                        self.scorelines[i][j].wins,
                        self.scorelines[i][j].draws,
                        self.scorelines[i][j].losses,
                    ) {
                        self.test_results[i][j] = Some(result);
                        result_count += 1;
                        match result {
                            TestResult::H0 => {
                                println!("Agent {} is no better than agent {}", i, j);
                            }
                            TestResult::H1 => {
                                println!("Agent {} is better than agent {}", i, j);
                            }
                        }
                    }
                }
            }
        }
    }
}
