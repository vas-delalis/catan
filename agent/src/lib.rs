use std::fmt::Debug;
use std::hash::Hash;

pub mod games;
mod human;
mod mcts;
mod random;

use rand::rng;
use rand::seq::SliceRandom;

pub use self::human::Human;
pub use self::mcts::Search;
pub use self::random::Random;

pub trait Agent<A: Action, P: Player, G: GameState<A, P>> {
    fn get_action(&self, game_state: G) -> A;
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

pub trait GameState<A: Action, P: Player>: Clone {
    fn new() -> Self;
    fn get_actions(&self, player: P) -> Vec<A>;
    fn apply_action(&mut self, action: A);
    fn current_player(&self) -> P;
    fn is_terminal(&self) -> bool;
    fn terminal_value(&self, player: P) -> Option<(Outcome, f64)>;
}

pub trait MultiplayerGameState<A: Action, P: Player>: GameState<A, P> {
    fn pairwise_terminal_value(&self, player1: P, player2: P) -> Option<(Outcome, f64)>;
}

pub struct Tournament<A, P, G> {
    roster: Vec<Participant<A, P, G>>,
    scorelines: Vec<Vec<Scoreline>>,
    test_results: Vec<Vec<Option<TestResult>>>,
}

#[derive(Debug, Clone)]
struct Scoreline {
    wins: usize,
    draws: usize,
    losses: usize,
}

struct Participant<A, P, G> {
    id: usize,
    agent: Box<dyn Agent<A, P, G>>,
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
    let elo1 = 10.0;
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

impl<A: Action, P: Player, G: GameState<A, P>> Tournament<A, P, G> {
    pub fn new(agents: Vec<Box<dyn Agent<A, P, G>>>) -> Self {
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

    pub fn play(&mut self, games: usize) {
        let mut roster_ids: Vec<usize> = (0..self.roster.len()).collect();
        let mut result_count = 0;
        let matchup_count = self.roster.len().pow(2) - self.roster.len();
        while result_count < matchup_count {
            let ids = loop {
                let (ids, _) = roster_ids.partial_shuffle(&mut rng(), 2);
                if self.test_results[ids[0]][ids[1]].is_none() {
                    break ids;
                }
            };
            let mut players = P::list();
            players.shuffle(&mut rng());

            let participants: Vec<(usize, P)> =
                ids.iter().cloned().zip(players.into_iter()).collect();

            let mut game = G::new();
            while !game.is_terminal() {
                let scratch = game.clone();
                let id = participants
                    .iter()
                    .find(|(_, p)| *p == game.current_player())
                    .map(|(id, _)| *id)
                    .unwrap();

                let agent = &self.roster[id].agent;
                let action = agent.get_action(scratch);
                game.apply_action(action);
            }

            let (outcome, score) = game.terminal_value(participants[0].1).unwrap();
            // dbg!(outcome, participants[0].0);

            match outcome {
                Outcome::Win => {
                    self.scorelines[ids[0]][ids[1]].wins += 1;
                    self.scorelines[ids[1]][ids[0]].losses += 1;
                    self.roster[participants[0].0].wins += 1;
                    self.roster[participants[1].0].losses += 1;
                }
                Outcome::Loss => {
                    self.scorelines[ids[0]][ids[1]].losses += 1;
                    self.scorelines[ids[1]][ids[0]].wins += 1;
                    self.roster[participants[0].0].losses += 1;
                    self.roster[participants[1].0].wins += 1;
                }
                Outcome::Draw => {
                    self.scorelines[ids[0]][ids[1]].draws += 1;
                    self.scorelines[ids[1]][ids[0]].draws += 1;
                    self.roster[participants[0].0].draws += 1;
                    self.roster[participants[1].0].draws += 1;
                }
            }

            let rating0 = self.roster[ids[0]].rating;
            let rating1 = self.roster[ids[1]].rating;
            let expected_score0 = expected_score(rating0 - rating1);

            let delta = 16.0 * (score - expected_score0);
            self.roster[ids[0]].rating += delta;
            self.roster[ids[1]].rating -= delta;

            if self.roster.len() > 0 {
                let scoreline = &self.scorelines[ids[0]][ids[1]];
                if let Some(result) =
                    self.termination_test(scoreline.wins, scoreline.draws, scoreline.losses)
                {
                    self.test_results[ids[0]][ids[1]] = Some(result);
                    result_count += 1;
                    match result {
                        TestResult::H0 => {
                            println!("Agent {} is no better than agent {}", ids[0], ids[1]);
                            // break;
                        }
                        TestResult::H1 => {
                            println!("Agent {} is better than agent {}", ids[0], ids[1]);
                            // break;
                        }
                    }
                }
            }
        }
    }

    pub fn leaderboard(&self) {
        for agent in &self.roster {
            println!(
                "{} {:.0} {}/{}/{}",
                agent.id, agent.rating, agent.wins, agent.draws, agent.losses
            )
        }
    }
}
