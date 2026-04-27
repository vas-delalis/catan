use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use rand::{rng, seq::SliceRandom};

use crate::{Agent, GameState, Outcome, Player};

pub struct Tournament<'a, G> {
    roster: Vec<Participant<'a, G>>,
    matchups: HashMap<(usize, usize), Matchup>,
    false_positive_rate: f64,
    false_negative_rate: f64,
}

#[derive(Debug, Clone)]
struct Matchup {
    wins: usize,
    draws: usize,
    losses: usize,
    test_result: Option<TestResult>,
    evaluate: bool,
}

impl Matchup {
    /// Runs a sequential probability ratio test for a two-agent match-up.
    fn termination_test(
        &mut self,
        false_positive_rate: f64,
        false_negative_rate: f64,
    ) -> Option<TestResult> {
        let alpha = false_positive_rate;
        let beta = false_negative_rate;
        let upper = f64::ln((1.0 - beta) / alpha);
        let lower = f64::ln(beta / (1.0 - alpha));
        let ratio = log_likelihood_ratio(self.wins, self.draws, self.losses);

        let result = if ratio > upper {
            Some(TestResult::H1)
        } else if ratio < lower {
            Some(TestResult::H0)
        } else {
            None
        };
        self.test_result = result;
        result
    }

    fn update(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Win => {
                self.wins += 1;
            }
            Outcome::Loss => {
                self.losses += 1;
            }
            Outcome::Draw => {
                self.draws += 1;
            }
        }
    }
}

struct Participant<'a, G> {
    id: usize,
    agent: Box<dyn Agent<G> + 'a>,
    rating: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    evaluate: bool,
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

impl<'a, G: GameState> Tournament<'a, G> {
    pub fn new(false_positive_rate: f64, false_negative_rate: f64) -> Self {
        Tournament {
            matchups: HashMap::new(),
            roster: vec![],
            false_positive_rate,
            false_negative_rate,
        }
    }

    pub fn add(&mut self, agent: Box<dyn Agent<G> + 'a>, evaluate: bool) {
        self.roster.push(Participant {
            id: self.roster.len(),
            agent: agent,
            rating: 1000.0,
            wins: 0,
            draws: 0,
            losses: 0,
            evaluate,
        });
    }

    fn init_matchups(&mut self) {
        for i in 0..self.roster.len() {
            for j in 0..self.roster.len() {
                if i == j {
                    continue;
                }
                self.matchups.insert(
                    (i, j),
                    Matchup {
                        wins: 0,
                        draws: 0,
                        losses: 0,
                        test_result: None,
                        evaluate: self.roster[i].evaluate && self.roster[j].evaluate,
                    },
                );
            }
        }
    }

    pub fn play(&mut self) {
        let player_count = G::Player::LEN;
        assert!(self.roster.len() >= player_count);
        println!("Running tournament with {} agents", self.roster.len());
        let start = Instant::now();

        self.init_matchups();
        let mut results = 0;
        let mut games_played = 0;
        let to_test = self.matchups.iter().filter(|(_, m)| m.evaluate).count();

        while results < to_test {
            // Select agents
            let mut agents = HashSet::with_capacity(player_count);
            for (matchup, _) in self
                .matchups
                .iter()
                .filter(|(_, m)| m.evaluate && m.test_result.is_none())
            {
                agents.insert(matchup.0);
                agents.insert(matchup.1);
                if agents.len() == player_count {
                    break;
                }
            }
            for i in 0..self.roster.len() {
                if agents.len() == player_count {
                    break;
                }
                agents.insert(i);
            }
            let agents: Vec<usize> = agents.into_iter().collect();

            let mut players = G::Player::list();
            players.shuffle(&mut rng());

            // Play game
            let mut game = G::new();
            while !game.is_terminal() {
                let scratch = game.clone();
                let idx = players
                    .iter()
                    .position(|&i| i == game.current_player())
                    .unwrap();

                let agent = &self.roster[agents[idx]].agent;
                let action = agent.get_action(scratch);
                game.apply_action(action);
            }
            games_played += 1;

            // Tally results and run tests
            for i in 0..player_count {
                let id1 = agents[i];
                let (outcome, _) = game.outcome(players[i]).unwrap();
                match outcome {
                    Outcome::Win => {
                        self.roster[id1].wins += 1;
                    }
                    Outcome::Loss => {
                        self.roster[id1].losses += 1;
                    }
                    Outcome::Draw => {
                        self.roster[id1].draws += 1;
                    }
                }
                for j in 0..player_count {
                    if i == j {
                        continue;
                    }
                    let id2 = agents[j];

                    let (outcome, _) = game.pairwise_outcome(players[i], players[j]).unwrap();
                    let matchup = self.matchups.get_mut(&(id1, id2)).unwrap();
                    matchup.update(outcome);
                    if matchup.evaluate
                        && matchup.test_result.is_none()
                        && let Some(result) = matchup
                            .termination_test(self.false_positive_rate, self.false_negative_rate)
                    {
                        results += 1;
                        match result {
                            TestResult::H0 => {
                                println!("Agent {} is no better than agent {}", id1, id2);
                            }
                            TestResult::H1 => {
                                println!("Agent {} is better than agent {}", id1, id2);
                            }
                        }
                    }
                }
            }

            // let expected_score = expected_score(self.roster[i].rating - self.roster[j].rating);

            // let delta = 16.0 * ((score + 1.0) / 2.0 - expected_score);
            // self.roster[i].rating += delta;
            // self.roster[j].rating -= delta;

            // if self.roster.len() > 0 {
            //     let scoreline = &self.scorelines[i][j];
            //     if let Some(result) =
            //         self.termination_test(scoreline.wins, max(scoreline.draws, 1), scoreline.losses)
            //     {
            //         self.test_results[i][j] = Some(result);
            //         matchups.remove(&(i, j));
            //         match result {
            //             TestResult::H0 => {
            //                 println!("Agent {} is no better than agent {}", i, j);
            //             }
            //             TestResult::H1 => {
            //                 println!("Agent {} is better than agent {}", i, j);
            //                 println!(
            //                     "{} {} {}",
            //                     scoreline.wins, scoreline.draws, scoreline.losses,
            //                 );
            //             }
            //         }
            //     }
            // }
        }

        println!("Played {} games", games_played);
        println!("Elapsed: {}s", start.elapsed().as_secs());
    }

    pub fn leaderboard(&self) {
        for i in 0..self.roster.len() {
            for j in 0..self.roster.len() {
                if i == j {
                    continue;
                }
                let line = &self.matchups[&(i, j)];
                let wr = (line.wins as f64 + line.draws as f64 / 2.0)
                    / (line.wins + line.draws + line.losses) as f64
                    * 100.0;
                println!(
                    "{}-{}  {:.0}%  {}/{}/{}",
                    i, j, wr, line.wins, line.draws, line.losses
                )
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
