use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use rand::{rng, seq::SliceRandom};

use crate::{Agent, INTERRUPTED, agents::Random};
use common::{GameState, Outcome, Player};

pub struct Tournament<'a, G> {
    roster: Vec<Participant<'a, G>>,
    matchups: HashMap<(usize, usize), Matchup>,
    false_positive_rate: f64,
    false_negative_rate: f64,
    max_moves: Option<usize>,
    max_time: Option<Duration>,
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
    agent: Box<dyn Agent<G> + 'a>,
    name: String,
    rating: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    evaluate: bool,
    game_clock: Duration,
}

impl<'a, G> Participant<'a, G> {
    fn winrate(&self) -> f64 {
        (self.wins as f64 + self.draws as f64 / 2.0) / (self.wins + self.draws + self.losses) as f64
    }
}

/// Returns the expected score ∈ \[0, 1] for a given Elo difference.
fn expected_score(diff: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-diff / 400.0))
}

fn log_likelihood_ratio(wins: usize, draws: usize, losses: usize) -> f64 {
    let n = (wins + draws + losses) as f64;
    if (wins == 0 || draws == 0 || losses == 0) && n < 100.0 {
        return 0.0;
    }
    let elo0 = 0.0;
    let elo1 = 5.0;
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
            max_moves: None,
            max_time: None,
        }
    }

    pub fn max_moves(mut self, n: usize) -> Self {
        self.max_moves = Some(n);
        self
    }

    pub fn max_time(mut self, d: Duration) -> Self {
        self.max_time = Some(d);
        self
    }

    pub fn add(&mut self, agent: Box<dyn Agent<G> + 'a>, name: &str, evaluate: bool) {
        self.roster.push(Participant {
            agent: agent,
            name: name.to_string(),
            rating: 1000.0,
            wins: 0,
            draws: 0,
            losses: 0,
            evaluate,
            game_clock: Duration::ZERO,
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
        let start = Instant::now();

        self.init_matchups();
        let luck = Random {};
        let mut results = 0;
        let mut games_played = 0;
        let to_test = self.matchups.iter().filter(|(_, m)| m.evaluate).count();
        println!(
            "Running tournament with {} agents ({} match-ups)...",
            self.roster.len(),
            to_test
        );

        while results < to_test {
            if INTERRUPTED.read() {
                println!("\r\x1b[KStopping tournament...");
                //        ^ Clear "^C"
                INTERRUPTED.reset();
                break;
            }
            if self.max_time.is_some_and(|max| start.elapsed() >= max) {
                println!("Time's up. Stopping tournament...");
                break;
            }

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
            let mut state = G::new();
            let mut moves = 0;
            let mut cancelled = false;
            while !state.is_terminal() {
                if self.max_moves.is_some_and(|max| moves >= max) {
                    cancelled = true;
                    break;
                }

                let idx = players
                    .iter()
                    .position(|&i| i == state.current_player())
                    .unwrap();

                let action = if state.is_random() {
                    luck.get_action(state.clone())
                } else {
                    let t = Instant::now();
                    let action = self.roster[agents[idx]].agent.get_action(state.clone());
                    self.roster[agents[idx]].game_clock += t.elapsed();
                    action
                };

                state.apply_action(action);
                for a in agents.iter() {
                    self.roster[*a].agent.inform(action);
                }
                moves += 1;
            }
            for a in agents.iter() {
                self.roster[*a].agent.reset();
            }
            if cancelled {
                continue;
            }
            games_played += 1;

            // Tally results and run tests
            for i in 0..player_count {
                let id1 = agents[i];
                let (outcome, _) = state.outcome(players[i]).unwrap();
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
                let mut delta = 0.0;
                for j in 0..player_count {
                    if i == j {
                        continue;
                    }
                    let id2 = agents[j];

                    let (outcome, score) = state.pairwise_outcome(players[i], players[j]).unwrap();
                    let matchup = self.matchups.get_mut(&(id1, id2)).unwrap();
                    matchup.update(outcome);

                    if matchup.evaluate
                        && matchup.test_result.is_none()
                        && let Some(result) = matchup
                            .termination_test(self.false_positive_rate, self.false_negative_rate)
                    {
                        results += 1;
                        let symbol = match result {
                            TestResult::H0 => "🟰 ",
                            TestResult::H1 => "✅",
                        };
                        println!(
                            "{:<15} {} {:>15}",
                            self.roster[id1].name, symbol, self.roster[id2].name
                        );
                    }
                    let expected_score =
                        expected_score(self.roster[id1].rating - self.roster[id2].rating) as f32;
                    delta += 16.0 * ((score + 1.0) / 2.0 - expected_score);
                    // dbg!(
                    //     self.roster[id1].rating,
                    //     self.roster[id2].rating,
                    //     expected_score,
                    //     delta
                    // );
                }
                self.roster[id1].rating += delta as f64;
            }
        }

        println!("Played {} games", games_played);
        println!("Elapsed: {}s", start.elapsed().as_secs());
    }

    pub fn leaderboard(&self) {
        for i in 0..self.roster.len() {
            println!("{} ", self.roster[i].name);
            for j in 0..self.roster.len() {
                if i == j {
                    continue;
                }
                let line = &self.matchups[&(i, j)];
                let wr = (line.wins as f64 + line.draws as f64 / 2.0)
                    / (line.wins + line.draws + line.losses) as f64
                    * 100.0;
                println!(
                    "    {:<12} {:>.0}%  {}/{}/{}",
                    self.roster[j].name, wr, line.wins, line.draws, line.losses
                )
            }
        }
        println!();
        for agent in &self.roster {
            println!(
                "{:<12} WR: {:>3.0}% | Rating: {:>4.0} | {}/{}/{} | Time: {:.1}s",
                agent.name,
                agent.winrate() * 100.0,
                agent.rating,
                agent.wins,
                agent.draws,
                agent.losses,
                agent.game_clock.as_secs_f64(),
            )
        }
    }
}
