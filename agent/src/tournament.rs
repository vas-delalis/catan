use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, unbounded};
use rand::{rng, seq::SliceRandom};

use crate::{Agent, INTERRUPTED, agents::Random};
use common::{GameState, Outcome, Player};

pub struct Tournament<'a, G> {
    roster: Vec<Participant<'a, G>>,
    stats: Vec<ParticipantStats>,
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
    agent_factory: Box<dyn Fn() -> Box<dyn Agent<G> + 'a> + Sync + 'a>,
    name: String,
    evaluate: bool,
}

#[derive(Clone)]
struct ParticipantStats {
    rating: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    game_clock: Duration,
}

impl ParticipantStats {
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

/// A game to be simulated by a worker: which roster participants occupy each seat.
struct GameJob {
    agents: Vec<usize>,
}

enum GameOutcome {
    Cancelled,
    Completed {
        outcomes: Vec<Outcome>,
        pairwise: Vec<Vec<(Outcome, f32)>>,
        /// Time each agent spent computing moves this game.
        game_clocks: Vec<Duration>,
    },
}

struct GameResult {
    agents: Vec<usize>,
    outcome: GameOutcome,
}

/// Picks `player_count` agents for the next game, favoring match-ups that still need data.
fn select_job(
    matchups: &HashMap<(usize, usize), Matchup>,
    roster_len: usize,
    player_count: usize,
) -> GameJob {
    let mut agents = HashSet::with_capacity(player_count);
    for (matchup, _) in matchups
        .iter()
        .filter(|(_, m)| m.evaluate && m.test_result.is_none())
    {
        agents.insert(matchup.0);
        agents.insert(matchup.1);
        if agents.len() == player_count {
            break;
        }
    }
    for i in 0..roster_len {
        if agents.len() == player_count {
            break;
        }
        agents.insert(i);
    }
    GameJob {
        agents: agents.into_iter().collect(),
    }
}

/// Folds a completed game's results into match-up stats, ratings, and SPRTs.
#[allow(clippy::too_many_arguments)]
fn apply_result<G>(
    matchups: &mut HashMap<(usize, usize), Matchup>,
    stats: &mut [ParticipantStats],
    roster: &[Participant<'_, G>],
    false_positive_rate: f64,
    false_negative_rate: f64,
    result: GameResult,
    results: &mut usize,
    games_played: &mut usize,
) {
    let GameOutcome::Completed {
        outcomes,
        pairwise,
        game_clocks,
    } = result.outcome
    else {
        return;
    };
    *games_played += 1;

    let agents = result.agents;
    let player_count = agents.len();
    for i in 0..player_count {
        let id1 = agents[i];
        stats[id1].game_clock += game_clocks[i];
        match outcomes[i] {
            Outcome::Win => stats[id1].wins += 1,
            Outcome::Loss => stats[id1].losses += 1,
            Outcome::Draw => stats[id1].draws += 1,
        }

        let mut delta = 0.0;
        for j in 0..player_count {
            if i == j {
                continue;
            }
            let id2 = agents[j];
            let (outcome, score) = pairwise[i][j];

            let matchup = matchups.get_mut(&(id1, id2)).unwrap();
            matchup.update(outcome);

            if matchup.evaluate
                && matchup.test_result.is_none()
                && let Some(result) =
                    matchup.termination_test(false_positive_rate, false_negative_rate)
            {
                *results += 1;
                let symbol = match result {
                    TestResult::H0 => "🟰 ",
                    TestResult::H1 => "✅",
                };
                println!(
                    "{:<15} {} {:>15}",
                    roster[id1].name, symbol, roster[id2].name
                );
            }

            let expected_score = expected_score(stats[id1].rating - stats[id2].rating) as f32;
            delta += 16.0 * ((score + 1.0) / 2.0 - expected_score);
        }
        stats[id1].rating += delta as f64;
    }
}

impl<'a, G: GameState> Tournament<'a, G> {
    pub fn new(false_positive_rate: f64, false_negative_rate: f64) -> Self {
        Tournament {
            matchups: HashMap::new(),
            roster: vec![],
            stats: vec![],
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

    /// Adds a participant.
    pub fn add(
        &mut self,
        agent_factory: impl Fn() -> Box<dyn Agent<G> + 'a> + Sync + 'a,
        name: &str,
        evaluate: bool,
    ) {
        self.roster.push(Participant {
            agent_factory: Box::new(agent_factory),
            name: name.to_string(),
            evaluate,
        });
        self.stats.push(ParticipantStats {
            rating: 1000.0,
            wins: 0,
            draws: 0,
            losses: 0,
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
        let to_test = self.matchups.iter().filter(|(_, m)| m.evaluate).count();
        println!(
            "Running tournament with {} agents ({} match-ups)...",
            self.roster.len(),
            to_test
        );

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let max_moves = self.max_moves;
        let roster = &self.roster;
        let mut matchups = std::mem::take(&mut self.matchups);
        let mut stats = std::mem::take(&mut self.stats);
        let false_positive_rate = self.false_positive_rate;
        let false_negative_rate = self.false_negative_rate;

        let (job_tx, job_rx): (Sender<GameJob>, Receiver<GameJob>) = unbounded();
        let (result_tx, result_rx): (Sender<GameResult>, Receiver<GameResult>) = unbounded();

        let mut results = 0;
        let mut games_played = 0;

        std::thread::scope(|scope| {
            for _ in 0..num_workers {
                let job_rx = job_rx.clone();
                let result_tx = result_tx.clone();
                scope.spawn(move || {
                    let agents: Vec<Box<dyn Agent<G> + 'a>> =
                        roster.iter().map(|p| (p.agent_factory)()).collect();
                    let luck = Random {};

                    while let Ok(job) = job_rx.recv() {
                        let seats = job.agents;

                        let mut players = G::Player::list();
                        players.shuffle(&mut rng());

                        let mut state = G::new();
                        let mut moves = 0;
                        let mut cancelled = false;
                        let mut game_clocks = vec![Duration::ZERO; seats.len()];
                        while !state.is_terminal() {
                            if max_moves.is_some_and(|max| moves >= max) {
                                cancelled = true;
                                break;
                            }

                            let idx = players
                                .iter()
                                .position(|&pl| pl == state.current_player())
                                .unwrap();

                            let action = if state.is_random() {
                                luck.get_action(state.clone())
                            } else {
                                let t = Instant::now();
                                let action = agents[seats[idx]].get_action(state.clone());
                                game_clocks[idx] += t.elapsed();
                                action
                            };

                            state.apply_action(action);
                            for &seat in &seats {
                                agents[seat].inform(action);
                            }
                            moves += 1;
                        }
                        for &seat in &seats {
                            agents[seat].reset();
                        }

                        let outcome = if cancelled {
                            GameOutcome::Cancelled
                        } else {
                            let mut outcomes = Vec::with_capacity(seats.len());
                            let mut pairwise = vec![Vec::with_capacity(seats.len()); seats.len()];
                            for i in 0..seats.len() {
                                let (outcome, _) = state.outcome(players[i]).unwrap();
                                outcomes.push(outcome);
                                for j in 0..seats.len() {
                                    pairwise[i].push(if i == j {
                                        (outcome, 0.0)
                                    } else {
                                        state.pairwise_outcome(players[i], players[j]).unwrap()
                                    });
                                }
                            }
                            GameOutcome::Completed {
                                outcomes,
                                pairwise,
                                game_clocks,
                            }
                        };

                        if result_tx
                            .send(GameResult {
                                agents: seats,
                                outcome,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                });
            }

            drop(job_rx);
            drop(result_tx);

            let capacity = num_workers;
            let mut in_flight = 0;

            loop {
                while let Ok(game_result) = result_rx.try_recv() {
                    in_flight -= 1;
                    apply_result(
                        &mut matchups,
                        &mut stats,
                        roster,
                        false_positive_rate,
                        false_negative_rate,
                        game_result,
                        &mut results,
                        &mut games_played,
                    );
                }
                if results >= to_test {
                    break;
                }
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

                if in_flight < capacity {
                    let job = select_job(&matchups, roster.len(), player_count);
                    job_tx.send(job).unwrap();
                    in_flight += 1;
                } else if let Ok(game_result) = result_rx.recv() {
                    in_flight -= 1;
                    apply_result(
                        &mut matchups,
                        &mut stats,
                        roster,
                        false_positive_rate,
                        false_negative_rate,
                        game_result,
                        &mut results,
                        &mut games_played,
                    );
                } else {
                    break;
                }
            }

            // Stop feeding workers; let games already in flight finish and drain them.
            drop(job_tx);
            while in_flight > 0 {
                match result_rx.recv() {
                    Ok(game_result) => {
                        in_flight -= 1;
                        apply_result(
                            &mut matchups,
                            &mut stats,
                            roster,
                            false_positive_rate,
                            false_negative_rate,
                            game_result,
                            &mut results,
                            &mut games_played,
                        );
                    }
                    Err(_) => break,
                }
            }
        });

        self.matchups = matchups;
        self.stats = stats;

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
        for (participant, stats) in self.roster.iter().zip(&self.stats) {
            println!(
                "{:<12} WR: {:>3.0}% | Rating: {:>4.0} | {}/{}/{} | Time: {:.1}s",
                participant.name,
                stats.winrate() * 100.0,
                stats.rating,
                stats.wins,
                stats.draws,
                stats.losses,
                stats.game_clock.as_secs_f64(),
            )
        }
    }
}
