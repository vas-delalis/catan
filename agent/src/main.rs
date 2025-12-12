use agent::{Action, Agent, GameState, Outcome, Player, Random, Search};
use rand::{rng, seq::SliceRandom};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TicTacToePlayer {
    X,
    O,
}

impl Player for TicTacToePlayer {
    fn list() -> Vec<TicTacToePlayer> {
        vec![TicTacToePlayer::X, TicTacToePlayer::O]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Move(u8); // board position 0-8

impl Hash for Move {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[derive(Clone)]
struct TicTacToe {
    board: [Option<TicTacToePlayer>; 9],
    current_player: TicTacToePlayer,
}

impl TicTacToe {
    fn check_winner(&self) -> Option<TicTacToePlayer> {
        let lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for [a, b, c] in &lines {
            if self.board[*a].is_some()
                && self.board[*a] == self.board[*b]
                && self.board[*b] == self.board[*c]
            {
                return self.board[*a];
            }
        }
        None
    }
}

impl agent::GameState<Move, TicTacToePlayer> for TicTacToe {
    fn new() -> Self {
        TicTacToe {
            board: [None; 9],
            current_player: TicTacToePlayer::X,
        }
    }

    fn get_actions(&self, _player: TicTacToePlayer) -> Vec<Move> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| cell.is_none().then_some(Move(i as u8)))
            .collect()
    }

    fn current_player(&self) -> TicTacToePlayer {
        self.current_player
    }

    fn apply_action(&mut self, mv: Move) {
        self.board[mv.0 as usize] = Some(self.current_player);
        self.current_player = if self.current_player == TicTacToePlayer::X {
            TicTacToePlayer::O
        } else {
            TicTacToePlayer::X
        };
    }

    fn terminal_value(&self, player: TicTacToePlayer) -> Option<(Outcome, f64)> {
        if self.is_terminal() {
            if let Some(winner) = self.check_winner() {
                if winner == player {
                    Some((Outcome::Win, 1.0))
                } else {
                    Some((Outcome::Loss, 0.0))
                }
            } else {
                Some((Outcome::Draw, 0.5))
            }
        } else {
            None
        }
    }
    fn is_terminal(&self) -> bool {
        self.check_winner().is_some() || self.get_actions(self.current_player).is_empty()
    }
}

struct Tournament<A, P, G> {
    roster: Vec<Participant<A, P, G>>,
}

struct Participant<A, P, G> {
    id: usize,
    agent: Box<dyn Agent<A, P, G>>,
    rating: f64,
    wins: usize,
    draws: usize,
    losses: usize,
}

impl<A: Action, P: Player, G: GameState<A, P>> Tournament<A, P, G> {
    fn new(agents: Vec<Box<dyn Agent<A, P, G>>>) -> Self {
        Tournament {
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

    fn play(&mut self, games: usize) {
        for _ in 0..games {
            let mut ids: Vec<usize> = (0..self.roster.len()).collect();
            let (ids, _) = ids.partial_shuffle(&mut rng(), 2);
            let mut players = P::list();
            players.shuffle(&mut rng());

            let participants: Vec<(usize, P)> = ids
                .iter()
                .cloned()
                .zip(players.into_iter())
                .map(|(id, p)| (id, p))
                .collect();

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
                    self.roster[participants[0].0].wins += 1;
                    self.roster[participants[1].0].losses += 1;
                }
                Outcome::Loss => {
                    self.roster[participants[0].0].losses += 1;
                    self.roster[participants[1].0].wins += 1;
                }
                Outcome::Draw => {
                    self.roster[participants[0].0].draws += 1;
                    self.roster[participants[1].0].draws += 1;
                }
            }

            let rating0 = self.roster[participants[0].0].rating;
            let rating1 = self.roster[participants[1].0].rating;
            let expected_score = 1.0 / (1.0 + 10.0f64.powf((rating1 - rating0) / 400.0));

            self.roster[participants[0].0].rating += 1.0 * (score - expected_score);
            self.roster[participants[1].0].rating += 1.0 * (expected_score - score);
        }
    }

    fn leaderboard(&self) {
        for agent in &self.roster {
            println!(
                "{} {:.0} {}/{}/{}",
                agent.id, agent.rating, agent.wins, agent.draws, agent.losses
            )
        }
    }
}

fn main() {
    let mut agents: Vec<Box<dyn Agent<Move, TicTacToePlayer, TicTacToe>>> = Vec::new();
    for value in [0.0, 0.5, 1.0] {
        for evals in [10] {
            for alpha in [0.25, 0.5, 0.75] {
                agents.push(Box::new(Search::new(evals, 1.41, 1.0, alpha, value)));
            }
        }
    }
    agents.push(Box::new(Random {}));
    let mut tournament: Tournament<Move, TicTacToePlayer, TicTacToe> = Tournament::new(agents);
    tournament.play(100000);
    tournament.leaderboard();

    // let game_count = 0;
    // let agent1: Search<Move, TicTacToePlayer> = Search::new(5000, 1.41, 1.0, 0.75, 0.0);
    // let agent2: Search<Move, TicTacToePlayer> = Search::new(10, 1.41, 1.0, 0.75, 0.0);
    // let mut elo1 = 1000.0;
    // let mut elo2 = 1000.0;
    // for i in 0..game_count {
    //     let (x_player, o_player) = if i % 2 == 0 {
    //         (&agent1, &agent2)
    //     } else {
    //         (&agent2, &agent1)
    //     };
    //     let mut game = TicTacToe::new();
    //     while !game.is_terminal() {
    //         let scratch = game.clone();
    //         let action = if game.current_player == TicTacToePlayer::X {
    //             x_player.get_action(scratch)
    //         } else {
    //             o_player.get_action(scratch)
    //         };
    //         game.apply_action(action);
    //     }
    //     let expected_score1 = 1.0 / (1.0 + 10.0f64.powf((elo2 - elo1) / 400.0));
    //     let score_x = match game.check_winner() {
    //         Some(TicTacToePlayer::X) => {
    //             // victories1 += 1;
    //             1.0
    //         }
    //         Some(TicTacToePlayer::O) => 0.0,
    //         None => {
    //             // draws += 1;
    //             0.5
    //         }
    //     };
    //     let score1 = if i % 2 == 0 { score_x } else { 1.0 - score_x };
    //     elo1 += 16.0 * (score1 - expected_score1);
    //     elo2 += 16.0 * (expected_score1 - score1);
    //     // dbg!(score1, expected_score1, elo1, elo2);
    // }

    // dbg!(elo1, elo2);
    // println!(
    //     "Player 1 wins: {:.1}%",
    //     100.0 * victories1 as f64 / game_count as f64
    // );
    // println!("Draws: {:.1}%", 100.0 * draws as f64 / game_count as f64);
    // println!(
    //     "Player 2 wins: {:.1}%",
    //     100.0 * (game_count - victories1 - draws) as f64 / game_count as f64
    // );
}
