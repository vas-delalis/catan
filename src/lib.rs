#![feature(portable_simd)]

pub mod board;
pub mod bundle;
pub mod common;
pub mod stockpile;

use board::Board;
use common::{Resource::*, *};
use enum_map::{enum_map, EnumMap};
use rand::prelude::*;
use stockpile::Stockpile;

struct State {
    bank: Stockpile,
    board: Board,
    player_order: [Player; 4],
    current_player_index: usize,
    stockpiles: EnumMap<Player, Stockpile>,
    has_rolled: bool,
    has_played_dev_card: bool,
}

impl State {
    fn new(&self) -> Self {
        let bank = Stockpile::bank();
        let player_banks: EnumMap<Player, Stockpile> = enum_map! {
            _ => Stockpile::player()
        };
        todo!()
    }

    fn current_player(&self) -> Player {
        self.player_order[self.current_player_index]
    }

    fn get_actions(&self) -> Vec<Action> {
        use Action::*;

        let mut actions = Vec::with_capacity(4);
        if !self.has_rolled {
            actions.push(RollDice);
        }
        actions
    }

    fn apply_acton(&mut self, action: Action) {
        use Action::*;

        let player = self.current_player();
        match action {
            BuildSettlement(vertex_id) => {
                // self.stockpiles[player].subtract(cost);
                self.board.add_settlement(player, vertex_id);
            }
            UpgradeSettlement(vertex_id) => {
                // self.stockpiles[player].subtract(cost);
                self.board.upgrade_settlement(vertex_id);
            }
            BuildRoad(edge_id) => {
                // self.stockpiles[player].subtract(cost);
                self.board.add_road(player, edge_id);
            }
            EndTurn => {
                self.current_player_index = (self.current_player_index + 1) % 4;
            }
            _ => {}
        }

        todo!()
    }

    fn roll_dice(&self) {
        let mut rng = rand::thread_rng();
        let roll = rng.gen_range(1..=6) + rng.gen_range(1..=6);

        // Calculate resource production (for each resource, for each player)
        // If the bank doesn't have enough of a resource:
        //   If only one player was supposed to get it, give them what remains
        //   Else, no one gets anything
    }
}
