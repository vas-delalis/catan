#![feature(portable_simd)]

mod board;
pub mod bundle;
pub mod common;
pub mod stockpile;

pub use common::*;

use {
    board::Board,
    bundle::{Bundle, BUY_COSTS},
    enum_map::EnumMap,
    rand::prelude::*,
    stockpile::Stockpile,
};

struct State {
    stockpile: Stockpile,
    board: Board,
    player_order: [Player; 4],
    current_player_index: usize,
    player_data: EnumMap<Player, PlayerData>,
    has_rolled: bool,
    has_played_dev_card: bool,
    // TODO: keep track of freshly bought dev cards
}

struct PlayerData {
    resources: Bundle,
}

impl State {
    pub fn new(&self) -> Self {
        // let bank = Stockpile::bank();
        // let player_banks: EnumMap<Player, Stockpile> = enum_map! {
        //     _ => Stockpile::player()
        // };
        todo!()
    }

    fn current_player(&self) -> Player {
        self.player_order[self.current_player_index]
    }

    pub fn get_actions(&self) -> Vec<Action> {
        use Action::*;
        let player = self.current_player();
        let player_data = &self.player_data[player];

        let mut actions = Vec::with_capacity(4);
        if !self.has_rolled {
            actions.push(RollDice);
        }

        for (item, cost) in BUY_COSTS.iter() {
            if self.stockpile.has_purchasable(player, item)
                && cost <= &self.stockpile.resources
                && cost <= &player_data.resources
            {
                match item {
                    Purchasable::DevCard => actions.push(BuyDevCard),
                    Purchasable::Settlement => {
                        let slots = self.board.available_settlements(player);
                        actions.extend(slots.map(|v_id| BuildSettlement(v_id)));
                    }
                    Purchasable::City => {
                        let settlements = self.board.available_cities(player);
                        actions.extend(settlements.map(|v_id| UpgradeSettlement(v_id)));
                    }
                    Purchasable::Road => {
                        let slots = self.board.available_roads(player);
                        actions.extend(slots.map(|e_id| BuildRoad(e_id)));
                    }
                }
            }
        }
        actions
    }

    pub fn apply_acton(&mut self, action: Action) {
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
        let roll: u8 = rng.gen_range(1..=6) + rng.gen_range(1..=6);

        if roll == 7 {
            // Move robber
        } else {
            // Calculate resource production (for each resource, for each player)
            // If the bank doesn't have enough of a resource:
            //   If only one player was supposed to get it, give them what remains
            //   Else, no one gets anything
        }
    }
}
