#![feature(portable_simd)]

mod board;
pub mod bundle;
pub mod common;
mod stockpile;

pub use board::*;
pub use common::*;

use std::simd::cmp::SimdPartialOrd;

use rand::distributions::WeightedIndex;
use Action::*;

use {
    bundle::{Bundle, BUY_COSTS},
    enum_map::EnumMap,
    rand::prelude::*,
    stockpile::Stockpile,
};

enum Phase {
    Normal,
    Setup, // Each player's second settlement produces resources at the end of setup
    Discarding(Bundle), // Remaining cards to discard per player
    MovingRobber,
    StealingResources(HexId),
    RoadBuilding(u8), // Remaining roads to build
    YearOfPlenty(u8), // Remaining resource units to take
    Monopoly,
    Trading(),
}

pub struct State {
    stockpile: Stockpile,
    board: Board,
    whose_turn: Player,
    turn_order: [Player; 4],
    player_data: EnumMap<Player, PlayerData>,
    has_rolled: bool,
    has_played_dev_card: bool,
    phase: Phase,
    // TODO: keep track of freshly bought dev cards
}

struct PlayerData {
    resources: Bundle,
}

impl State {
    pub fn new(resources: Vec<Option<Resource>>, rolls: Vec<Option<u8>>) -> Self {
        let stockpile = Stockpile::bank();

        State {
            stockpile,
            board: Board::new(resources, rolls),
            whose_turn: PLAYERS[0],
            turn_order: PLAYERS,
            player_data: EnumMap::from_fn(|_| PlayerData {
                resources: Bundle::splat(0),
            }),
            has_rolled: false,
            has_played_dev_card: false,
            phase: Phase::Normal,
        }
    }

    // === Helpers ===

    /// Returns the player who picks the next action.
    pub fn current_player(&self) -> Player {
        match self.phase {
            Phase::Discarding(remaining) => *PLAYERS.iter().find(|&p| remaining[*p] > 0).unwrap(),
            Phase::Setup => {
                todo!()
            }
            _ => self.whose_turn,
        }
    }

    /// Transfers resources from bank to player
    fn give(&mut self, player: Player, bundle: Bundle) {
        self.stockpile.resources -= bundle;
        self.player_data[player].resources += bundle;
    }

    /// Transfers resources from player to bank
    fn take(&mut self, player: Player, bundle: Bundle) {
        self.stockpile.resources += bundle;
        self.player_data[player].resources -= bundle;
    }

    // === Action generation ===

    pub fn get_actions(&mut self) -> Vec<Action> {
        let player = self.current_player();
        let player_data = &self.player_data[player];

        match self.phase {
            Phase::Normal => {
                let mut actions = vec![];
                if !self.has_rolled {
                    actions.push(RollDice);
                }

                // BuyDevCard, BuildSettlement, UpgradeSettlement, BuildRoad
                for (item, cost) in BUY_COSTS.iter() {
                    if self.stockpile.has_purchasable(player, item)
                        && cost.data.simd_le(self.stockpile.resources.data).all()
                        && cost.data.simd_le(player_data.resources.data).all()
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
                                actions.extend(slots.map(|edge_id| BuildRoad(edge_id)));
                            }
                        }
                    }
                }

                // ExchangeResource (maritime trade)
                actions.append(&mut self.get_exchange_actions(player));
                actions
            }
            Phase::Setup => todo!(),
            Phase::StealingResources(hex_id) => self.get_steal_actions(hex_id),
            Phase::Discarding(_) => self.get_discard_actions(self.current_player()),
            Phase::MovingRobber => self.get_robber_actions(),
            Phase::RoadBuilding(remaining) => {
                let slots = self.board.available_roads(player);
                assert!(slots.count_ones() > 0);
                slots
                    .take(remaining as usize)
                    .map(|edge_id| BuildRoad(edge_id))
                    .collect()
            }
            Phase::YearOfPlenty(_) => todo!(),
            Phase::Monopoly => todo!(),
            Phase::Trading() => todo!(),
        }
    }

    fn get_robber_actions(&self) -> Vec<Action> {
        (0..N_HEXES)
            .filter(|&id| id != self.board.robber_hex_id())
            .map(|id| MoveRobber(id))
            .collect()
    }

    fn get_steal_actions(&self, hex_id: HexId) -> Vec<Action> {
        self.board
            .players_on_hex(hex_id)
            .into_iter()
            .map(|p| StealResource(p))
            .collect()
    }

    fn get_discard_actions(&self, player: Player) -> Vec<Action> {
        RESOURCES
            .into_iter()
            .filter(|&r| self.player_data[player].resources[r] > 0)
            .map(|r| DiscardResource(r))
            .collect()
    }

    /// Returns possible actions for exchanging resources (maritime trade).
    fn get_exchange_actions(&self, player: Player) -> Vec<Action> {
        let mut actions = vec![];
        let ratios = self.board.exchange_ratios(player);

        for res1 in RESOURCES {
            // Check if player has enough of res1
            if self.player_data[player].resources[res1] < ratios[res1] {
                continue;
            }
            for res2 in RESOURCES {
                // Prevent nonsensical trades and ensure bank has enough of res2 in stock
                if res1 == res2 || self.stockpile.resources[res2] == 0 {
                    continue;
                }
                actions.push(ExchangeResources(((res1, ratios[res1]), res2)));
            }
        }
        actions
    }

    // === Action execution/application ===

    pub fn apply_acton(&mut self, action: Action) {
        use Action::*;

        let player = self.current_player();
        match action {
            RollDice => {
                self.handle_dice_roll(self.roll_dice());
            }
            BuildSettlement(vertex_id) => {
                self.take(player, BUY_COSTS[Purchasable::Settlement]);
                self.board.add_settlement(player, vertex_id);
            }
            UpgradeSettlement(vertex_id) => {
                self.take(player, BUY_COSTS[Purchasable::City]);
                self.board.upgrade_settlement(vertex_id);
            }
            BuildRoad(edge_id) => {
                self.board.add_road(player, edge_id);
                self.phase = match self.phase {
                    Phase::RoadBuilding(1) => {
                        self.take(player, BUY_COSTS[Purchasable::Road]);
                        Phase::Normal
                    }
                    Phase::RoadBuilding(remaining) => Phase::RoadBuilding(remaining - 1),
                    Phase::Setup => Phase::Setup,
                    _ => panic!("tried to build road in invalid phase"),
                }
            }
            MoveRobber(hex_id) => {
                self.board.move_robber(hex_id);
                self.phase = Phase::StealingResources(hex_id);
            }
            StealResource(target) => {
                self.steal_resource(self.current_player(), target);
                self.phase = Phase::Normal;
            }
            DiscardResource(res) => {
                let mut bundle = Bundle::splat(0);
                bundle[res] += 1;
                self.take(player, bundle);
                self.phase = match self.phase {
                    Phase::Discarding(mut remaining) => {
                        remaining[player] -= 1;
                        if remaining.reduce_sum() == 0 {
                            Phase::MovingRobber
                        } else {
                            Phase::Discarding(remaining)
                        }
                    }
                    _ => panic!("tried to discard in invalid phase"),
                };
            }
            EndTurn => {
                self.whose_turn = self.turn_order[(self.whose_turn as usize + 1) % 4];
                // TODO: reset state variables
                self.has_rolled = false;
                self.has_played_dev_card = false;
            }
            _ => {}
        }
    }

    /// Player steals random resource card from target
    fn steal_resource(&mut self, player: Player, target: Player) {
        assert_ne!(player, target);
        let target_bundle = &mut self.player_data[target].resources;

        if target_bundle.count_nonzero() == 0 {
            return;
        }

        let arr = target_bundle.data.as_array();
        let index = WeightedIndex::new(&arr[..5]).unwrap();
        let mut rng = rand::thread_rng();

        let res = RESOURCES[index.sample(&mut rng)];
        target_bundle[res] -= 1;
        let player_bundle = &mut self.player_data[player].resources;
        player_bundle[res] += 1;
    }

    /// Returns the sum of two fair dice rolls.
    fn roll_dice(&self) -> u8 {
        let mut rng = rand::thread_rng();
        rng.gen_range(1..=6) + rng.gen_range(1..=6)
    }

    fn handle_dice_roll(&mut self, roll: u8) {
        self.has_rolled = true;
        if roll == 7 {
            let mut to_discard = Bundle::splat(0);
            for player in PLAYERS {
                let resources = &self.player_data[player].resources;
                let sum = resources.reduce_sum();
                if sum > 7 {
                    to_discard[player] = sum / 2;
                }
            }
            self.phase = if to_discard.reduce_sum() == 0 {
                Phase::MovingRobber
            } else {
                Phase::Discarding(to_discard)
            };
        } else {
            // Calculate resource production (for each resource, for each player)
            let production = self.board.produce_resources(roll, self.stockpile.resources);
            for player in PLAYERS {
                self.give(player, production[player]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sett(s: &mut State, p: Player, v: Vertex) {
        s.board.add_settlement(p, s.board.vertex_id(v));
    }

    fn road(s: &mut State, p: Player, e: Edge) {
        s.board.add_road(p, s.board.edge_id(e));
    }

    fn setup() -> State {
        let mut resources: Vec<Option<Resource>> = [
            Ore, Wool, Lumber, Grain, Brick, Wool, Brick, Grain, Lumber, Ore, // <- Desert
            Lumber, Ore, Lumber, Ore, Grain, Wool, Brick, Grain, Wool,
        ]
        .into_iter()
        .map(Some)
        .collect();

        let mut rolls: Vec<Option<u8>> = [
            10, 2, 9, 12, 6, 4, 10, 9, 11, 7, // <- Desert
            3, 8, 8, 3, 4, 5, 5, 6, 11,
        ]
        .into_iter()
        .map(Some)
        .collect();

        resources[9] = None;
        rolls[9] = None;

        let mut state = State::new(resources, rolls);
        let mut s = |p: Player, v: Vertex| sett(&mut state, p, v);

        s(Blue, Vertex(-2, 2, N));
        s(Blue, Vertex(0, 2, N));
        s(Orange, Vertex(2, -2, S));
        s(Orange, Vertex(-1, 2, N));
        s(Red, Vertex(0, -1, N));
        s(Red, Vertex(-2, 1, N));
        s(White, Vertex(-1, 0, N));
        s(White, Vertex(1, 1, N));

        let mut r = |p: Player, e: Edge| road(&mut state, p, e);
        r(Blue, Edge(-2, 2, NE));
        r(Blue, Edge(1, 1, W));
        r(Orange, Edge(-1, 2, NE));
        r(Orange, Edge(1, -1, NE));
        r(Red, Edge(-2, 1, NE));
        r(Red, Edge(0, -1, NE));
        r(White, Edge(-1, 0, NW));
        r(White, Edge(2, 0, W));

        state
    }

    fn apply_actions(s: &mut State, actions: Vec<Action>) {
        for action in actions {
            s.get_actions();
            s.apply_acton(action);
        }
    }

    #[test]
    fn resource_discarding() {
        let mut s = setup();
        s.player_data[Blue].resources = Bundle::from_slice(&[2, 2, 2, 2, 0]);
        s.player_data[Orange].resources = Bundle::from_slice(&[0, 0, 0, 5, 6]);
        s.player_data[Red].resources = Bundle::from_slice(&[2, 2, 0, 0, 0]);
        s.handle_dice_roll(7);
        let b = DiscardResource(Brick);
        let g = DiscardResource(Grain);
        let o = DiscardResource(Ore);

        // Blue discards 4, Orange 5, and Red 0
        // Order alternates between blue and orange. Blue finishes first, so orange discards again.
        apply_actions(&mut s, vec![b, o, b, o, g, o, g, o, o]);

        // After everyone discards, current player must move the robber.
        let next_actions = s.get_actions();
        assert_eq!(s.current_player(), s.whose_turn);
        assert!(next_actions.into_iter().all(|a| matches!(a, MoveRobber(_))))
    }

    #[test]
    fn steal_actions() {
        let mut s = setup();
        s.handle_dice_roll(7);

        // Move robber to hex with red & white settlements
        s.apply_acton(MoveRobber(4));

        let actions = s.get_actions();
        assert_eq!(actions.len(), 2);
        assert!(actions.contains(&StealResource(Red)));
        assert!(actions.contains(&StealResource(White)));
    }

    #[test]
    fn steal_resource() {
        let mut s = setup();
        let starting_blue = Bundle::from_slice(&[2, 2, 2, 0, 0]);
        let starting_red = Bundle::from_slice(&[2, 2, 0, 0, 0]);
        s.player_data[Blue].resources = starting_blue;
        s.player_data[Red].resources = starting_red;
        s.handle_dice_roll(7);
        s.apply_acton(MoveRobber(4));

        // Blue steals from red
        s.apply_acton(StealResource(Red));

        // Blue gains 1, red loses 1. Stolen resource is Brick or Grain, since that's all red had.
        assert_eq!(
            s.player_data[Red].resources.reduce_sum(),
            starting_red.reduce_sum() - 1
        );
        assert!(
            s.player_data[Blue].resources[Brick] == 3 || s.player_data[Blue].resources[Grain] == 3
        );
    }

    #[test]
    fn exchange_actions() {
        let mut s = setup();
        s.player_data[Blue].resources = Bundle::from_slice(&[4, 2, 0, 0, 0]);

        // Add harbors
        sett(&mut s, Blue, Vertex(2, -3, S)); // Grain harbor
        sett(&mut s, Blue, Vertex(-2, 0, N)); // Lumber harbor

        let actions = s.get_exchange_actions(Blue);
        assert_eq!(actions.len(), 8); // 4 from Brick, 4 from Grain
        assert!(&actions[..4] // 4 Brick for 1 of something else
            .into_iter()
            .all(|a| matches!(a, ExchangeResources(((Brick, 4), _)))));
        assert!(&actions[4..] // 2 Grain for 1 of something else
            .into_iter()
            .all(|a| matches!(a, ExchangeResources(((Grain, 2), _)))));
    }
}
