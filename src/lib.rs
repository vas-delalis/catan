#![feature(portable_simd)]

mod bank;
mod board;
pub mod bundle;
pub mod common;

pub use board::*;
pub use common::*;
use enum_map::Enum;

use std::{
    cmp::min,
    simd::{cmp::SimdPartialOrd, num::SimdUint},
};

use rand::distributions::WeightedIndex;
use Action::*;

use {
    bank::Bank,
    bundle::{Bundle, BUY_COSTS},
    enum_map::EnumMap,
    rand::prelude::*,
};

#[derive(Debug, Clone, Copy)]
pub enum Phase {
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
    pub phase: Phase,
    pub bank: Bank,
    pub board: Board,
    whose_turn: Player,
    turn_order: [Player; 4],
    player_data: EnumMap<Player, PlayerData>,

    armies: Bundle,
    army_leader: Option<Player>,

    dev_cards: EnumMap<Player, Bundle>,
    locked_dev_cards: EnumMap<DevCard, bool>,
    has_played_dev_card: bool,

    has_rolled: bool,
}

struct PlayerData {
    resources: Bundle,
}

impl Default for State {
    fn default() -> Self {
        let mut b = Board::default();

        let mut s = |p: Player, v: Vertex| b.add_settlement(p, b.vertex_id(v));
        s(Blue, Vertex(-2, 2, N));
        s(Blue, Vertex(0, 2, N));
        s(Orange, Vertex(2, -2, S));
        s(Orange, Vertex(-1, 2, N));
        s(Red, Vertex(0, -1, N));
        s(Red, Vertex(-2, 1, N));
        s(White, Vertex(-1, 0, N));
        s(White, Vertex(1, 1, N));

        let mut r = |p: Player, e: Edge| b.add_road(p, b.edge_id(e));
        r(Blue, Edge(-2, 2, NE));
        r(Blue, Edge(1, 1, W));
        r(Orange, Edge(-1, 2, NE));
        r(Orange, Edge(1, -1, NE));
        r(Red, Edge(-2, 1, NE));
        r(Red, Edge(0, -1, NE));
        r(White, Edge(-1, 0, NW));
        r(White, Edge(2, 0, W));

        State::new(b)
    }
}

impl State {
    pub fn new(board: Board) -> Self {
        let bank = Bank::bank();

        State {
            bank,
            board,
            whose_turn: PLAYERS[0],
            turn_order: PLAYERS,
            player_data: EnumMap::from_fn(|_| PlayerData {
                resources: Bundle::splat(0),
            }),
            armies: Bundle::splat(0),
            dev_cards: EnumMap::from_fn(|_| Bundle::splat(0)),
            locked_dev_cards: EnumMap::from_fn(|_| false),
            army_leader: None,
            has_rolled: false,
            has_played_dev_card: false,
            phase: Phase::Normal,
        }
    }

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

    pub fn is_terminal(&self) -> bool {
        self.victory_points(self.whose_turn) >= 10
    }

    /// Get the current state from the perspective of a given player.
    pub fn observe(&self, observer: Player) -> Observation {
        Observation {
            observer,
            current_player: self.current_player(),
            is_terminal: self.is_terminal(),
            actions: self.get_actions(),
            robber: self.board.robber_hex_id(),
            buildings: PLAYERS
                .into_iter()
                .flat_map(|p| {
                    let settlements = self.board.settlements(p).map(move |v| (p, v, false));
                    let cities = self.board.cities(p).map(move |v| (p, v, true));
                    settlements.chain(cities)
                })
                .collect(),
            roads: PLAYERS
                .into_iter()
                .flat_map(|p| self.board.roads(p).map(move |e| (p, e)))
                .collect(),
            observer_hand: ObserverHand {
                resources: EnumMap::from_fn(|r| self.player_data[observer].resources[r]),
                dev_cards: EnumMap::from_fn(|c| self.dev_cards[observer][c]),
            },
            hidden_hands: PLAYERS
                .into_iter()
                .filter(|&p| p != observer)
                .map(|p| HiddenHand {
                    player: p,
                    resources: self.player_data[p].resources.reduce_sum(),
                    dev_cards: self.dev_cards[p].reduce_sum(),
                })
                .collect(),
        }
    }

    // === Helpers ===

    /// Returns the a player's victory points.
    ///
    /// 1 per settlement; 2 per city; 1 per VP card; 2 for largest army; 2 for longest road.
    fn victory_points(&self, player: Player) -> u32 {
        let from_board = self.board.victory_points(player);
        let largest_army = 2 * (self.army_leader == Some(player)) as u32;
        from_board + self.dev_cards[player][DevCard::VictoryPoint] as u32 + largest_army
    }

    /// Transfers resources from bank to player
    fn take_from_bank(&mut self, player: Player, bundle: Bundle) {
        self.bank.resources -= bundle;
        self.player_data[player].resources += bundle;
    }

    /// Transfers resources from player to bank
    fn give_to_bank(&mut self, player: Player, bundle: Bundle) {
        self.bank.resources += bundle;
        self.player_data[player].resources -= bundle;
    }

    // === Action generation ===

    pub fn get_actions(&self) -> Vec<Action> {
        let player = self.current_player();
        let player_data = &self.player_data[player];

        match self.phase {
            Phase::Normal => {
                let mut actions = vec![];
                actions.push(if !self.has_rolled { RollDice } else { EndTurn });

                // BuyDevCard, BuildSettlement, UpgradeSettlement, BuildRoad
                for (item, cost) in BUY_COSTS.iter() {
                    if self.bank.purchasable_count(player, item) > 0
                        && cost.data.simd_le(self.bank.resources.data).all()
                        && cost.data.simd_le(player_data.resources.data).all()
                    {
                        match item {
                            Purchasable::DevCard => actions.push(BuyDevCard),
                            Purchasable::Settlement => {
                                let slots = self.board.available_settlements(player);
                                actions.extend(slots.map(|v_id| BuildSettlement(v_id)));
                            }
                            Purchasable::City => {
                                let settlements = self.board.settlements(player);
                                actions.extend(settlements.map(|v_id| UpgradeSettlement(v_id)));
                            }
                            Purchasable::Road => {
                                let slots = self.board.available_roads(player);
                                actions.extend(slots.map(|edge_id| BuildRoad(edge_id)));
                            }
                        }
                    }
                }

                // PlayDevCard
                if !self.has_played_dev_card {
                    use DevCard::*;

                    actions.extend(
                        [Knight, Monopoly, RoadBuilding, YearOfPlenty]
                            .into_iter()
                            .filter_map(|card| {
                                if self.dev_cards[player][card] > 0 && !self.locked_dev_cards[card]
                                {
                                    Some(PlayDevCard(card))
                                } else {
                                    None
                                }
                            }),
                    );
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
            Phase::YearOfPlenty(_) => RESOURCES
                .into_iter()
                .filter_map(|res| {
                    if self.bank.resources[res] > 0 {
                        Some(TakeFreeResource(res))
                    } else {
                        None
                    }
                })
                .collect(),
            Phase::Monopoly => RESOURCES.into_iter().map(|r| Monopolize(r)).collect(),
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
                if res1 == res2 || self.bank.resources[res2] == 0 {
                    continue;
                }
                actions.push(ExchangeResources(((res1, ratios[res1]), res2)));
            }
        }
        actions
    }

    // === Action execution/application ===

    pub fn apply_action(&mut self, action: Action) -> Option<ActionResult> {
        use Action::*;

        let player = self.current_player();
        match action {
            RollDice => {
                let roll = self.roll_dice();
                let prod = self.handle_dice_roll(roll);
                return Some(ActionResult::DiceRolled(roll, prod));
            }
            BuildSettlement(vertex_id) => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::Settlement]);
                self.bank.buildings[player][Purchasable::Settlement] -= 1;
                self.board.add_settlement(player, vertex_id);
            }
            UpgradeSettlement(vertex_id) => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::City]);
                self.bank.buildings[player][Purchasable::Settlement] += 1;
                self.bank.buildings[player][Purchasable::City] -= 1;
                self.board.upgrade_settlement(vertex_id);
            }
            BuildRoad(edge_id) => {
                self.board.add_road(player, edge_id);
                self.bank.buildings[player][Purchasable::Road] -= 1;
                self.phase = match self.phase {
                    Phase::RoadBuilding(1) => {
                        self.give_to_bank(player, BUY_COSTS[Purchasable::Road]);
                        Phase::Normal
                    }
                    Phase::RoadBuilding(remaining) => Phase::RoadBuilding(remaining - 1),
                    Phase::Setup => Phase::Setup,
                    _ => panic!("tried to build road in invalid phase"),
                }
            }
            MoveRobber(hex_id) => {
                self.board.move_robber(hex_id);
                self.phase = if self.board.players_on_hex(hex_id).len() > 0 {
                    Phase::StealingResources(hex_id)
                } else {
                    Phase::Normal
                }
            }
            StealResource(target) => {
                let result = self.steal_resource(self.current_player(), target);
                self.phase = Phase::Normal;
                if let Some(r) = result {
                    return Some(ActionResult::ResourceStolen(r));
                }
            }
            DiscardResource(res) => {
                let mut bundle = Bundle::splat(0);
                bundle[res] += 1;
                self.give_to_bank(player, bundle);
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
            PlayDevCard(card) => {
                self.activate_dev_card(card);
                self.has_played_dev_card = true;
                self.dev_cards[self.whose_turn][card] -= 1;
            }
            Monopolize(res) => {
                // Take all of res from other players
                let mut total = 0;
                for p in PLAYERS {
                    if p == player {
                        continue;
                    }
                    let count = self.player_data[p].resources[res];
                    total += count;
                    self.player_data[p].resources[res] = 0;
                }
                self.player_data[player].resources[res] += total;
                self.phase = Phase::Normal;
                return Some(ActionResult::Monopolized(res, total));
            }
            TakeFreeResource(res) => {
                let mut bundle = Bundle::splat(0);
                bundle[res] += 1;
                self.take_from_bank(player, bundle);
                self.phase = match self.phase {
                    Phase::YearOfPlenty(1) => Phase::Normal,
                    Phase::YearOfPlenty(remaining) => Phase::YearOfPlenty(remaining - 1),
                    _ => panic!("tried to take resource in invalid phase"),
                };
            }
            ExchangeResources(((res1, cost), res2)) => {
                let mut bundle = Bundle::splat(0);
                bundle[res1] = cost;
                self.give_to_bank(player, bundle);
                let mut bundle = Bundle::splat(0);
                bundle[res2] = 1;
                self.take_from_bank(player, bundle);
            }
            BuyDevCard => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::DevCard]);
                let card = self.bank.draw_random_dev_card();
                // You can only play one dev card per turn.
                // This means that we don't need to track how many cards of each type are locked.
                if self.dev_cards[player][card] == 0 {
                    self.locked_dev_cards[card] = true;
                }
                self.dev_cards[player][card] += 1;
                return Some(ActionResult::DevCardBought(card));
            }
            EndTurn => {
                self.whose_turn = self.turn_order[(self.whose_turn as usize + 1) % 4];
                // TODO: reset state variables
                self.has_rolled = false;
                self.locked_dev_cards = EnumMap::from_fn(|_| false);
                self.has_played_dev_card = false;
            }
            _ => {}
        }
        None
    }

    fn activate_dev_card(&mut self, card: DevCard) {
        use DevCard::*;
        match card {
            RoadBuilding => {
                let to_build = min(
                    2,
                    self.bank
                        .purchasable_count(self.whose_turn, Purchasable::Road),
                );
                self.phase = Phase::RoadBuilding(to_build);
            }
            Knight => {
                self.phase = Phase::MovingRobber;

                let old_max = self.armies.data.reduce_max();
                let army = &mut self.armies[self.whose_turn];
                *army += 1;
                if *army >= 3 && *army > old_max {
                    self.army_leader = Some(self.whose_turn);
                }
            }
            Monopoly => {
                // Choose a resource type. Take everyone's resource units of that type.
                self.phase = Phase::Monopoly;
            }
            YearOfPlenty => {
                // Take min(2, bank total) resource units from the bank.
                let to_take = min(2, self.bank.resources.reduce_sum());
                self.phase = Phase::YearOfPlenty(to_take);
            }
            _ => panic!("tried to play unplayable dev card"),
        }
    }

    /// Takes random resource unit from `target`, gives it to `player`,
    /// and returns the resource type (or `None` if target had nothing to steal).
    fn steal_resource(&mut self, player: Player, target: Player) -> Option<Resource> {
        assert_ne!(player, target);
        let target_bundle = &mut self.player_data[target].resources;

        if target_bundle.count_nonzero() == 0 {
            return None;
        }

        let arr = target_bundle.data.as_array();
        let index = WeightedIndex::new(&arr[..5]).unwrap();
        let mut rng = rand::thread_rng();

        let res = RESOURCES[index.sample(&mut rng)];
        target_bundle[res] -= 1;
        let player_bundle = &mut self.player_data[player].resources;
        player_bundle[res] += 1;
        Some(res)
    }

    /// Returns the sum of two fair dice rolls.
    fn roll_dice(&self) -> u8 {
        let mut rng = rand::thread_rng();
        rng.gen_range(1..=6) + rng.gen_range(1..=6)
    }

    /// Carries out the effects of a given dice roll.
    ///
    /// If the roll is a 7: handles resource discarding, moving the robber, etc.
    ///
    /// Otherwise: gives each player their resource production and then returns it.
    fn handle_dice_roll(&mut self, roll: u8) -> Option<EnumMap<Player, Bundle>> {
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
            return None;
        } else {
            // Calculate resource production (for each resource, for each player)
            let production = self.board.produce_resources(roll, self.bank.resources);
            for player in PLAYERS {
                self.take_from_bank(player, production[player]);
            }
            return Some(production);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_actions(s: &mut State, actions: Vec<Action>) {
        for action in actions {
            s.get_actions();
            s.apply_action(action);
        }
    }

    #[test]
    fn resource_discarding() {
        // TODO: split
        let mut s = State::default();
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
    fn must_steal_after_moving_robber() {
        let mut s = State::default();
        s.handle_dice_roll(7);

        // Move robber to hex with red & white settlements
        s.apply_action(MoveRobber(4));

        let actions = s.get_actions();
        assert_eq!(actions.len(), 2);
        assert!(actions.contains(&StealResource(Red)));
        assert!(actions.contains(&StealResource(White)));
    }

    #[test]
    fn stealing_transfers_one_resource() {
        let mut s = State::default();
        let starting_blue = Bundle::from_slice(&[2, 2, 2, 0, 0]);
        let starting_red = Bundle::from_slice(&[2, 2, 0, 0, 0]);
        s.player_data[Blue].resources = starting_blue;
        s.player_data[Red].resources = starting_red;
        s.handle_dice_roll(7);
        s.apply_action(MoveRobber(4));

        // Blue steals from red
        s.apply_action(StealResource(Red));

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
    fn can_exchange_4_to_1() {
        let mut s = State::default();

        s.player_data[Blue].resources = Bundle::splat(4);

        let actions = s.get_actions();
        assert!(actions
            .into_iter()
            .any(|a| matches!(a, ExchangeResources(((Brick, 4), _)))));
    }
    #[test]
    fn can_exchange_3_to_1_with_generic_harbor() {
        let mut s = State::default();
        s.player_data[Blue].resources = Bundle::splat(3);

        // Add generic harbor
        s.board
            .add_settlement(Blue, s.board.vertex_id(Vertex(0, -2, N)));

        let actions = s.get_actions();
        assert!(actions
            .into_iter()
            .any(|a| matches!(a, ExchangeResources(((Brick, 3), _)))));
    }

    #[test]
    fn can_exchange_2_to_1_with_resource_harbor() {
        let mut s = State::default();
        s.player_data[Blue].resources = Bundle::splat(2);

        // Add grain harbor
        s.board
            .add_settlement(Blue, s.board.vertex_id(Vertex(2, -3, S)));

        let actions = s.get_actions();
        assert!(actions
            .into_iter()
            .any(|a| matches!(a, ExchangeResources(((Grain, 2), _)))));
    }

    #[test]
    fn bank_shortage_prevents_exchange() {
        let mut s = State::default();
        s.bank.resources = Bundle::splat(0);

        s.player_data[Blue].resources = Bundle::splat(4);

        let actions = s.get_actions();
        assert!(!actions
            .into_iter()
            .any(|a| matches!(a, ExchangeResources((_, _)))));
    }

    #[test]
    fn exchange_resources() {
        let mut s = State::default();
        s.player_data[Blue].resources = Bundle::splat(4);

        s.apply_action(ExchangeResources(((Brick, 4), Grain)));

        assert_eq!(s.player_data[Blue].resources[Brick], 0);
        assert_eq!(s.player_data[Blue].resources[Grain], 5);
    }

    #[test]
    fn road_building_card_returns_to_normal_phase() {
        let mut s = State::default();
        s.activate_dev_card(DevCard::RoadBuilding);

        s.apply_action(BuildRoad(s.board.edge_id(Edge(0, 1, NE))));
        s.apply_action(BuildRoad(s.board.edge_id(Edge(0, 1, NW))));

        // An attempt at an implementation-independent way to check if we're back in normal phase
        let actions = s.get_actions();
        let can_roll_dice = actions.iter().any(|a| matches!(a, RollDice));
        assert!(can_roll_dice || s.has_rolled);
    }

    #[test]
    fn must_build_roads_after_playing_road_building_card() {
        let mut s = State::default();

        s.activate_dev_card(DevCard::RoadBuilding);

        let actions = s.get_actions();
        assert!(
            actions.iter().all(|a| matches!(a, BuildRoad(_))),
            "all available actions should be BuildRoad"
        );
    }

    #[test]
    fn roads_from_road_building_card_are_free() {
        let mut s = State::default();
        let starting_resources = Bundle::splat(5);
        s.player_data[Blue].resources = starting_resources;
        s.activate_dev_card(DevCard::RoadBuilding);

        s.apply_action(BuildRoad(s.board.edge_id(Edge(0, 1, NE))));

        assert_eq!(
            s.player_data[Blue].resources, starting_resources,
            "resources should remain unchanged after building road"
        );
    }

    #[test]
    fn must_move_robber_after_playing_knight() {
        let mut s = State::default();

        s.activate_dev_card(DevCard::Knight);

        let actions = s.get_actions();
        assert!(
            actions.iter().all(|a| matches!(a, MoveRobber(_))),
            "all available actions should be MoveRobber"
        );
    }

    #[test]
    fn first_with_size_3_army_gets_points() {
        let mut s = State::default();
        s.activate_dev_card(DevCard::Knight);
        s.activate_dev_card(DevCard::Knight);
        let before = s.victory_points(Blue);

        s.activate_dev_card(DevCard::Knight);

        let after = s.victory_points(Blue);
        assert_eq!(after - before, 2);
    }

    #[test]
    fn surpassing_army_leader_transfers_points() {
        let mut s = State::default();
        s.army_leader = Some(Red);
        s.armies[Red] = 3;
        s.armies[Blue] = 3; // (Red got to 3 before Blue)
        let b_before = s.victory_points(Blue);
        let r_before = s.victory_points(Red);

        // Blue plays a knight
        s.activate_dev_card(DevCard::Knight);

        let b_after = s.victory_points(Blue);
        let r_after = s.victory_points(Red);
        assert_eq!(b_after - b_before, 2); // Up 2
        assert_eq!(r_before - r_after, 2); // Down 2
    }

    #[test]
    fn must_monopolize_after_playing_monopoly() {
        let mut s = State::default();

        s.activate_dev_card(DevCard::Monopoly);

        let actions = s.get_actions();
        assert!(
            actions.iter().all(|a| matches!(a, Monopolize(_))),
            "all available actions should be Monopolize"
        );
    }

    #[test]
    fn monopolizing_transfers_all_resources_of_type() {
        let mut s = State::default();
        s.player_data[Blue].resources = Bundle::from_slice(&[2, 2, 2, 2, 0]);
        s.player_data[Orange].resources = Bundle::from_slice(&[0, 0, 0, 5, 6]);
        s.player_data[Red].resources = Bundle::from_slice(&[2, 2, 0, 0, 0]);
        s.player_data[White].resources = Bundle::from_slice(&[1, 2, 0, 0, 0]);
        s.activate_dev_card(DevCard::Monopoly);

        s.apply_action(Monopolize(Brick));

        assert_eq!(s.player_data[Blue].resources[Brick], 5);
        assert_eq!(s.player_data[Orange].resources[Brick], 0);
        assert_eq!(s.player_data[Red].resources[Brick], 0);
        assert_eq!(s.player_data[White].resources[Brick], 0);
    }

    #[test]
    fn must_take_resources_after_playing_year_of_plenty() {
        let mut s = State::default();

        s.activate_dev_card(DevCard::YearOfPlenty);

        let actions = s.get_actions();
        assert!(
            actions.iter().all(|a| matches!(a, TakeFreeResource(_))),
            "all available actions should be TakeResource"
        );
    }

    #[test]
    fn take_resource_transfers_from_bank_to_player() {
        let mut s = State::default();
        s.player_data[Blue].resources = Bundle::splat(0);
        let bank_before = s.bank.resources[Brick];
        s.activate_dev_card(DevCard::YearOfPlenty);

        s.apply_action(TakeFreeResource(Brick));
        s.apply_action(TakeFreeResource(Brick));

        assert_eq!(s.player_data[Blue].resources[Brick], 2);
        assert_eq!(s.bank.resources[Brick], bank_before - 2);
    }

    #[test]
    fn newly_bought_dev_cards_are_not_playable() {
        let mut s = State::default();
        s.player_data[Blue].resources = Bundle::splat(5);
        s.apply_action(BuyDevCard);

        let actions = s.get_actions();

        assert!(!actions.iter().any(|a| matches!(a, PlayDevCard(_))));
    }

    #[test]
    fn max_one_dev_card_played_per_turn() {
        let mut s = State::default();
        s.apply_action(BuyDevCard);
        s.apply_action(BuyDevCard);
        s.locked_dev_cards = EnumMap::from_fn(|_| false);

        // Play first dev card
        let play_action = s
            .get_actions()
            .into_iter()
            .find(|a| matches!(a, PlayDevCard(_)))
            .unwrap();
        s.apply_action(play_action);

        let actions = s.get_actions();
        assert!(!actions.iter().any(|a| matches!(a, PlayDevCard(_))));
    }
}
