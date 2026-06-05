mod bitboard;
mod hex_board;
mod road_trails;
mod shared_data;

use enum_map::EnumMap;
use std::{
    simd::{cmp::SimdOrd, u8x8},
    sync::Arc,
};

use crate::{board::shared_data::*, bundle::Bundle, common::*};
pub use bitboard::Bitboard;
pub use hex_board::*;
pub use road_trails::{RoadTrailTable, RoadTrailTableLoader};

/// Implements behavior that relates to the Catan hex board.
pub struct Board {
    shared_data: Arc<SharedBoardData>,
    player_buildings: EnumMap<Player, Bitboard<V>>, // Idea: store city flags in 10 free bits
    player_roads: EnumMap<Player, Bitboard<E>>,
    player_settlement_slots: EnumMap<Player, Bitboard<V>>,
    player_road_slots: EnumMap<Player, Bitboard<E>>,
    settlement_slots: Bitboard<V>,
    road_slots: Bitboard<E>,
    cities: Bitboard<V>,
    robber_verts: Bitboard<V>, // 8 bytes to store 6 vertices? Or only store hex id but have to do a hex-to-vert lookup at runtime?
    robber: HexId,             // ...might have to store robber hex id anyway
    longest_roads: EnumMap<Player, u8>,
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Board {
            shared_data: Arc::clone(&self.shared_data),
            player_buildings: self.player_buildings.clone(),
            player_roads: self.player_roads.clone(),
            player_settlement_slots: self.player_settlement_slots.clone(),
            player_road_slots: self.player_road_slots.clone(),
            settlement_slots: self.settlement_slots.clone(),
            road_slots: self.road_slots.clone(),
            cities: self.cities.clone(),
            robber_verts: self.robber_verts.clone(),
            robber: self.robber.clone(),
            longest_roads: self.longest_roads.clone(),
        }
    }
}

impl Default for Board {
    fn default() -> Self {
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

        // Overwrite desert tile
        resources[9] = None;
        rolls[9] = None;

        Board::new(resources, rolls)
    }
}

impl Board {
    /// Creates and returns a new Board.
    ///
    /// Internally, a [HexBoard] is created and used to populate the bitboards.
    pub fn new(resources: Vec<Option<Resource>>, rolls: Vec<Option<u8>>) -> Self {
        let shared_data = SharedBoardData::new(resources, rolls);

        let center_hex_id = shared_data.hex_board.hex_ids[&Hex(0, 0)];

        Board {
            player_buildings: EnumMap::default(),
            player_roads: EnumMap::default(),
            player_settlement_slots: EnumMap::default(),
            player_road_slots: EnumMap::default(),
            settlement_slots: Bitboard::ones(),
            road_slots: Bitboard::ones(),
            cities: Bitboard::zeros(),
            robber_verts: ADJACENCY.hex_to_verts[center_hex_id],
            robber: center_hex_id,
            shared_data: Arc::new(shared_data),
            longest_roads: EnumMap::default(),
        }
    }

    // Helpers

    pub fn hex_id(&self, hex: Hex) -> HexId {
        self.shared_data.hex_board.hex_ids[&hex]
    }

    pub fn hex(&self, id: HexId) -> Hex {
        self.shared_data.hex_board.hexes[id]
    }

    pub fn vertex_id(&self, vertex: Vertex) -> VertexId {
        self.shared_data.hex_board.vertex_ids[&vertex]
    }

    pub fn vertex(&self, id: VertexId) -> Vertex {
        self.shared_data.hex_board.vertices[id]
    }

    pub fn edge_id(&self, edge: Edge) -> EdgeId {
        self.shared_data.hex_board.edge_ids[&edge]
    }

    pub fn edge(&self, id: EdgeId) -> Edge {
        self.shared_data.hex_board.edges[id]
    }

    pub fn players_on_hex(&self, hex_id: HexId) -> Vec<Player> {
        let verts = ADJACENCY.hex_to_verts[hex_id];
        PLAYERS
            .into_iter()
            .filter(|&p| !(self.player_buildings[p] & verts).is_zeros())
            .collect()
    }

    /// Get the set of vertices that receive a given resource on a given roll.
    fn roll_resource_vertices(&self, roll: u8, resource: Resource) -> Bitboard<V> {
        self.shared_data.roll_resources[(roll - 2) as usize][resource as usize]
    }

    pub fn game_map(&self) -> InitialObservation {
        InitialObservation {
            resources: self.shared_data.resources.clone(),
            rolls: self.shared_data.rolls.clone(),
        }
    }

    // Gameplay

    pub fn available_settlements(&self, player: Player) -> Bitboard<V> {
        self.settlement_slots & self.player_settlement_slots[player]
    }

    pub fn available_roads(&self, player: Player) -> Bitboard<E> {
        (self.road_slots & self.player_road_slots[player]).into()
    }

    pub fn buildings(&self, player: Player) -> Bitboard<V> {
        self.player_buildings[player]
    }

    pub fn settlements(&self, player: Player) -> Bitboard<V> {
        self.player_buildings[player] & !self.cities
    }

    pub fn cities(&self, player: Player) -> Bitboard<V> {
        self.player_buildings[player] & self.cities
    }

    pub fn roads(&self, player: Player) -> Bitboard<E> {
        self.player_roads[player]
    }

    pub fn longest_road(&self, player: Player) -> u8 {
        let enemy_buildings = player
            .enemies()
            .iter()
            .map(|&p| self.player_buildings[p])
            .reduce(|acc, e| acc | e)
            .unwrap();
        ROAD_TRAILS.longest_trail(self.roads(player), enemy_buildings)
    }

    pub fn add_settlement(&mut self, player: Player, vertex_id: VertexId) {
        self.player_buildings[player].add(vertex_id);
        // Mark vertex and neighbors as occupied
        // (Currently, same_to_same[i] includes i itself, not just its neighbors.)
        self.settlement_slots &= !ADJACENCY.vert_to_verts[vertex_id];
        // Allow adjacent roads
        let spokes = ADJACENCY.vert_to_edges[vertex_id];
        self.player_road_slots[player] |= spokes;

        // Check for split roads
        for enemy in player.enemies() {
            let enemy_roads = self.player_roads[enemy];
            let enemy_spokes = spokes & enemy_roads;

            for e in spokes & !enemy_roads {
                // If a roadless spoke just got completely disconnected from other roads,
                // disallow building on it
                if (ADJACENCY.edge_to_edges[e] & !spokes & enemy_roads).is_zeros() {
                    self.player_road_slots[enemy].remove(e);
                }
            }

            if enemy_spokes.count_ones() >= 2 {
                // Neighboring roads > 1 means
                // `player` just split 'enemy`'s road, which means
                // length must be recalculated
                self.longest_roads[enemy] = self.longest_road(enemy);
                break; // This can only happen to one enemy at a time
            }
        }
    }

    pub fn add_road(&mut self, player: Player, edge_id: EdgeId) {
        // Mark edge as occupied
        self.player_roads[player].add(edge_id);
        self.road_slots.remove(edge_id);

        // Allow adjacent settlements
        let verts = ADJACENCY.edge_to_verts[edge_id];
        self.player_settlement_slots[player] |= verts;

        // Allow adjacent roads
        let edges = ADJACENCY.edge_to_edges[edge_id];
        self.player_road_slots[player] |= edges;
        // Disallow roads blocked by enemy settlements
        for enemy in player.enemies() {
            let mut enemy_buildings = verts & self.player_buildings[enemy];
            if enemy_buildings.count_ones() == 1 {
                let building = enemy_buildings.next().unwrap();
                let building_edges = ADJACENCY.vert_to_edges[building];
                if let Some(edge_in_question) = (building_edges & !self.road_slots).next() {
                    if (ADJACENCY.edge_to_edges[edge_in_question] & self.player_roads[player])
                        .count_ones()
                        == 1
                    {
                        self.player_road_slots[player] &= !ADJACENCY.vert_to_edges[building];
                    }
                }
                break;
            }
        }
        self.longest_roads[player] = self.longest_road(player);
    }

    pub fn upgrade_settlement(&mut self, vertex_id: VertexId) {
        self.cities.add(vertex_id);
    }

    pub fn robber_hex_id(&self) -> HexId {
        self.robber
    }

    pub fn move_robber(&mut self, hex_id: HexId) {
        self.robber = hex_id;
        self.robber_verts = ADJACENCY.hex_to_verts[hex_id];
    }

    /// Returns the amount of each resource needed to trade with the bank (maritime trade).
    pub fn exchange_ratios(&self, player: Player) -> Bundle {
        // Default is 4:1. Generic harbors reduce it to 3:1.
        // Resource-specific harbors reduce it to 2:1 for that resource.
        let buildings = self.player_buildings[player];
        let mut ratios = Bundle::splat(
            if (buildings & self.shared_data.generic_harbors).is_zeros() {
                4
            } else {
                3
            },
        );
        for (res, harbors) in self.shared_data.resource_harbors {
            if !(buildings & harbors).is_zeros() {
                ratios[res] = 2;
            }
        }
        ratios
    }

    pub fn produce_resources(&self, roll: u8, in_stock: Bundle) -> EnumMap<Player, Bundle> {
        // TODO: optimize
        let mut player_bundles: EnumMap<Player, Bundle> = EnumMap::default();

        for res in RESOURCES {
            let mut resource_verts = self.roll_resource_vertices(roll, res);
            resource_verts &= !self.robber_verts;
            let mut bundle = Bundle::default();

            for (player, buildings) in self.player_buildings {
                let first_pass = resource_verts & buildings;
                let amount =
                    (first_pass.count_ones() + (first_pass & self.cities).count_ones()) as u8;
                bundle[player] = amount;
            }

            if bundle.reduce_sum() > in_stock[res] {
                bundle.data = bundle.data.simd_min(u8x8::splat(in_stock[res]));
                if bundle.count_nonzero() > 1 {
                    // No one gets any
                    bundle = Bundle::splat(0);
                }
            }

            for player in PLAYERS {
                player_bundles[player][res] = bundle[player];
            }
        }
        player_bundles
    }

    // City flags in leftover bits would save 1 lanewise popcnt and 1 bitwise AND
    // https://stackoverflow.com/questions/51104493/is-it-possible-to-popcount-m256i-and-store-result-in-8-32-bit-words-instead-of

    /// Returns the victory points a player gets from buildings and the longest road marker.
    pub fn victory_points(&self, player: Player) -> u32 {
        let buildings = self.player_buildings[player];
        buildings.count_ones() + (buildings & self.cities).count_ones()
        // TODO: longest road
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enum_map::{Enum, enum_map};

    // Helpers

    fn sett(b: &mut Board, p: Player, v: Vertex) {
        b.add_settlement(p, b.vertex_id(v));
    }

    fn road(b: &mut Board, p: Player, e: Edge) {
        b.add_road(p, b.edge_id(e));
    }

    fn city(b: &mut Board, v: Vertex) {
        b.upgrade_settlement(b.vertex_id(v));
    }

    fn to_bundles(map: EnumMap<Player, [u8; Resource::LENGTH]>) -> EnumMap<Player, Bundle> {
        map.map(|_, arr| Bundle::from(arr.as_slice()))
    }

    fn add_settlements_from_hex(board: &mut Board, player: Player, hex: &str) {
        let verts: Bitboard<V> = Bitboard::from_hex(hex);
        for v in verts {
            board.add_settlement(player, v);
        }
    }

    fn add_roads_from_hex(board: &mut Board, player: Player, hex: &str) {
        let edges: Bitboard<E> = Bitboard::from_hex(hex);
        for e in edges {
            board.add_road(player, e);
        }
    }

    /// *Illustration A* in the [Catan manual]
    ///
    /// [Catan manual]: https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf
    fn setup() -> Board {
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

        let mut b = Board::new(resources, rolls);
        let mut s = |p: Player, v: Vertex| sett(&mut b, p, v);

        s(Blue, Vertex(-2, 2, N));
        s(Blue, Vertex(0, 2, N));
        s(Orange, Vertex(2, -2, S));
        s(Orange, Vertex(-1, 2, N));
        s(Red, Vertex(0, -1, N));
        s(Red, Vertex(-2, 1, N));
        s(White, Vertex(-1, 0, N));
        s(White, Vertex(1, 1, N));

        let mut r = |p: Player, e: Edge| road(&mut b, p, e);
        r(Blue, Edge(-2, 2, NE));
        r(Blue, Edge(1, 1, W));
        r(Orange, Edge(-1, 2, NE));
        r(Orange, Edge(1, -1, NE));
        r(Red, Edge(-2, 1, NE));
        r(Red, Edge(0, -1, NE));
        r(White, Edge(-1, 0, NW));
        r(White, Edge(2, 0, W));

        b
    }

    #[test]
    fn resource_production() {
        let b = setup();
        let roll = 6;

        let in_stock = Bundle::splat(20);
        let production = b.produce_resources(roll, in_stock);
        let expected = to_bundles(enum_map! {
            Blue => [0; 5],
            Orange => [0, 1, 0, 0, 0],
            Red => [1, 0, 0, 0, 0],
            White => [1, 0, 0, 0, 0]
        });
        assert_eq!(production, expected);
    }

    #[test]
    fn resource_production_cities() {
        let mut b = setup();

        sett(&mut b, Red, Vertex(0, -2, N));
        sett(&mut b, Red, Vertex(-1, -1, N));
        city(&mut b, Vertex(0, -2, N));
        city(&mut b, Vertex(-1, -1, N));

        city(&mut b, Vertex(0, -1, N));

        let in_stock = Bundle::splat(20);
        let production = b.produce_resources(10, in_stock);
        let expected = to_bundles(enum_map! {
            Blue => [0; 5],
            Orange => [1, 0, 0, 0, 0],
            Red => [0, 0, 0, 6, 0],
            White => [0; 5]
        });
        assert_eq!(production, expected);
    }

    #[test]
    fn insufficient_resource_production() {
        // 2 ore produced for more than 1 player; 1 in stock; no one gets any
        // 1 brick produced; 1 in stock; orange gets it per usual
        let mut b = setup();

        sett(&mut b, Blue, Vertex(0, -2, N));

        let in_stock = Bundle::splat(1);
        let production = b.produce_resources(10, in_stock);
        let expected = to_bundles(enum_map! {
            Blue => [0; 5],
            Orange => [1, 0, 0, 0, 0],
            Red => [0; 5],
            White => [0; 5]
        });
        assert_eq!(production, expected);
    }

    #[test]
    fn insufficient_resource_production_exception() {
        // 2 ore produced for 1 player; 1 in stock; player gets 1
        // 1 brick produced; 1 in stock; orange gets it per usual
        let mut b = setup();

        city(&mut b, Vertex(0, -1, N));

        let in_stock = Bundle::splat(1);
        let production = b.produce_resources(10, in_stock);
        let expected = to_bundles(enum_map! {
            Blue => [0; 5],
            Orange => [1, 0, 0, 0, 0],
            Red => [0, 0, 0, 1, 0],
            White => [0; 5]
        });
        assert_eq!(production, expected);
    }

    #[test]
    fn no_desert_production() {
        let mut b = setup();

        let in_stock = Bundle::splat(10);
        let before = b.produce_resources(7, in_stock);

        sett(&mut b, Red, Vertex(0, 0, N));

        let after = b.produce_resources(7, in_stock);

        assert_eq!(before, after);
        assert_eq!(after[Red], Bundle::splat(0));
    }

    #[test]
    fn robber_prevents_production() {
        let mut b = setup();

        let in_stock = Bundle::splat(10);
        let before = b.produce_resources(6, in_stock);

        b.move_robber(b.hex_id(Hex(0, -1)));

        let after = b.produce_resources(6, in_stock);

        assert_ne!(before, after);
        assert_eq!(
            after,
            to_bundles(enum_map! {
                Blue => [0; 5],
                Orange => [0, 1, 0, 0, 0],
                Red => [0; 5],
                White => [0; 5]
            })
        );
    }

    #[test]
    fn no_adjacent_settlements() {
        let b = setup();

        // No one can build anything
        assert!(b.available_settlements(Blue).next().is_none());
        assert!(b.available_settlements(Orange).next().is_none());
        assert!(b.available_settlements(Red).next().is_none());
        assert!(b.available_settlements(White).next().is_none());
    }

    #[test]
    fn placeable_settlement() {
        let mut b = setup();

        road(&mut b, Red, Edge(1, -1, W));
        let settlement_vert = b.vertex_id(Vertex(0, 0, N));
        let red_verts: Vec<VertexId> = b.available_settlements(Red).collect();

        // Red can only build this settlement. The rest can't build any settlements.
        assert_eq!(red_verts, vec![settlement_vert]);
        assert!(b.available_settlements(Blue).next().is_none());
        assert!(b.available_settlements(Orange).next().is_none());
        assert!(b.available_settlements(White).next().is_none());
    }

    #[test]
    fn no_road_from_enemy_building() {
        let b = setup();

        // Only Red can build this road
        let road = b.edge_id(Edge(0, -1, NW));
        assert!(!b.available_roads(Orange).contains(road));
        assert!(b.available_roads(Red).contains(road));
        assert!(!b.available_roads(Blue).contains(road));
        assert!(!b.available_roads(White).contains(road));
    }

    #[test]
    fn no_road_from_enemy_road() {
        let b = setup();

        // Only Red and Orange can build this road
        let road = b.edge_id(Edge(1, -1, NW));
        assert!(b.available_roads(Orange).contains(road));
        assert!(b.available_roads(Red).contains(road));
        assert!(!b.available_roads(Blue).contains(road));
        assert!(!b.available_roads(White).contains(road));
    }

    #[test]
    fn no_out_of_bounds_roads() {
        let mut b = setup();

        road(&mut b, Red, Edge(1, -2, W));
        road(&mut b, Red, Edge(0, -2, NE));

        // Assert all placeable roads are on edges with ids in 0..72
        let placeable = b.available_roads(Red);
        assert!(placeable < Bitboard::new(1 << N_EDGES));
    }

    #[test]
    fn players_on_hex() {
        let b = setup();
        let hex_id = b.hex_id(Hex(-2, 1));

        let players = b.players_on_hex(hex_id);

        assert_eq!(players, vec![Blue, Red]);
    }

    #[test]
    fn enemy_settlement_splits_road() {
        let mut b = Board::default();
        add_roads_from_hex(&mut b, Blue, "18001800000000");
        assert_eq!(b.longest_roads[Blue], 4, "longest road before split");

        add_settlements_from_hex(&mut b, Orange, "400000");
        assert_eq!(b.longest_roads[Blue], 2, "longest road after split");
    }

    #[test]
    fn old_enemy_settlement_prevents_new_road() {
        let mut b = Board::default();
        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));
        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));

        assert!(!b.available_roads(Blue).contains(b.edge_id(Edge(0, 0, NE))));
    }

    #[test]
    fn new_enemy_settlement_prevents_new_road() {
        let mut b = Board::default();
        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));
        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));

        assert!(!b.available_roads(Blue).contains(b.edge_id(Edge(0, 0, NE))));
    }

    #[test]
    fn old_enemy_settlement_allows_road_that_goes_around() {
        let mut b = Board::default();
        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));
        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));
        b.add_road(Blue, b.edge_id(Edge(-1, 0, NE)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, W)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NW)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NE)));

        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));
    }

    #[test]
    fn new_enemy_settlement_allows_road_that_goes_around() {
        let mut b = Board::default();
        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));
        b.add_road(Blue, b.edge_id(Edge(-1, 0, NE)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, W)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NW)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NE)));

        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));

        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));
    }

    #[test]
    fn old_enemy_settlement_allows_road_that_goes_around2() {
        let mut b = Board::default();
        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));
        b.add_road(Blue, b.edge_id(Edge(-1, 0, NE)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, W)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NW)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NE)));

        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(0, 0, NW))));
        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));

        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));
        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));
    }

    #[test]
    fn new_enemy_settlement_allows_road_that_goes_around2() {
        let mut b = Board::default();
        b.add_road(Blue, b.edge_id(Edge(-1, 0, NE)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, W)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NW)));
        b.add_road(Blue, b.edge_id(Edge(0, -1, NE)));

        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(0, 0, NW))));
        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));

        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));

        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(0, 0, NW))));
        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));

        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));

        assert!(b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));
    }

    #[test]
    fn enemy_settlement_prevents_road_branch() {
        let mut b = Board::default();
        b.add_road(Blue, b.edge_id(Edge(0, 0, NW)));
        b.add_road(Blue, b.edge_id(Edge(0, 0, NE)));

        b.add_settlement(Orange, b.vertex_id(Vertex(0, 0, N)));

        assert!(!b.available_roads(Blue).contains(b.edge_id(Edge(1, -1, W))));
    }
}
