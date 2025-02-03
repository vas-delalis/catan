mod bitboard;

use enum_map::{enum_map, Enum, EnumMap};
use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    fmt::Debug,
    simd::{cmp::SimdOrd, u8x8},
    sync::Arc,
};

use crate::{bundle::Bundle, common::*};
use bitboard::Bitboard;
pub use EdgeDir::*;
pub use VertexDir::*;

type V = u64;
type E = u128;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct Hex(pub i8, pub i8);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Vertex(pub i8, pub i8, pub VertexDir);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Edge(pub i8, pub i8, pub EdgeDir);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexDir {
    N,
    S,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum EdgeDir {
    W,
    NW,
    NE,
}

impl Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {:?})", self.0, self.1, self.2)
    }
}

impl Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {:?})", self.0, self.1, self.2)
    }
}

impl Hex {
    fn vertices(&self) -> Vec<Vertex> {
        let &Hex(q, r) = self;
        [
            Vertex(q, r, N),
            Vertex(q, r, S),
            Vertex(q, r + 1, N),
            Vertex(q, r - 1, S),
            Vertex(q - 1, r + 1, N),
            Vertex(q + 1, r - 1, S),
        ]
        .to_vec()
    }

    fn edges(&self) -> Vec<Edge> {
        let &Hex(q, r) = self;
        [
            Edge(q, r, W),
            Edge(q, r, NW),
            Edge(q, r, NE),
            Edge(q + 1, r, W),
            Edge(q, r + 1, NW),
            Edge(q - 1, r + 1, NE),
        ]
        .to_vec()
    }
}

impl Vertex {
    fn coords(&self) -> (f64, f64) {
        let &Vertex(q, r, dir) = self;
        let (dq, dr) = {
            match dir {
                N => (1.0 / 3.0, -2.0 / 3.0),
                S => (-1.0 / 3.0, 2.0 / 3.0),
            }
        };
        let q = (q as f64) + dq;
        let r = (r as f64) + dr;
        (q, r)
    }

    fn ordering_value(&self) -> f64 {
        let (q, r) = self.coords();
        3.0 * q + 21.0 * r.ceil()
    }

    /// Returns neighboring vertices.
    fn neighbors(&self) -> [Vertex; 3] {
        let &Vertex(q, r, dir) = self;
        match dir {
            N => [
                Vertex(q + 1, r - 2, S),
                Vertex(q, r - 1, S),
                Vertex(q + 1, r - 1, S),
            ],
            S => [
                Vertex(q - 1, r + 1, N),
                Vertex(q - 1, r + 2, N),
                Vertex(q, r + 1, N),
            ],
        }
    }

    /// Returns the edges that "protrude" from this vertex.
    fn edges(&self) -> [Edge; 3] {
        let &Vertex(q, r, dir) = self;
        match dir {
            N => [Edge(q, r, NE), Edge(q, r, NW), Edge(q + 1, r - 1, W)],
            S => [Edge(q, r + 1, W), Edge(q, r + 1, W), Edge(q - 1, r + 1, NE)],
        }
    }
}

impl Edge {
    /// Returns neighboring edges.
    fn neighbors(&self) -> [Edge; 4] {
        let &Edge(q, r, dir) = self;
        match dir {
            NE => [
                Edge(q, r, NW),
                Edge(q + 1, r, W),
                Edge(q + 1, r, NW),
                Edge(q + 1, r - 1, W),
            ],
            NW => [
                Edge(q, r, W),
                Edge(q, r, NE),
                Edge(q - 1, r, NE),
                Edge(q + 1, r - 1, W),
            ],
            W => [
                Edge(q, r, NW),
                Edge(q - 1, r, NE),
                Edge(q - 1, r + 1, NW),
                Edge(q - 1, r + 1, NE),
            ],
        }
    }

    /// Returns the endpoints of this edge.
    fn vertices(&self) -> [Vertex; 2] {
        let &Edge(q, r, dir) = self;
        match dir {
            NE => [Vertex(q, r, N), Vertex(q + 1, r - 1, S)],
            NW => [Vertex(q, r, N), Vertex(q, r - 1, S)],
            W => [Vertex(q, r - 1, S), Vertex(q - 1, r + 1, N)],
        }
    }
}

// TODO: Which adjacency maps do we need?
// (A adjacent to each B)
// A / B |    Hex    |     Vert     |     Edge
// ----------------------------------------------
// Hexes |     ?     |       ?      |      ?
// Verts |  Robber   | Sett. plcmnt | Sett. plcmnt
// Edges |     ?     | Road plcmnt  | Road plcmnt

/// Data -- mainly bitmaps -- that doesn't depend on the game state.
struct SharedBoardData {
    hex_to_verts: Vec<Bitboard<V>>,
    vert_to_verts: Vec<Bitboard<V>>,
    edge_to_verts: Vec<Bitboard<V>>,

    vert_to_edges: Vec<Bitboard<E>>,
    edge_to_edges: Vec<Bitboard<E>>,

    roll_resources: Vec<Vec<Bitboard<V>>>, // Vertices that receive a given resource on a given roll
    generic_harbors: Bitboard<V>,
    resource_harbors: EnumMap<Resource, Bitboard<V>>,
    simple_board: SimpleBoard,
} // TODO: use arrays? (fixed length is good)

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
        }
    }
}

impl Board {
    /// Creates and returns a new Board.
    ///
    /// Internally, a [SimpleBoard] is created and used to populate the bitboards.
    pub fn new(resources: Vec<Option<Resource>>, rolls: Vec<Option<u8>>) -> Self {
        // TODO: return Result instead
        let sb = SimpleBoard::new();

        let mut hex_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_HEXES];
        let mut roll_resources: Vec<Vec<Bitboard<V>>> =
            vec![vec![Bitboard::zeros(); Resource::LENGTH]; N_ROLLS];

        // Populate hex-to-vert maps and roll-resource maps
        for (i, hex) in sb.hexes.iter().enumerate() {
            let adj = sb.vert_bitboard(&hex.vertices());
            hex_to_verts[i] = adj;

            if let Some(resource) = resources[i] {
                let roll = rolls[i].expect("hexes with a resource should also have a roll");
                roll_resources[(roll - 2) as usize][resource as usize] |= adj;
            }
        }

        let mut vert_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_VERTICES];
        let mut vert_to_edges: Vec<Bitboard<E>> = vec![Bitboard::zeros(); N_VERTICES];

        // Populate vert-to-* maps
        for (vert, &i) in sb.vertex_ids.iter() {
            vert_to_edges[i] = sb.edge_bitboard(&vert.edges());
            vert_to_verts[i] = sb.vert_bitboard(&vert.neighbors());
            vert_to_verts[i].add(i); // Include self
        }

        let mut edge_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_EDGES];
        let mut edge_to_edges: Vec<Bitboard<E>> = vec![Bitboard::zeros(); N_EDGES];

        // Populate edge-to-* maps
        for (edge, &i) in sb.edge_ids.iter() {
            edge_to_verts[i] = sb.vert_bitboard(&edge.vertices());
            edge_to_edges[i] = sb.edge_bitboard(&edge.neighbors());
            edge_to_edges[i].add(i); // Include self
        }

        // Populate harbor maps
        // TODO: make customizable
        let resource_harbors = enum_map! {
            Brick => Edge(-2, 1, W),
            Grain => Edge(1, -2, NE),
            Lumber => Edge(-1, -1, W),
            Ore => Edge(2, -1, NE),
            Wool => Edge(1, 2, NW)
        }
        .map(|_, edge| {
            let id = sb
                .edge_ids
                .get(&edge)
                .expect("harbor edges should be valid");
            edge_to_verts[*id]
        });

        let mut generic_harbors = Bitboard::zeros();

        for edge in [
            Edge(0, -2, NW),
            Edge(3, 0, W),
            Edge(-1, 3, NW),
            Edge(-3, 3, NE),
        ] {
            let id = sb
                .edge_ids
                .get(&edge)
                .expect("harbor edges should be valid");
            let ends = edge_to_verts[*id];
            generic_harbors |= ends;
        }

        let shared_data = SharedBoardData {
            hex_to_verts,
            vert_to_verts,
            edge_to_verts,

            vert_to_edges,
            edge_to_edges,

            roll_resources,
            generic_harbors,
            resource_harbors,
            simple_board: sb,
        };

        let center_hex_id = shared_data.simple_board.hex_ids[&Hex(0, 0)];

        Board {
            player_buildings: EnumMap::default(),
            player_roads: EnumMap::default(),
            player_settlement_slots: EnumMap::default(),
            player_road_slots: EnumMap::default(),
            settlement_slots: Bitboard::ones(),
            road_slots: Bitboard::ones(),
            cities: Bitboard::zeros(),
            robber_verts: shared_data.hex_to_verts[center_hex_id],
            robber: center_hex_id,
            shared_data: Arc::new(shared_data),
        }
    }

    // Helpers

    pub fn hex_id(&self, hex: Hex) -> HexId {
        self.shared_data.simple_board.hex_ids[&hex]
    }

    pub fn hex(&self, id: HexId) -> Hex {
        self.shared_data.simple_board.hexes[id]
    }

    pub fn vertex_id(&self, vertex: Vertex) -> VertexId {
        self.shared_data.simple_board.vertex_ids[&vertex]
    }

    pub fn vertex(&self, id: VertexId) -> Vertex {
        self.shared_data.simple_board.vertices[id]
    }

    pub fn edge_id(&self, edge: Edge) -> EdgeId {
        self.shared_data.simple_board.edge_ids[&edge]
    }

    pub fn edge(&self, id: EdgeId) -> Edge {
        self.shared_data.simple_board.edges[id]
    }

    pub fn players_on_hex(&self, hex_id: HexId) -> Vec<Player> {
        let verts = self.shared_data.hex_to_verts[hex_id];
        PLAYERS
            .into_iter()
            .filter(|&p| (self.player_buildings[p] & verts) > Bitboard::zeros())
            .collect()
    }

    /// Get the set of vertices that receive a given resource on a given roll.
    fn roll_resource_vertices(&self, roll: u8, resource: Resource) -> Bitboard<V> {
        self.shared_data.roll_resources[(roll - 2) as usize][resource as usize]
    }

    // Gameplay

    pub fn available_settlements(&self, player: Player) -> Bitboard<V> {
        self.settlement_slots & self.player_settlement_slots[player]
    }

    pub fn available_roads(&self, player: Player) -> Bitboard<E> {
        (self.road_slots & self.player_road_slots[player]).into()
    }

    pub fn available_cities(&self, player: Player) -> Bitboard<V> {
        self.player_buildings[player] & !self.cities
    }

    pub fn add_settlement(&mut self, player: Player, vertex_id: VertexId) {
        self.player_buildings[player].add(vertex_id);
        // Mark vertex and neighbors as occupied
        // (Currently, same-to-same[i] includes i itself, not just its neighbors.)
        self.settlement_slots &= !self.shared_data.vert_to_verts[vertex_id];
        // Allow adjacent roads
        self.player_road_slots[player] |= self.shared_data.vert_to_edges[vertex_id];
    }

    pub fn add_road(&mut self, player: Player, edge_id: EdgeId) {
        // Mark edge as occupied
        self.player_roads[player].add(edge_id);
        self.road_slots.remove(edge_id);
        // Allow adjacent settlements
        self.player_settlement_slots[player] |= self.shared_data.edge_to_verts[edge_id];
        // Allow adjacent roads
        self.player_road_slots[player] |= self.shared_data.edge_to_edges[edge_id];

        // TODO: longest road
    }

    pub fn upgrade_settlement(&mut self, vertex_id: VertexId) {
        self.cities.add(vertex_id);
    }

    pub fn robber_hex_id(&self) -> HexId {
        self.robber
    }

    pub fn move_robber(&mut self, hex_id: HexId) {
        self.robber = hex_id;
        self.robber_verts = self.shared_data.hex_to_verts[hex_id];
    }

    /// Returns the amount of each resource needed to trade with the bank (maritime trade).
    pub fn exchange_ratios(&self, player: Player) -> Bundle {
        // Default is 4:1. Generic harbors reduce it to 3:1.
        // Resource-specific harbors reduce it to 2:1 for that resource.
        let buildings = self.player_buildings[player];
        let mut ratios = Bundle::splat(
            if buildings & self.shared_data.generic_harbors > Bitboard::zeros() {
                3
            } else {
                4
            },
        );
        for (res, harbors) in self.shared_data.resource_harbors {
            if buildings & harbors > Bitboard::zeros() {
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

    // City flags in leftover bits would save 1 lanewise popcnt and 1 bitwise and
    // https://stackoverflow.com/questions/51104493/is-it-possible-to-popcount-m256i-and-store-result-in-8-32-bit-words-instead-of

    /// Returns the victory points a player gets from buildings and the longest road marker.
    pub fn victory_points(&self, player: Player) -> u32 {
        let buildings = self.player_buildings[player];
        buildings.count_ones() + (buildings & self.cities).count_ones()
        // TODO: longest road
    }
}

/// A high-level representation of the Catan board. Used to generate the much more efficient [Board].
#[derive(Debug)]
struct SimpleBoard {
    hexes: Vec<Hex>,
    hex_ids: HashMap<Hex, HexId>,
    vertices: Vec<Vertex>,
    vertex_ids: HashMap<Vertex, VertexId>,
    edges: Vec<Edge>,
    edge_ids: HashMap<Edge, EdgeId>,
}

impl SimpleBoard {
    fn new() -> Self {
        let n: i8 = 2; // Grid size
        let mut hexes = Vec::with_capacity(N_HEXES);
        let mut vertices = HashSet::with_capacity(N_VERTICES);
        let mut edges: HashSet<Edge> = HashSet::with_capacity(N_EDGES);

        for r in -n..=n {
            let q1 = max(-n, -r - n);
            let q2 = min(n, -r + n);
            for q in q1..=q2 {
                let hex = Hex(q, r);
                hexes.push(hex);
                for v in hex.vertices() {
                    vertices.insert(v);
                }
                for e in hex.edges() {
                    edges.insert(e);
                }
            }
        }

        let hex_ids: HashMap<Hex, HexId> =
            hexes.iter().enumerate().map(|(id, &h)| (h, id)).collect();

        let mut vertices: Vec<Vertex> = vertices.into_iter().collect();
        vertices.sort_by(|a, b| a.ordering_value().total_cmp(&b.ordering_value()));
        let vertex_ids: HashMap<Vertex, VertexId> = vertices
            .iter()
            .enumerate()
            .map(|(id, &v)| (v, id))
            .collect();

        let edges: Vec<Edge> = edges.into_iter().collect();
        // TODO: sort edges
        let edge_ids: HashMap<Edge, EdgeId> =
            edges.iter().enumerate().map(|(id, &e)| (e, id)).collect();

        SimpleBoard {
            hexes,
            hex_ids,
            vertices,
            vertex_ids,
            edges,
            edge_ids,
        }
    }

    /// Converts a list of vertices to the corresponding bitmap.
    pub fn vert_bitboard(&self, vertices: &[Vertex]) -> Bitboard<V> {
        let mut bitboard = Bitboard::zeros();
        for v in vertices {
            if let Some(&id) = self.vertex_ids.get(v) {
                bitboard.add(id);
            }
        }
        bitboard
    }

    pub fn edge_bitboard(&self, edges: &[Edge]) -> Bitboard<E> {
        let mut bitboard = Bitboard::zeros();
        for e in edges {
            if let Some(&id) = self.edge_ids.get(e) {
                bitboard.add(id);
            }
        }
        bitboard
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enum_map::enum_map;

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
}
