mod bitboard;

use enum_map::{Enum, EnumMap};
use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    fmt::Debug,
    rc::Rc,
};

use crate::{bundle::Bundle, common::*};
use bitboard::BitIterator;
use EdgeDir::*;
use VertexDir::*;

#[derive(Debug, Clone, Copy)]
struct Hex(i8, i8);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Vertex(i8, i8, VertexDir);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Edge(i8, i8, EdgeDir);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum VertexDir {
    N,
    S,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum EdgeDir {
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
    hex_to_verts: Vec<VertexMap>,
    vert_to_verts: Vec<VertexMap>,
    edge_to_verts: Vec<VertexMap>,

    vert_to_edges: Vec<EdgeMap>,
    edge_to_edges: Vec<EdgeMap>,

    roll_resources: Vec<Vec<VertexMap>>, // Vertices that receive a given resource on a given roll
    simple_board: SimpleBoard,
} // TODO: use arrays? (fixed length is good)

/// Implements behavior that relates to the Catan hex board.
pub struct Board {
    shared_data: Rc<SharedBoardData>,
    player_buildings: EnumMap<Player, VertexMap>,
    player_roads: EnumMap<Player, EdgeMap>,
    player_settlement_slots: EnumMap<Player, VertexMap>,
    player_road_slots: EnumMap<Player, EdgeMap>,
    settlement_slots: VertexMap,
    road_slots: EdgeMap,
    cities: VertexMap,
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Board {
            shared_data: Rc::clone(&self.shared_data),
            player_buildings: self.player_buildings.clone(),
            player_roads: self.player_roads.clone(),
            player_settlement_slots: self.player_settlement_slots.clone(),
            player_road_slots: self.player_road_slots.clone(),
            settlement_slots: self.settlement_slots.clone(),
            road_slots: self.road_slots.clone(),
            cities: self.cities.clone(),
        }
    }
}

impl Board {
    /// Creates and returns a new Board.
    ///
    /// Internally, a [SimpleBoard] is created and used to populate the bitboards.
    pub fn new(resources: Vec<Resource>, rolls: Vec<u8>) -> Self {
        let simple_board = SimpleBoard::new();

        let mut roll_resources: Vec<Vec<VertexMap>> = vec![vec![0u64; Resource::LENGTH]; N_ROLLS];

        let mut hex_to_verts: Vec<VertexMap> = vec![0; N_HEXES];
        let mut vert_to_verts: Vec<VertexMap> = vec![0; N_VERTICES];
        let mut edge_to_verts: Vec<VertexMap> = vec![0; N_EDGES];

        let mut edge_to_edges = vec![0; N_EDGES];
        let mut vert_to_edges = vec![0; N_VERTICES];

        // Populate hex-to-vert maps and roll-resource maps
        for (i, hex) in simple_board.hexes.iter().enumerate() {
            let adj = simple_board.vert_bitmap(&hex.vertices());
            hex_to_verts[i] = adj;
            roll_resources[(rolls[i] - 2) as usize][resources[i] as usize] |= adj;
        }

        // Populate vert-to-* maps
        for (vert, &i) in simple_board.vertex_ids.iter() {
            vert_to_edges[i] = simple_board.edge_bitmap(&vert.edges());
            vert_to_verts[i] = simple_board.vert_bitmap(&vert.neighbors());
            vert_to_verts[i] |= 1 << i; // Include self
        }

        // Populate edge-to-* maps
        for (edge, &i) in simple_board.edge_ids.iter() {
            edge_to_verts[i] = simple_board.vert_bitmap(&edge.vertices());
            edge_to_edges[i] = simple_board.edge_bitmap(&edge.neighbors());
            edge_to_edges[i] |= 1 << i; // Include self
        }

        let shared_data = SharedBoardData {
            hex_to_verts,
            vert_to_verts,
            edge_to_verts,

            vert_to_edges,
            edge_to_edges,

            roll_resources,
            simple_board,
        };
        Board {
            shared_data: Rc::new(shared_data),
            player_buildings: EnumMap::default(),
            player_roads: EnumMap::default(),
            player_settlement_slots: EnumMap::default(),
            player_road_slots: EnumMap::default(),
            settlement_slots: !VertexMap::default(),
            road_slots: !EdgeMap::default(),
            cities: VertexMap::default(),
        }
    }

    // Helpers

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

    // Gameplay

    pub fn available_settlements(&self, player: Player) -> Vec<VertexId> {
        // TODO: return iterator
        let bitboard = self.settlement_slots & self.player_settlement_slots[player];
        BitIterator::new(bitboard).collect()
    }

    pub fn available_roads(&self, player: Player) -> Vec<EdgeId> {
        // TODO: return iterator
        let bitboard = self.road_slots & self.player_road_slots[player];
        BitIterator::new(bitboard).collect()
    }

    pub fn add_settlement(&mut self, player: Player, vertex_id: VertexId) {
        self.player_buildings[player] |= 1 << vertex_id;
        // Mark vertex and neighbors as occupied
        // (Currently, same-to-same[i] includes i itself, not just its neighbors.)
        self.settlement_slots &= !self.shared_data.vert_to_verts[vertex_id];
        // Allow adjacent roads
        self.player_road_slots[player] |= self.shared_data.vert_to_edges[vertex_id];
    }

    pub fn add_road(&mut self, player: Player, edge_id: EdgeId) {
        let delta = 1 << edge_id;
        // Mark edge as occupied
        self.player_roads[player] |= delta;
        self.road_slots &= !delta;
        // Allow adjacent settlements
        self.player_settlement_slots[player] |= self.shared_data.edge_to_verts[edge_id];
        // Allow adjacent roads
        self.player_road_slots[player] |= self.shared_data.edge_to_edges[edge_id];
    }

    pub fn upgrade_settlement(&mut self, vertex_id: VertexId) {
        self.cities |= 1 << vertex_id;
    }

    pub fn produce_resources(&self, roll: u8) -> EnumMap<Player, Bundle> {
        // TODO: Vectorize?
        let mut bundles: EnumMap<Player, Bundle> = EnumMap::default();

        for res in RESOURCES {
            let resource_map: VertexMap =
                self.shared_data.roll_resources[(roll - 2) as usize][res as usize];
            for (player, buildings) in self.player_buildings {
                let first_pass: VertexMap = resource_map & buildings;
                let amount = first_pass.count_ones() + (first_pass & self.cities).count_ones();
                bundles[player][res] = amount as u8;
            }
        }
        bundles
    }

    pub fn victory_points(&self, player: Player) -> u32 {
        let buildings: VertexMap = self.player_buildings[player];
        buildings.count_ones() + (buildings & self.cities).count_ones()
    }
}

/// A high-level representation of the Catan board.
/// Used to generate the much more efficient [Board].
#[derive(Debug)]
struct SimpleBoard {
    hexes: Vec<Hex>,
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

        let mut vertices: Vec<Vertex> = vertices.into_iter().collect();
        vertices.sort_by(|a, b| a.ordering_value().total_cmp(&b.ordering_value()));
        let vertex_ids: HashMap<Vertex, usize> = vertices
            .iter()
            .enumerate()
            .map(|(id, &v)| (v, id))
            .collect();

        let edges: Vec<Edge> = edges.into_iter().collect();
        // TODO: sort edges
        let edge_ids: HashMap<Edge, usize> =
            edges.iter().enumerate().map(|(id, &e)| (e, id)).collect();

        SimpleBoard {
            hexes,
            vertices,
            vertex_ids,
            edges,
            edge_ids,
        }
    }

    /// Converts a list of vertices to the corresponding bitmap.
    pub fn vert_bitmap(&self, vertices: &[Vertex]) -> VertexMap {
        vertices
            .into_iter()
            .fold(0u64, |bitmap, v| match self.vertex_ids.get(&v) {
                Some(v_id) => bitmap | (1 << v_id),
                _ => bitmap,
            })
    }

    pub fn edge_bitmap(&self, edges: &[Edge]) -> EdgeMap {
        edges
            .into_iter()
            .fold(0u128, |bitmap, e| match self.edge_ids.get(e) {
                Some(e_id) => bitmap | (1 << e_id),
                _ => bitmap,
            })
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

    /// *Illustration A* in the [Catan manual](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf)
    fn setup() -> Board {
        let resources = vec![
            Ore, Wool, Lumber, Grain, Brick, Wool, Brick, Grain, Lumber, Ore, Lumber, Ore, Lumber,
            Ore, Grain, Wool, Brick, Grain, Wool,
        ];
        let rolls: Vec<u8> = vec![10, 2, 9, 12, 6, 4, 10, 9, 11, 7, 3, 8, 8, 3, 4, 5, 5, 6, 11];

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

        let production = b.produce_resources(roll);
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

        let production = b.produce_resources(10);
        let expected = to_bundles(enum_map! {
            Blue => [0; 5],
            Orange => [1, 0, 0, 0, 0],
            Red => [0, 0, 0, 6, 0],
            White => [0; 5]
        });
        assert_eq!(production, expected);
    }

    #[test]
    fn no_adjacent_settlements() {
        let b = setup();

        let total_available: usize = PLAYERS
            .into_iter()
            .map(|p| b.available_settlements(p).len())
            .sum();
        assert_eq!(total_available, 0);
    }

    #[test]
    fn placeable_settlement() {
        let mut b = setup();

        road(&mut b, Red, Edge(1, -1, W));

        // Red and only Red can build this settlement
        let red = b.available_settlements(Red);
        let rest = [Blue, Orange, White]
            .map(|p| b.available_settlements(p))
            .concat();

        assert_eq!(red, vec![b.vertex_id(Vertex(0, 0, N))]);
        assert_eq!(rest, vec![]);
    }

    #[test]
    fn no_road_from_enemy_building() {
        let b = setup();

        // Only Red can build this road
        let road = b.edge_id(Edge(0, -1, NW));
        let mut edges = [Blue, Orange, White]
            .into_iter()
            .flat_map(|p| b.available_roads(p));

        assert!(edges.find(|&e| e == road).is_none());
    }

    #[test]
    fn no_road_from_enemy_road() {
        let b = setup();

        // Only Red and Orange can build this road
        let road = b.edge_id(Edge(1, -1, NW));
        assert!(b.available_roads(Orange).contains(&road));
        assert!(b.available_roads(Red).contains(&road));
        assert!(!b.available_roads(Blue).contains(&road));
        assert!(!b.available_roads(White).contains(&road));
    }

    #[test]
    fn no_out_of_bounds_roads() {
        let mut b = setup();

        road(&mut b, Red, Edge(1, -2, W));
        road(&mut b, Red, Edge(0, -2, NE));

        // Assert all placeable roads are on edges with ids in 0..72
        let placeable = b.available_roads(Red);
        assert!(placeable
            .iter()
            .all(|edge_id| (0..N_EDGES).contains(edge_id)));
    }
}
