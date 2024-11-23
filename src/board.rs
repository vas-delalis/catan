use enum_map::{Enum, EnumMap};
use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    fmt::Debug,
    rc::Rc,
};

use crate::{bundle::Bundle, EdgeId, Player, Resource, VertexId, N_HEXES, N_ROLLS, RESOURCES};

type VertexMap = u64;

#[derive(Debug, Clone, Copy)]
struct Hex {
    q: i8,
    r: i8,
}

impl Hex {
    fn vertices(&self) -> Vec<Vertex> {
        let q = self.q;
        let r = self.r;
        [
            (q, r, N),
            (q, r, S),
            (q, r + 1, N),
            (q, r - 1, S),
            (q - 1, r + 1, N),
            (q + 1, r - 1, S),
        ]
        .into_iter()
        .map(|(q, r, dir)| Vertex { q, r, dir })
        .collect()
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum VertexDir {
    N,
    S,
}

use VertexDir::*;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Vertex {
    q: i8,
    r: i8,
    dir: VertexDir,
}

impl Vertex {
    fn coords(&self) -> (f64, f64) {
        let (dq, dr) = {
            match self.dir {
                VertexDir::N => (1.0 / 3.0, -2.0 / 3.0),
                VertexDir::S => (-1.0 / 3.0, 2.0 / 3.0),
            }
        };
        let q = (self.q as f64) + dq;
        let r = (self.r as f64) + dr;
        (q, r)
    }

    fn ordering_value(&self) -> f64 {
        let (q, r) = self.coords();
        3.0 * q + 21.0 * r.ceil()
    }
}

impl Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {:?})", self.q, self.r, self.dir)
    }
}

#[derive(Debug)]
struct SharedBoardData {
    adjacency: Vec<VertexMap>,
    roll_resources: Vec<Vec<VertexMap>>,
    hex_board: HexBoard,
}

#[derive(Debug)]
pub struct Board {
    static_data: Rc<SharedBoardData>,
    player_buildings: EnumMap<Player, VertexMap>,
    cities: VertexMap,
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Board {
            player_buildings: self.player_buildings.clone(),
            static_data: Rc::clone(&self.static_data),
            cities: self.cities.clone(),
        }
    }
}

impl Board {
    pub fn new(resources: Vec<Resource>, rolls: Vec<u8>) -> Self {
        let hex_board = HexBoard::new();
        let mut adj_bitmaps: Vec<VertexMap> = vec![0u64; N_HEXES];
        let mut roll_resource_bitmaps: Vec<Vec<VertexMap>> =
            vec![vec![0u64; Resource::LENGTH]; N_ROLLS];

        for (i, hex) in hex_board.hexes.iter().enumerate() {
            let adj = hex_board.get_bitmap(hex.vertices());
            adj_bitmaps[i] = adj;
            roll_resource_bitmaps[(rolls[i] - 2) as usize][resources[i] as usize] |= adj;
        }

        let static_data = SharedBoardData {
            adjacency: adj_bitmaps,
            roll_resources: roll_resource_bitmaps,
            hex_board,
        };
        Board {
            static_data: Rc::new(static_data),
            player_buildings: EnumMap::default(),
            cities: VertexMap::default(),
        }
    }

    // Helpers

    pub fn vertex_id(&self, vertex: Vertex) -> VertexId {
        self.static_data.hex_board.vertex_ids[&vertex]
    }

    // Gameplay

    pub fn add_settlement(&mut self, player: Player, vertex_id: VertexId) {
        self.player_buildings[player] |= 1 << vertex_id;
    }

    pub fn upgrade_settlement(&mut self, vertex_id: VertexId) {
        self.cities |= 1 << vertex_id;
    }

    pub fn add_road(&mut self, player: Player, edge_id: EdgeId) {
        todo!()
    }

    pub fn produce_resources(&self, roll: u8) -> EnumMap<Player, Bundle> {
        // TODO: Vectorize?
        let mut bundles: EnumMap<Player, Bundle> = EnumMap::default();

        for res in RESOURCES {
            let resource_map: VertexMap =
                self.static_data.roll_resources[(roll - 2) as usize][res as usize];
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
struct HexBoard {
    hexes: Vec<Hex>,
    vertex_ids: HashMap<Vertex, usize>,
}

impl HexBoard {
    fn new() -> Self {
        let n: i8 = 2; // Grid size
        let mut hexes = Vec::with_capacity(19);
        let mut vertices = HashSet::with_capacity(54);

        for r in -n..=n {
            let q1 = max(-n, -r - n);
            let q2 = min(n, -r + n);
            for q in q1..=q2 {
                let hex = Hex { q, r };
                hexes.push(hex);
                for v in hex.vertices() {
                    vertices.insert(v);
                }
            }
        }

        let mut vertices: Vec<Vertex> = vertices.into_iter().collect();
        vertices.sort_by(|a, b| a.ordering_value().total_cmp(&b.ordering_value()));
        let vertex_ids: HashMap<Vertex, usize> = vertices
            .into_iter()
            .enumerate()
            .map(|(id, v)| (v, id))
            .collect();

        HexBoard { hexes, vertex_ids }
    }

    /// Converts a list of vertices to the corresponding bitmap.
    pub fn get_bitmap(&self, vertices: Vec<Vertex>) -> VertexMap {
        vertices
            .into_iter()
            .fold(0u64, |bitmap, v| bitmap | (1 << self.vertex_ids[&v]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Player::*;
    use Resource::*;

    fn setup() -> Board {
        let resources = vec![
            Ore, Wool, Lumber, Grain, Brick, Wool, Brick, Grain, Lumber, Ore, Lumber, Ore, Lumber,
            Ore, Grain, Wool, Brick, Grain, Wool,
        ];
        let rolls: Vec<u8> = vec![10, 2, 9, 12, 6, 4, 10, 9, 11, 7, 3, 8, 8, 3, 4, 5, 5, 6, 11];

        Board::new(resources, rolls)
    }

    #[test]
    fn resource_production1() {
        let mut board = setup();

        board.add_settlement(Red, 10);
        board.add_settlement(White, 19);
        board.add_settlement(
            Orange,
            board.vertex_id(Vertex {
                q: -1,
                r: 2,
                dir: N,
            }),
        );

        let production = board.produce_resources(6);

        assert_eq!(production[Blue], Bundle::default());
        assert_eq!(production[Orange], Bundle::from_array([0, 1, 0, 0, 0]));
        assert_eq!(production[Red], Bundle::from_array([1, 0, 0, 0, 0]));
        assert_eq!(production[White], Bundle::from_array([1, 0, 0, 0, 0]));
    }

    #[test]
    fn resource_production2() {
        let mut board = setup();

        [(Red, 0, -2, N), (Red, 0, -1, N), (Red, -1, -1, N)]
            .into_iter()
            .for_each(|(player, q, r, dir)| {
                let id = board.vertex_id(Vertex { q, r, dir });
                board.add_settlement(player, id);
                board.upgrade_settlement(id);
            });

        let production = board.produce_resources(10);

        assert_eq!(production[Blue], Bundle::default());
        assert_eq!(production[Orange], Bundle::default());
        assert_eq!(production[Red], Bundle::from_array([0, 0, 0, 6, 0]));
        assert_eq!(production[White], Bundle::default());
    }
}
