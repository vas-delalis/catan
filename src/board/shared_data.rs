// TODO: Which adjacency maps do we need?
// (A adjacent to each B)
// A / B |    Hex    |     Vert     |     Edge
// ----------------------------------------------
// Hexes |     ?     |       ?      |      ?
// Verts |  Robber   | Sett. plcmnt | Sett. plcmnt
// Edges |     ?     | Road plcmnt  | Road plcmnt

use enum_map::{enum_map, Enum, EnumMap};

use crate::{
    board::hex_board::*, board::road_trails::RoadTrailTable, common::*, Bitboard, Edge, HexBoard,
    E, V,
};

pub struct Adjacency {
    pub hex_to_verts: Vec<Bitboard<V>>,
    pub vert_to_verts: Vec<Bitboard<V>>,
    pub edge_to_verts: Vec<Bitboard<V>>,

    pub vert_to_edges: Vec<Bitboard<E>>,
    pub edge_to_edges: Vec<Bitboard<E>>,
}

impl Adjacency {
    pub fn new() -> Self {
        let hb = HexBoard::new();

        let mut hex_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_HEXES];
        // Populate hex-to-vert maps
        for (i, hex) in hb.hexes.iter().enumerate() {
            let adj = hb.vert_bitboard(&hex.vertices());
            hex_to_verts[i] = adj;
        }

        let mut vert_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_VERTICES];
        let mut vert_to_edges: Vec<Bitboard<E>> = vec![Bitboard::zeros(); N_VERTICES];

        // Populate vert-to-* maps
        for (vert, &i) in hb.vertex_ids.iter() {
            vert_to_edges[i] = hb.edge_bitboard(&vert.edges());
            vert_to_verts[i] = hb.vert_bitboard(&vert.neighbors());
            vert_to_verts[i].add(i); // Include self
        }

        let mut edge_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_EDGES];
        let mut edge_to_edges: Vec<Bitboard<E>> = vec![Bitboard::zeros(); N_EDGES];

        // Populate edge-to-* maps
        for (edge, &i) in hb.edge_ids.iter() {
            edge_to_verts[i] = hb.vert_bitboard(&edge.vertices());
            edge_to_edges[i] = hb.edge_bitboard(&edge.neighbors());
            edge_to_edges[i].add(i); // Include self
        }

        Adjacency {
            hex_to_verts,
            vert_to_verts,
            edge_to_verts,
            vert_to_edges,
            edge_to_edges,
        }
    }
}

/// Data -- mainly bitmaps -- that doesn't depend on the game state.
pub struct SharedBoardData {
    pub hex_to_verts: Vec<Bitboard<V>>,
    pub vert_to_verts: Vec<Bitboard<V>>,
    pub edge_to_verts: Vec<Bitboard<V>>,

    pub vert_to_edges: Vec<Bitboard<E>>,
    pub edge_to_edges: Vec<Bitboard<E>>,

    pub resources: Vec<Option<Resource>>,
    pub rolls: Vec<Option<u8>>,

    pub roll_resources: Vec<Vec<Bitboard<V>>>, // Vertices that receive a given resource on a given roll
    pub generic_harbors: Bitboard<V>,
    pub resource_harbors: EnumMap<Resource, Bitboard<V>>,
    pub hex_board: HexBoard,

    pub road_trails: RoadTrailTable,
} // TODO: use arrays? (fixed length is good)

impl SharedBoardData {
    pub fn new(resources: Vec<Option<Resource>>, rolls: Vec<Option<u8>>) -> Self {
        // TODO: return Result instead
        let hb = HexBoard::new();

        let mut hex_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_HEXES];
        let mut roll_resources: Vec<Vec<Bitboard<V>>> =
            vec![vec![Bitboard::zeros(); Resource::LENGTH]; N_ROLLS];

        // Populate hex-to-vert maps and roll-resource maps
        for (i, hex) in hb.hexes.iter().enumerate() {
            let adj = hb.vert_bitboard(&hex.vertices());
            hex_to_verts[i] = adj;

            if let Some(resource) = resources[i] {
                let roll = rolls[i].expect("hexes with a resource should also have a roll");
                roll_resources[(roll - 2) as usize][resource as usize] |= adj;
            }
        }

        let mut vert_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_VERTICES];
        let mut vert_to_edges: Vec<Bitboard<E>> = vec![Bitboard::zeros(); N_VERTICES];

        // Populate vert-to-* maps
        for (vert, &i) in hb.vertex_ids.iter() {
            vert_to_edges[i] = hb.edge_bitboard(&vert.edges());
            vert_to_verts[i] = hb.vert_bitboard(&vert.neighbors());
            vert_to_verts[i].add(i); // Include self
        }

        let mut edge_to_verts: Vec<Bitboard<V>> = vec![Bitboard::zeros(); N_EDGES];
        let mut edge_to_edges: Vec<Bitboard<E>> = vec![Bitboard::zeros(); N_EDGES];

        // Populate edge-to-* maps
        for (edge, &i) in hb.edge_ids.iter() {
            edge_to_verts[i] = hb.vert_bitboard(&edge.vertices());
            edge_to_edges[i] = hb.edge_bitboard(&edge.neighbors());
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
            let id = hb
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
            let id = hb
                .edge_ids
                .get(&edge)
                .expect("harbor edges should be valid");
            let ends = edge_to_verts[*id];
            generic_harbors |= ends;
        }

        let road_trails = RoadTrailTable::load();

        SharedBoardData {
            hex_to_verts,
            vert_to_verts,
            edge_to_verts,

            vert_to_edges,
            edge_to_edges,

            resources,
            rolls,

            roll_resources,
            generic_harbors,
            resource_harbors,
            hex_board: hb,
            road_trails,
        }
    }
}
