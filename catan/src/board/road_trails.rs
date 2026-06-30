//! # Road trail table
//!
//! This module contains `RoadTrailTable`, which can find the longest trail in a given road network.
//!
//! The table can be loaded or generated using `RoadTableData`.

use std::{
    cmp::max,
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    fs::File,
    hash::BuildHasher,
    io::{BufWriter, Read},
    path::PathBuf,
};

use crate::{Bitboard, EdgeId, board::shared_data::ADJACENCY};
use ahash::RandomState;
use num::Zero;
use rkyv::{Archive, Deserialize, Serialize, api::high::to_bytes_in, ser::writer::IoWriter};
use rkyv_util::owned::OwnedArchive;

const LOOKUP_TABLE_NAME: &str = "roads.bin";
const AHASH_SEEDS: (u64, u64, u64, u64) = (7129002836, 567957864, 9421963134, 7836118570);
const HASH_MAP_CAPACITY: usize = 38_000_000;

/// Allows loading or generating-and-saving the `RoadTrailTable`.
///
/// This is actually the original (unarchived) version of `RoadTrailTable`.
#[derive(Archive, Serialize, Deserialize)]
pub struct RoadTrailTableLoader {
    map: RoadTrailHashMap,
}

type RoadTrailHashMap = HashMap<u128, u8, SeededRandomState>;
pub type RoadTrailTable = OwnedArchive<RoadTrailTableLoader, Vec<u8>>;

/// Makes `ahash` deterministic.
struct SeededRandomState(RandomState);

impl SeededRandomState {
    pub fn new() -> Self {
        let (s0, s1, s2, s3) = AHASH_SEEDS;
        SeededRandomState(RandomState::with_seeds(s0, s1, s2, s3))
    }
}

impl BuildHasher for SeededRandomState {
    type Hasher = ahash::AHasher;

    fn build_hasher(&self) -> Self::Hasher {
        self.0.build_hasher()
    }
}

impl RoadTrailTableLoader {
    fn get_path() -> PathBuf {
        let mut path = common::PROJECT_DIRS.data_dir().to_path_buf();
        path.push(LOOKUP_TABLE_NAME);
        path
    }

    pub fn load() -> OwnedArchive<RoadTrailTableLoader, Vec<u8>> {
        let path = Self::get_path();
        let mut file =
            File::open(&path).unwrap_or_else(|_| panic!("{} should exist", path.display()));
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        OwnedArchive::<RoadTrailTableLoader, _>::new::<rkyv::rancor::Error>(buffer).unwrap()
    }

    pub fn generate_and_save() -> Result<(), Box<dyn Error>> {
        let path = Self::get_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut table: RoadTrailHashMap =
            HashMap::with_capacity_and_hasher(HASH_MAP_CAPACITY, SeededRandomState::new());

        let graphs = RoadGraphIterator::new();
        for graph in graphs {
            table.insert(graph.value, slow_longest_trail(graph));
        }

        table.shrink_to_fit();

        let file = File::create(path)?;
        let writer = IoWriter::new(BufWriter::new(file));

        to_bytes_in::<_, rkyv::rancor::Error>(&table, writer)?;

        Ok(())
    }
}

// This is the impl for RoadTrailTable
impl ArchivedRoadTrailTableLoader {
    pub fn lookup(&self, graph: u128) -> u8 {
        *self
            .map
            .get(&graph.into())
            .unwrap_or_else(|| panic!("lookup table should contain road graph: {:#x}", graph))
    }

    /// Returns the longest trail for a road graph by querying the lookup table.
    pub fn longest_trail(&self, roads: Bitboard<u128>, enemy_buildings: Bitboard<u64>) -> u8 {
        // 1. Starting from an arbitrary unvisited vertex, do a BFS.
        // 2. Look up the longest trail for the resulting connected component.
        // 3. If there are still unvisited vertices, return to 1.
        // 4. Return the longest longest trail.
        let mut longest = 0u8;
        let mut visited: Bitboard<u128> = Bitboard::zeros();
        let mut queue: VecDeque<EdgeId> = VecDeque::new();

        while visited != roads {
            let mut component = Bitboard::<u128>::zeros();
            let root: EdgeId = (roads & !visited).value.trailing_zeros() as EdgeId;
            queue.push_back(root);

            while let Some(edge) = queue.pop_front() {
                visited.add(edge);
                component.add(edge);
                for neighbor in ADJACENCY.edge_to_edges[edge] & roads & !visited {
                    // Skip neighboring road if connecting vertex has enemy building
                    let v = ADJACENCY.edge_to_verts[edge] & ADJACENCY.edge_to_verts[neighbor];
                    if (v & enemy_buildings).value.is_zero() {
                        queue.push_back(neighbor);
                    }
                }
            }

            longest = max(self.lookup(component.value), longest)
        }
        longest
    }
}

/// Returns the longest trail for a connected road graph.
///
/// This algorithm is too slow to use at runtime.
/// Instead, we use it to build a lookup table by precomputing the longest trail of every graph.
pub fn slow_longest_trail(roads: Bitboard<u128>) -> u8 {
    let road_count = roads.count_ones() as usize;

    let verts = roads.fold(Bitboard::zeros(), |bb, eid| {
        ADJACENCY.edge_to_verts[eid] | bb
    });
    let n = verts.count_ones() as usize;

    let mut vert_indices: Vec<usize> = vec![54; 54];
    for (idx, id) in verts.enumerate() {
        vert_indices[id] = idx;
    }
    let mut table: Vec<Vec<Vec<Bitboard<u128>>>> = vec![vec![vec![]; n]; road_count + 1];

    for vi in 0..n {
        table[0][vi].push(Bitboard::zeros());
    }
    let mut longest = 0;
    for length in 1..=road_count {
        let (prev, curr) = table.split_at_mut(length);
        for v in verts {
            let vi = vert_indices[v];
            let connections = ADJACENCY.vert_to_edges[v] & roads;
            let mask = !Bitboard::single(v);
            for c in connections {
                // for &(c, w) in &vert_neighbors[vi] {
                let w = (ADJACENCY.edge_to_verts[c] & mask).next().unwrap();
                let wi = vert_indices[w];
                // for path in &old_paths[wi] {
                for path in &prev[length - 1][wi] {
                    if path.contains(c) {
                        continue;
                    }
                    let mut new_path = *path;
                    new_path.add(c);
                    curr[0][vi].push(new_path);
                    // dbg!(curr[0][vi].len());
                    longest = length;
                }
            }
        }
    }
    longest as u8
}

/// A simplified version of [Board] used for walking the road-graph tree.
#[derive(Debug, Clone, Copy)]
struct RoadBoard {
    pub road_slots: Bitboard<u128>,
    pub roads: Bitboard<u128>,
}

impl RoadBoard {
    fn add_road(&mut self, edge_id: EdgeId, neighbors: Bitboard<u128>) {
        self.roads.add(edge_id);
        if self.road_slots.count_ones() == 72 {
            self.road_slots = Bitboard::zeros();
        }
        self.road_slots |= neighbors;
        self.road_slots.remove(edge_id);
    }
}

/// Iterates through every possible road graph.
struct RoadGraphIterator {
    unique: HashSet<u128>,
    queue: VecDeque<RoadBoard>,
}

impl RoadGraphIterator {
    pub fn new() -> RoadGraphIterator {
        let unique: HashSet<u128> = HashSet::new();
        let mut queue: VecDeque<RoadBoard> = VecDeque::with_capacity(1 << 20);

        // First road piece could be anywhere.
        let root = RoadBoard {
            road_slots: Bitboard {
                value: !0 >> (128 - 72),
            },
            roads: Bitboard { value: 0 },
        };

        queue.push_front(root);

        RoadGraphIterator { unique, queue }
    }
}

impl Iterator for RoadGraphIterator {
    type Item = Bitboard<u128>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.queue.pop_back() {
            if b.roads.count_ones() < 15 {
                for next_edge in b.road_slots {
                    let mut next_b = b;
                    next_b.add_road(next_edge, ADJACENCY.edge_to_edges[next_edge]);
                    if !self.unique.insert(next_b.roads.value) {
                        continue;
                    }
                    self.queue.push_front(next_b);
                }
            }
            return Some(b.roads);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{board::shared_data::ROAD_TRAILS, *};

    #[test]
    fn load_table() {
        ROAD_TRAILS.lookup(0);
    }

    #[test]
    fn lookup() {
        let data = vec![
            ("7c1e601c00200000", 10),
            ("3609b801880090000", 11),
            ("1000000", 1),
        ];
        for (hex, length) in data {
            let roads: Bitboard<u128> = Bitboard::from_hex(hex);
            assert_eq!(ROAD_TRAILS.lookup(roads.value), length);
        }
    }

    #[test]
    fn enemy_building_blocks_trail() {
        let mut b = Board::default();
        let roads: Bitboard<u128> = Bitboard::from_hex("10019801e00400000");
        for road in roads {
            b.add_road(Blue, road);
        }

        let setts: Bitboard<u64> = Bitboard::from_hex("1200400");
        for sett in setts {
            b.add_settlement(Orange, sett);
        }

        let buildings = b.buildings(Orange);
        assert_eq!(ROAD_TRAILS.longest_trail(b.roads(Blue), buildings), 3);
    }
}
