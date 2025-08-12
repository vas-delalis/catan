use std::{
    cmp::max,
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    fs::File,
    hash::BuildHasher,
    io::{BufReader, BufWriter},
};

use crate::{board::shared_data::Adjacency, Bitboard, EdgeId};
use ahash::RandomState;
use bincode::config::Configuration;

const LOOKUP_TABLE_PATH: &str = "roads.bin";
const BINCODE_CONFIG: Configuration = bincode::config::standard();
const AHASH_SEEDS: (u64, u64, u64, u64) = (7129002836, 567957864, 9421963134, 7836118570);
const HASH_MAP_CAPACITY: usize = 38_000_000;

pub struct RoadTrailTable {
    map: RoadTrailHashMap,
    adjacency: Adjacency,
}

type RoadTrailHashMap = HashMap<u128, u8, SeededRandomState>;

/// Overrides `RandomState`'s `Default` trait in order to seed it.
///
/// This is required because `bincode` uses the trait to initialize the deserialized `HashMap`.
struct SeededRandomState(RandomState);

impl Default for SeededRandomState {
    fn default() -> Self {
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

impl RoadTrailTable {
    pub fn load() -> Self {
        let file =
            File::open(LOOKUP_TABLE_PATH).expect(&format!("{} should exist", LOOKUP_TABLE_PATH));
        let mut reader = BufReader::new(file);

        let map: RoadTrailHashMap =
            bincode::decode_from_reader(&mut reader, BINCODE_CONFIG).unwrap();

        RoadTrailTable {
            map,
            adjacency: Adjacency::new(),
        }
    }

    pub fn generate_and_save() -> Result<(), Box<dyn Error>> {
        let mut table: RoadTrailHashMap =
            HashMap::with_capacity_and_hasher(HASH_MAP_CAPACITY, SeededRandomState::default());

        let adjacency = Adjacency::new();
        let graphs = RoadGraphIterator::new(&adjacency);

        for graph in graphs {
            table.insert(graph.value, slow_longest_trail(graph, &adjacency));
        }

        table.shrink_to_fit();

        let file = File::create(LOOKUP_TABLE_PATH)?;
        let mut writer = BufWriter::new(file);
        bincode::encode_into_std_write(table, &mut writer, BINCODE_CONFIG)?;

        Ok(())
    }

    fn lookup(&self, graph: u128) -> u8 {
        *self
            .map
            .get(&graph)
            .expect("lookup table should contain road graph")
    }

    /// Returns the longest trail for a road graph by querying the lookup table.
    pub fn longest_trail(&self, roads: Bitboard<u128>) -> u8 {
        // Find connected components (up to two)
        // TODO: Enemy settlements block the trail, creating more components
        let mut visited: Bitboard<u128> = Bitboard::zeros();
        let mut queue: VecDeque<EdgeId> = VecDeque::new();
        let root = roads.value.trailing_zeros() as EdgeId;
        queue.push_back(root);

        while let Some(edge) = queue.pop_front() {
            visited.add(edge);
            for neighbor in self.adjacency.edge_to_edges[edge] & roads & !visited {
                queue.push_back(neighbor);
            }
        }

        // Compute longest trail of each component; Return longest of the two
        let component1_length = self.lookup(visited.value);
        if visited != roads {
            let component2_length = self.lookup((!visited & roads).value);
            return max(component1_length, component2_length);
        }
        component1_length
    }
}

/// Returns the longest trail for a connected road graph.
///
/// This algorithm is too slow to use at runtime.
/// Instead, we use it to build a lookup table by precomputing the longest trail of every graph.
pub fn slow_longest_trail(roads: Bitboard<u128>, adjacency: &Adjacency) -> u8 {
    let road_count = roads.count_ones() as usize;

    let verts = roads.fold(Bitboard::zeros(), |bb, eid| {
        adjacency.edge_to_verts[eid] | bb
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
            let connections = adjacency.vert_to_edges[v] & roads;
            let mask = !Bitboard::single(v);
            for c in connections {
                // for &(c, w) in &vert_neighbors[vi] {
                let w = (adjacency.edge_to_verts[c] & mask).next().unwrap();
                let wi = vert_indices[w];
                // for path in &old_paths[wi] {
                for path in &prev[length - 1][wi] {
                    if path.contains(c) {
                        continue;
                    }
                    let mut new_path = path.clone();
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
struct RoadGraphIterator<'a> {
    unique: HashSet<u128>,
    queue: VecDeque<RoadBoard>,
    adjacency: &'a Adjacency,
}

impl<'a> RoadGraphIterator<'a> {
    pub fn new(adjacency: &'_ Adjacency) -> RoadGraphIterator<'_> {
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

        RoadGraphIterator {
            unique,
            queue,
            adjacency,
        }
    }
}

impl Iterator for RoadGraphIterator<'_> {
    type Item = Bitboard<u128>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.queue.pop_back() {
            if b.roads.count_ones() < 15 {
                for next_edge in b.road_slots {
                    let mut next_b = b.clone();
                    next_b.add_road(next_edge, self.adjacency.edge_to_edges[next_edge]);
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
    use super::*;

    fn bb(hex: &str) -> Bitboard<u128> {
        Bitboard::from(u128::from_str_radix(hex, 16).unwrap())
    }

    #[test]
    fn load_table() {
        let table = RoadTrailTable::load();

        assert_eq!(table.longest_trail(bb("7c1e601c00200000")), 10);
        assert_eq!(table.longest_trail(bb("3609b801880090000")), 11);
        assert_eq!(table.longest_trail(bb("1000000")), 1);
    }
}
