use std::collections::{HashMap, HashSet, VecDeque};

use crate::{Bitboard, Board, Edge, EdgeDir, EdgeId, Player};

pub fn longest(b: &Board, player: Player, _: bool) -> usize {
    let roads = b.roads(player);
    let road_count = roads.count_ones() as usize;

    let verts = roads.fold(Bitboard::zeros(), |bb, eid| {
        b.shared_data.edge_to_verts[eid] | bb
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
            let connections = b.shared_data.vert_to_edges[v] & roads;
            let mask = !Bitboard::single(v);
            for c in connections {
                // for &(c, w) in &vert_neighbors[vi] {
                let w = (b.shared_data.edge_to_verts[c] & mask).next().unwrap();
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
        // swap(&mut old_paths, &mut new_paths);
    }
    return longest;

    // dbg!(&table);
    // dbg!(longest);
    // for v_paths in &table[longest] {
    //     if v_paths.is_empty() {
    //         continue;
    //     }
    //     let path: Vec<Vertex> = v_paths[0].iter().map(|&v| b.vertex(vert_ids[v])).collect();
    //     dbg!(path);
    // }
}

pub fn polyominos() {
    let b = Board::default();
    let p = Player::Blue;
    let root = (b.edge_id(Edge(0, 0, EdgeDir::NE)), b);

    let mut count = 0;
    let mut unique: HashSet<u128> = HashSet::new();
    let mut queue: VecDeque<(EdgeId, Board)> = VecDeque::with_capacity(64);
    // let mut sequences: Vec<Vec<u32>> = vec![];
    let examples: Vec<(Vec<u32>, u128, usize)> = vec![];
    let mut sequences: HashMap<String, (u128, usize)> = HashMap::new();

    let mut size = 0;
    queue.push_front(root);
    while let Some((edge_id, mut board)) = queue.pop_back() {
        board.add_road(p, edge_id);
        if !unique.insert(board.roads(p).value) {
            continue;
        }
        let roads = board.roads(p);
        if roads.count_ones() == 7 {
            count += 1;

            let mut seq = Vec::with_capacity(15);
            for road in roads {
                seq.push((board.shared_data.edge_to_edges[road] & roads).count_ones() - 1);
            }
            seq.sort();
            let seq: String = seq.iter().map(|deg| deg.to_string()).collect();

            let longest_trail = longest(&board, p, false);
            if let Some((example, known_length)) = sequences.get(&seq) {
                if longest_trail != *known_length {
                    dbg!(seq);
                    dbg!(example);
                    dbg!(known_length);
                    dbg!(roads.value);
                    dbg!(longest_trail);
                    dbg!(longest(&board, p, true));
                    panic!();
                }
            } else {
                sequences.insert(seq, (roads.value, longest_trail));
            }
            continue;
        }
        for next in board.available_roads(p) {
            queue.push_front((next, board.clone()));
        }
        // dbg!(queue.len());
        if queue.len() > size {
            size = queue.len();
        }
    }
    dbg!(size);
    dbg!(count);
    dbg!(&sequences);
    dbg!(&examples);
    dbg!(sequences.len());
}

#[derive(Debug, Clone, Copy)]
struct RoadBoard {
    pub road_slots: Bitboard<u128>,
    pub roads: Bitboard<u128>,
}

impl RoadBoard {
    fn add_road(&mut self, edge_id: EdgeId, neighbors: Bitboard<u128>) {
        self.roads.add(edge_id);
        self.road_slots |= neighbors;
        self.road_slots.remove(edge_id);
    }
}

pub fn longest_trail(roads: Bitboard<u128>, reference_board: &Board) -> usize {
    let road_count = roads.count_ones() as usize;

    let verts = roads.fold(Bitboard::zeros(), |bb, eid| {
        reference_board.shared_data.edge_to_verts[eid] | bb
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
            let connections = reference_board.shared_data.vert_to_edges[v] & roads;
            let mask = !Bitboard::single(v);
            for c in connections {
                // for &(c, w) in &vert_neighbors[vi] {
                let w = (reference_board.shared_data.edge_to_verts[c] & mask)
                    .next()
                    .unwrap();
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
    return longest;
}

pub struct RoadGraphIterator {
    unique: HashSet<u128>,
    queue: VecDeque<RoadBoard>,
    pub reference_board: Board,
}

impl RoadGraphIterator {
    pub fn new() -> RoadGraphIterator {
        let unique: HashSet<u128> = HashSet::new();
        let mut queue: VecDeque<RoadBoard> = VecDeque::with_capacity(1 << 20);
        let reference_board = Board::default();

        let mut root = RoadBoard {
            road_slots: Bitboard { value: 0 },
            roads: Bitboard { value: 0 },
        };

        let root_edge = reference_board.edge_id(Edge(0, 0, EdgeDir::NE));
        root.add_road(
            root_edge,
            reference_board.shared_data.edge_to_edges[root_edge],
        );

        queue.push_front(root);

        RoadGraphIterator {
            unique,
            queue,
            reference_board,
        }
    }
}

impl Iterator for RoadGraphIterator {
    type Item = Bitboard<u128>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.queue.pop_back() {
            if b.roads.count_ones() < 15 {
                for next_edge in b.road_slots {
                    let mut next_b = b.clone();
                    next_b.add_road(
                        next_edge,
                        self.reference_board.shared_data.edge_to_edges[next_edge],
                    );
                    // let next_graph = b.roads.value | (1 << next_edge);
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

pub fn paths(max_edges: u32) {
    let mut count = 0;
    let mut unique: HashSet<u128> = HashSet::new();
    let mut queue: VecDeque<(EdgeId, RoadBoard)> = VecDeque::with_capacity(64);

    let reference_board = Board::default();
    let b = RoadBoard {
        road_slots: Bitboard { value: 0 },
        roads: Bitboard { value: 0 },
    };

    let root = (reference_board.edge_id(Edge(0, 0, EdgeDir::NE)), b);

    // let mut max_queue_size = 0;
    queue.push_front(root);
    while let Some((edge_id, mut b)) = queue.pop_back() {
        b.add_road(edge_id, reference_board.shared_data.edge_to_edges[edge_id]);
        if !unique.insert(b.roads.value) {
            continue;
        }
        if b.roads.count_ones() == max_edges {
            count += 1;
            // unique.insert(board.roads(p).value);
            continue;
        }
        for next in b.road_slots {
            queue.push_front((next, b.clone()));
        }

        // if queue.len() > max_queue_size {
        //     max_queue_size = queue.len();
        // }
    }
    // dbg!(max_queue_size);
    dbg!(count, unique.len());
}
