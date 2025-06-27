use std::collections::{HashSet, VecDeque};

use catan::*;

fn main() {
    let p = Player::Blue;

    // for i in 0..72 {
    //     let b = Board::default();
    //     let e = b.edge(i);
    //     println!("{{q:{},r:{},dir:\"{:?}\"}},", e.0, e.1, e.2);
    // }

    let mut count = 0;
    let mut unique: HashSet<u128> = HashSet::new();
    let mut queue: VecDeque<(EdgeId, Board)> = VecDeque::with_capacity(64);

    let b = Board::default();
    let root = (b.edge_id(Edge(-3, 3, EdgeDir::NE)), b);

    queue.push_front(root);
    while let Some((edge_id, mut board)) = queue.pop_back() {
        board.add_road(p, edge_id);
        if board.roads(p).count_ones() == 12 {
            count += 1;
            unique.insert(board.roads(p).value);
            continue;
        }
        for next in board.available_roads(p) {
            queue.push_front((next, board.clone()));
        }
    }

    dbg!(count, unique.len());
}
