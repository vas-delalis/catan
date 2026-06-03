use std::io;

use catan::{State as CatanState, *};
use common::GameState;

fn main() {
    let mut state = CatanState::default();

    fn sett(s: &mut State, p: Player, v: Vertex) {
        s.board.add_settlement(p, s.board.vertex_id(v));
    }

    fn road(s: &mut State, p: Player, e: Edge) {
        s.board.add_road(p, s.board.edge_id(e));
    }

    let mut s = |p: Player, v: Vertex| sett(&mut state, p, v);

    s(Blue, Vertex(-2, 2, N));
    s(Blue, Vertex(0, 2, N));
    s(Orange, Vertex(2, -2, S));
    s(Orange, Vertex(-1, 2, N));
    s(Red, Vertex(0, -1, N));
    s(Red, Vertex(-2, 1, N));
    s(White, Vertex(-1, 0, N));
    s(White, Vertex(1, 1, N));

    let mut r = |p: Player, e: Edge| road(&mut state, p, e);
    r(Blue, Edge(-2, 2, NE));
    r(Blue, Edge(1, 1, W));
    r(Orange, Edge(-1, 2, NE));
    r(Orange, Edge(1, -1, NE));
    r(Red, Edge(-2, 1, NE));
    r(Red, Edge(0, -1, NE));
    r(White, Edge(-1, 0, NW));
    r(White, Edge(2, 0, W));

    while !state.is_terminal() {
        let actions = state.get_actions(state.current_player()).0;
        let labels: Vec<String> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                Action::MoveRobber(hex) => format!("{i} {:?}", state.board.hex(*hex)),
                _ => format!("{i} {:?}", a),
            })
            .collect();

        println!("{:?} {:?}", state.current_player(), state.phase);
        println!("{}", labels.join("\n"));

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        let input: usize = input.trim().parse().expect("Please type a number!");

        state.apply_action(actions[input]);
    }
}
