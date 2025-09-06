use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use crate::{common::*, Bitboard};

pub use EdgeDir::*;
pub use VertexDir::*;

pub type V = u64;
pub type E = u128;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct Hex(pub i8, pub i8);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Vertex(pub i8, pub i8, pub VertexDir);

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge(pub i8, pub i8, pub EdgeDir);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexDir {
    N,
    S,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
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
    pub fn vertices(&self) -> Vec<Vertex> {
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

    pub fn edges(&self) -> Vec<Edge> {
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
    pub fn coords(&self) -> (f64, f64) {
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

    pub fn ordering_value(&self) -> f64 {
        let (q, r) = self.coords();
        3.0 * q + 21.0 * r.ceil()
    }

    /// Returns neighboring vertices.
    pub fn neighbors(&self) -> [Vertex; 3] {
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
    pub fn edges(&self) -> [Edge; 3] {
        let &Vertex(q, r, dir) = self;
        match dir {
            N => [Edge(q, r, NE), Edge(q, r, NW), Edge(q + 1, r - 1, W)],
            S => [
                Edge(q, r + 1, W),
                Edge(q, r + 1, NW),
                Edge(q - 1, r + 1, NE),
            ],
        }
    }
}

impl Edge {
    pub fn coords(&self) -> (f64, f64) {
        let &Edge(q, r, dir) = self;
        let (dq, dr) = {
            match dir {
                NE => (0.5, -0.5),
                NW => (0.0, -0.5),
                W => (-0.5, 0.0),
            }
        };
        let q = (q as f64) + dq;
        let r = (r as f64) + dr;
        (q, r)
    }

    /// Returns neighboring edges.
    pub fn neighbors(&self) -> [Edge; 4] {
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
    pub fn vertices(&self) -> [Vertex; 2] {
        let &Edge(q, r, dir) = self;
        match dir {
            NE => [Vertex(q, r, N), Vertex(q + 1, r - 1, S)],
            NW => [Vertex(q, r, N), Vertex(q, r - 1, S)],
            W => [Vertex(q, r - 1, S), Vertex(q - 1, r + 1, N)],
        }
    }
}

/// A high-level representation of the Catan board. Used to generate the much more efficient [crate::Board].
#[derive(Debug)]
pub struct HexBoard {
    pub hexes: Vec<Hex>,
    pub hex_ids: HashMap<Hex, HexId>,
    pub vertices: Vec<Vertex>,
    pub vertex_ids: HashMap<Vertex, VertexId>,
    pub edges: Vec<Edge>,
    pub edge_ids: HashMap<Edge, EdgeId>,
}

impl HexBoard {
    pub fn new() -> Self {
        HexBoard::with_radius(2)
    }

    fn with_radius(radius: i8) -> Self {
        let mut hexes = Vec::with_capacity(N_HEXES);
        let mut vertices = HashSet::with_capacity(N_VERTICES);
        let mut edges: HashSet<Edge> = HashSet::with_capacity(N_EDGES);

        for r in -radius..=radius {
            let q1 = max(-radius, -r - radius);
            let q2 = min(radius, -r + radius);
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

        let mut edges: Vec<Edge> = edges.into_iter().collect();
        edges.sort();
        // TODO: sort edges
        let edge_ids: HashMap<Edge, EdgeId> =
            edges.iter().enumerate().map(|(id, &e)| (e, id)).collect();

        HexBoard {
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
