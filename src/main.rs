use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    vec,
};

// struct Hex {
//     q: i8,
//     r: i8,
//     s: i8,
// }

// enum HexMaybe {
//     Null,
//     Hex { q: i8, r: i8, s: i8 },
// }

// struct HexBoard {
//     data: Vec<HexMaybe>,
// }

// impl HexBoard {
//     fn new() -> HexBoard {
//         let mut vec: Vec<HexMaybe> = Vec::with_capacity(25);

//         for q in -2i8..3 {
//             for r in -2..3 {
//                 let s = -q - r;
//                 if -2 <= s && s <= 2 {
//                     vec.push(HexMaybe::Hex { q, r, s });
//                 } else {
//                     vec.push(HexMaybe::Null);
//                 }
//             }
//         }

//         HexBoard { data: vec }
//     }

//     fn get(&self, q: i8, r: i8) -> HexMaybe {
//         self.data.get()
//     }
// }

#[derive(Debug, Clone, Copy)]
enum Resource {
    Brick,
    Grain,
    Lumber,
    Ore,
    Wool,
}

static RESOURCE_TYPES: [Resource; 5] = [Brick, Grain, Lumber, Ore, Wool];

// enum Building {
//     Settlement,
//     City,
// }

// enum Harbor {
//     ThreeToOne,
//     TwoToOne(Resource),
// }

// enum Directions {
//     TL,
//     T,
//     TR,
//     BR,
//     B,
//     BL,
// }

#[derive(Debug)]
struct Hex {
    id: usize,
    roll: u8,
    resource: Resource,
    q: i8,
    r: i8,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Vertex {
    q: i8,
    r: i8,
    is_north: bool,
}

impl Vertex {
    fn coords(&self) -> (f64, f64) {
        let (dq, dr) = {
            if self.is_north {
                (1.0 / 3.0, -2.0 / 3.0)
            } else {
                (-1.0 / 3.0, 2.0 / 3.0)
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
        write!(f, "({}, {}, {})", self.q, self.r, {
            if self.is_north {
                "N"
            } else {
                "S"
            }
        })
    }
}

use catan::bundle::Bundle;

use crate::Resource::*;

fn main() {
    println!("Hello, world!");
    let hex_resources = vec![
        Ore, Wool, Lumber, Grain, Brick, Wool, Brick, Grain, Lumber, Ore, Lumber, Ore, Lumber, Ore,
        Grain, Wool, Brick, Grain, Wool,
    ];
    let rolls: Vec<u8> = vec![10, 2, 9, 12, 6, 4, 10, 9, 11, 7, 3, 8, 8, 3, 4, 5, 5, 6, 11];

    let mut hexes = Vec::with_capacity(19);
    let mut id = 0;

    // Make hex vector
    for r in -2i8..=2 {
        for q in -2..=2 {
            let s = -q - r;
            if (-2..=2).contains(&s) {
                hexes.push(Hex {
                    id,
                    q,
                    r,
                    resource: hex_resources[id],
                    roll: rolls[id],
                });
                id += 1;
            }
        }
    }

    // Build adjacency vector
    let mut hex_vertex_adjacency: Vec<Vec<Vertex>> = Vec::with_capacity(19);
    let mut vertices = HashSet::with_capacity(54);
    for hex in &hexes {
        let q = hex.q;
        let r = hex.r;
        let mut adj = Vec::with_capacity(6);
        for (new_q, new_r, is_north) in [
            (q, r, true),
            (q, r, false),
            (q, r + 1, true),
            (q, r - 1, false),
            (q - 1, r + 1, true),
            (q + 1, r - 1, false),
        ] {
            let v = Vertex {
                q: new_q,
                r: new_r,
                is_north,
            };
            adj.push(v.clone());
            vertices.insert(v);
        }
        hex_vertex_adjacency.push(adj);
    }

    let mut vertices: Vec<Vertex> = vertices.into_iter().collect();
    vertices.sort_by(|a, b| a.ordering_value().total_cmp(&b.ordering_value()));

    let mut resource_bitmaps = vec![0u64; 5];
    let mut roll_bitmaps = vec![0u64; 11];
    for (i, hex) in hexes.iter().enumerate() {
        let mut adj_bitmap = 0u64;
        for vertex in &hex_vertex_adjacency[i] {
            let bit = vertices.iter().position(|v| v == vertex).unwrap();
            adj_bitmap |= 1 << bit;
        }
        resource_bitmaps[hex.resource as usize] |= adj_bitmap;
        roll_bitmaps[(hex.roll - 2) as usize] |= adj_bitmap;
    }

    println!("{:X}", roll_bitmaps[4] & resource_bitmaps[Brick as usize]);
    println!("{:X}", roll_bitmaps[4] & resource_bitmaps[Wool as usize]);
    // dbg!(resource_bitmaps[0]);
    // println!("{}", vertices.len());
    // println!("{:?}", vertices);
    // println!(
    //     "{:?}",
    //     vertices
    //         .iter()
    //         .map(|v| v.coords())
    //         .collect::<Vec<(f64, f64)>>()
    // );
    // println!(
    //     "{:?}",
    //     vertices
    //         .iter()
    //         .map(|v| v.ordering_value())
    //         .collect::<Vec<f64>>()
    // );

    // for q in -3i8..=3 {
    //     for r in -3i8..=3 {
    //         let s = -q - r;
    //         // Exclude hexes not in size 3 grid, as well as the two hexes on the far left and right
    //         // (Their N and S vertices don't touch the inner size 2 grid)
    //         if !(-3..=3).contains(&s) || (q == 0 && (r == 3 || r == -3)) {
    //             continue;
    //         }

    //         // For north and south vertex of each hex
    //         for is_north in [true, false] {
    //             // Exclude north verts of north half and south verts of south half of outer ring
    //             if ((q == 3 || r == -3 || s == 3) && is_north)
    //                 || ((q == -3 || r == 3 || s == -3) && !is_north)
    //             {
    //                 continue;
    //             }
    //         }
    //     }
    // }

    println!("{:?}", Bundle::from_bytes([0, 0, 0, 1, 1, 1, 1, 1]));
}
