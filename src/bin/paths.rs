use std::{error::Error, process, time::Instant};

use catan::{Bitboard, RoadTrailTableLoader};

fn run() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let archive = RoadTrailTableLoader::load();

    dbg!(archive.longest_trail(Bitboard::from_hex("c000400480000"), Bitboard::zeros()));

    dbg!(start.elapsed());
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}
