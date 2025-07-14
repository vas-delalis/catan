use std::{error::Error, process};

use catan::*;

fn run() -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path("roads.csv")?;
    let graphs = RoadGraphIterator::new();
    // dbg!(graphs.count());
    let reference = graphs.reference_board.clone();

    graphs
        // .take(100)
        .map(|bb| (format!("{:x}", bb.value), longest_trail(bb, &reference)))
        .for_each(|t| wtr.serialize(t).unwrap());

    wtr.flush()?;

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}
