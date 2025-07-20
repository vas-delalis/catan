use std::{error::Error, process};

use catan::*;

fn run() -> Result<(), Box<dyn Error>> {
    RoadTrailTable::generate_and_save()?;

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}
