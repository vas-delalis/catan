use std::{error::Error, process};

use catan::RoadTrailTableLoader;

fn run() -> Result<(), Box<dyn Error>> {
    RoadTrailTableLoader::generate_and_save()?;

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}
