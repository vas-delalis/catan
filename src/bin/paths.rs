use std::{collections::HashMap, error::Error, process};

use catan::State;

fn run() -> Result<(), Box<dyn Error>> {
    // let mut reader = csv::Reader::from_path("roads.csv")?;
    // let records = reader.records();
    // let mut graph_lengths: HashMap<u128, u8> = HashMap::with_capacity(30_000_000);
    // for r in records {
    //     let r = r?;
    //     let graph = u128::from_str_radix(r.get(0).unwrap(), 16)?;
    //     let length: u8 = r.get(1).unwrap().parse()?;
    //     graph_lengths.insert(graph, length);
    // }
    // let l = graph_lengths.get(&u128::from_str_radix("7c1e601c00200000", 16)?);
    // dbg!(l);
    // let lengths: HashMap<String, u8> = HashMap::from_iter(records);

    let s = State::default();
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}
