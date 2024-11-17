use crate::{DevCard, Resource};
use enum_map::{enum_map, EnumMap};

pub struct Stockpile {
    dev_cards: EnumMap<DevCard, u8>,
    resources: EnumMap<Resource, u8>,
}

impl Stockpile {
    pub fn bank() -> Self {
        use DevCard::*;
        Stockpile {
            dev_cards: enum_map! {
                Knight => 14,
                VictoryPoint => 5,
                RoadBuilding => 2,
                YearOfPlenty => 2,
                Monopoly => 2
            },
            resources: enum_map! {
                _ => 19
            },
        }
    }

    pub fn player() -> Self {
        Stockpile {
            dev_cards: EnumMap::default(),
            resources: EnumMap::default(),
        }
    }

    pub fn add(&self, other: Self) -> Self {
        todo!()
    }

    pub fn subtract(&self, other: Self) -> Self {
        todo!()
    }
}
