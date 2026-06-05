use crate::{bundle::Bundle, common::*};
use DevCard::*;
use enum_map::{EnumMap, enum_map};

#[derive(Clone)]
pub struct Bank {
    pub resources: Bundle,
    pub buildings: EnumMap<Player, Bundle>,
    dev_cards: Bundle,
}

impl Bank {
    pub fn bank() -> Self {
        let cards = (enum_map! {
            Knight => 14,
            VictoryPoint => 5,
            RoadBuilding => 2,
            YearOfPlenty => 2,
            Monopoly => 2
        })
        .into_array();

        let resources = [19; 5];

        Bank {
            resources: Bundle::from_slice(&resources),
            dev_cards: Bundle::from_slice(&cards),
            buildings: enum_map! {
                _ => Bundle::from_slice(&enum_map! {
                    Purchasable::Road => 13, // TODO: change to 15 when setup-phase roads no longer appear out of thin air
                    Purchasable::Settlement => 5,
                    Purchasable::City => 4,
                    _ => 0
                }.into_array())
            },
        }
    }

    pub fn purchasable_count(&self, player: Player, item: Purchasable) -> u8 {
        match item {
            Purchasable::DevCard => self.dev_cards.reduce_sum(),
            _ => self.buildings[player][item],
        }
    }

    pub fn dev_card_weights(&self) -> Vec<f64> {
        self.dev_cards
            .data
            .as_array()
            .iter()
            .map(|&n| n as f64)
            .collect()
    }

    pub fn take_dev_card(&mut self, card: DevCard) {
        assert!(
            self.dev_cards[card] > 0,
            "no more dev cards of type {:?}",
            card
        );
        self.dev_cards[card] -= 1;
    }
}
