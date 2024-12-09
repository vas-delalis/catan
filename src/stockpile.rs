use crate::{bundle::Bundle, common::*};
use enum_map::{enum_map, Enum, EnumMap};
use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng};
use DevCard::*;

pub struct Stockpile {
    pub resources: Bundle,
    dev_cards: Bundle,
    dev_card_rand_index: WeightedIndex<u8>,
    buildings: EnumMap<Player, Bundle>,
}

impl Stockpile {
    pub fn bank() -> Self {
        let cards = (enum_map! {
            Knight => 14,
            VictoryPoint => 5,
            RoadBuilding => 2,
            YearOfPlenty => 2,
            Monopoly => 2
        })
        .into_array();

        Stockpile {
            resources: Bundle::from_slice(&[19; 5]),
            dev_cards: Bundle::from_slice(&cards),
            dev_card_rand_index: WeightedIndex::new(cards).unwrap(),
            buildings: enum_map! {
                _ => Bundle::from_slice(&enum_map! {
                    Purchasable::Road => 15,
                    Purchasable::Settlement => 5,
                    Purchasable::City => 4,
                    _ => 0
                }.into_array())
            },
        }
    }

    pub fn has_purchasable(&self, player: Player, item: Purchasable) -> bool {
        match item {
            Purchasable::DevCard => self.dev_cards.reduce_sum() > 0,
            _ => self.buildings[player][item] > 0,
        }
    }

    pub fn return_settlement(&mut self, player: Player) {
        self.buildings[player][Purchasable::Settlement] += 1
    }

    pub fn take_dev_card(&mut self) -> DevCard {
        let mut rng = thread_rng();
        let card = DevCard::from_usize(self.dev_card_rand_index.sample(&mut rng));
        self.dev_cards[card] -= 1;
        self.dev_card_rand_index
            .update_weights(&[(card as usize, &self.dev_cards[card])])
            .unwrap();
        card
    }

    pub fn take(&mut self, player: Player, item: Purchasable) -> Option<DevCard> {
        let mut rng = thread_rng();
        match item {
            Purchasable::DevCard => Some(self.take_dev_card()),
            _ => {
                self.buildings[player][item] -= 1;
                None
            }
        }
    }
}
