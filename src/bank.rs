use crate::{bundle::Bundle, common::*};
use enum_map::{enum_map, Enum, EnumMap};
use rand::{distr::weighted::WeightedIndex, prelude::Distribution, rng};
use DevCard::*;

pub struct Bank {
    pub resources: Bundle,
    pub buildings: EnumMap<Player, Bundle>,
    dev_cards: Bundle,
    dev_card_rand_index: WeightedIndex<u8>,
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

    pub fn purchasable_count(&self, player: Player, item: Purchasable) -> u8 {
        match item {
            Purchasable::DevCard => self.dev_cards.reduce_sum(),
            _ => self.buildings[player][item],
        }
    }

    pub fn draw_random_dev_card(&mut self) -> DevCard {
        let mut rng = rng();
        let card = DevCard::from_usize(self.dev_card_rand_index.sample(&mut rng));
        self.dev_cards[card] -= 1;
        self.dev_card_rand_index
            .update_weights(&[(card as usize, &self.dev_cards[card])])
            .unwrap();
        card
    }

    pub fn take_dev_card(&mut self, card: DevCard) {
        assert!(
            self.dev_cards[card] > 0,
            "no more dev cards of type {:?}",
            card
        );
        self.dev_cards[card] -= 1;
        self.dev_card_rand_index
            .update_weights(&[(card as usize, &self.dev_cards[card])])
            .unwrap();
    }
}
