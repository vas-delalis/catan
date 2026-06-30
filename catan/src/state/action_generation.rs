use std::simd::cmp::SimdPartialOrd;

use common::GameState;

use crate::{Action::*, DevCard, State, bundle::BUY_COSTS, common::*};

impl State {
    pub(super) fn get_normal_actions(&self, player: Player) -> Vec<Action> {
        let mut actions = vec![EndTurn];
        let resources = &self.player_resources[player];

        // BuyDevCard, BuildSettlement, UpgradeSettlement, BuildRoad
        for (item, cost) in BUY_COSTS.iter() {
            if self.bank.purchasable_count(player, item) > 0
                && cost.data.simd_le(self.bank.resources.data).all()
                && cost.data.simd_le(resources.data).all()
            {
                match item {
                    Purchasable::DevCard => actions.push(BuyDevCard),
                    Purchasable::Settlement => {
                        let slots = self.board.available_settlements(player);
                        actions.extend(slots.map(BuildSettlement));
                    }
                    Purchasable::City => {
                        let settlements = self.board.settlements(player);
                        actions.extend(settlements.map(UpgradeSettlement));
                    }
                    Purchasable::Road => {
                        actions.extend(self.get_build_road_actions(player));
                    }
                }
            }
        }

        // PlayDevCard
        if !self.has_played_dev_card {
            use DevCard::*;

            actions.extend(
                [Knight, Monopoly, RoadBuilding, YearOfPlenty]
                    .into_iter()
                    .filter_map(|card| {
                        if self.dev_cards[player][card] > 0 && !self.locked_dev_cards[card] {
                            Some(PlayDevCard(card))
                        } else {
                            None
                        }
                    }),
            );
        }

        // ExchangeResource (maritime trade)
        actions.append(&mut self.get_exchange_actions(player));
        actions
    }

    pub(super) fn get_robber_actions(&self) -> Vec<Action> {
        (0..N_HEXES)
            .filter(|&id| id != self.board.robber_hex_id())
            .map(MoveRobber)
            .collect()
    }

    pub(super) fn get_hex_steal_actions(&self, hex_id: HexId) -> Vec<Action> {
        self.board
            .players_on_hex(hex_id)
            .into_iter()
            .filter(|&p| p != self.current_player() && self.player_resources[p].count_nonzero() > 0)
            .map(StealFrom)
            .collect()
    }

    /// Precondition: target has at least one resource card.
    pub(super) fn get_player_steal_actions(
        &self,
        target: Player,
    ) -> (Vec<Action>, Option<Vec<f64>>) {
        let counts = &self.player_resources[target].data[..5];
        let mut actions = vec![];
        let mut weights = vec![];
        for i in 0..5 {
            if counts[i] > 0 {
                actions.push(StealResourceFrom(target, RESOURCES[i]));
                weights.push(counts[i] as f64);
            }
        }
        (actions, Some(weights))
    }

    pub(super) fn get_discard_actions(&self, player: Player) -> Vec<Action> {
        RESOURCES
            .into_iter()
            .filter(|&r| self.player_resources[player][r] > 0)
            .map(DiscardResource)
            .collect()
    }

    /// Returns possible actions for exchanging resources (maritime trade).
    pub(super) fn get_exchange_actions(&self, player: Player) -> Vec<Action> {
        let mut actions = vec![];
        let ratios = self.board.exchange_ratios(player);

        for res1 in RESOURCES {
            // Check if player has enough of res1
            if self.player_resources[player][res1] < ratios[res1] {
                continue;
            }
            for res2 in RESOURCES {
                // Prevent nonsensical trades and ensure bank has enough of res2 in stock
                if res1 == res2 || self.bank.resources[res2] == 0 {
                    continue;
                }
                actions.push(ExchangeResources(((res1, ratios[res1]), res2)));
            }
        }
        actions
    }

    pub(super) fn get_receive_dev_card_actions(&self) -> (Vec<Action>, Option<Vec<f64>>) {
        let counts = self.bank.dev_card_weights();
        let mut actions = vec![];
        let mut weights = vec![];
        for i in 0..5 {
            if counts[i] > 0.0 {
                actions.push(ReceiveDevCard(DEV_CARDS[i]));
                weights.push(counts[i]);
            }
        }
        (actions, Some(weights))
    }

    pub(super) fn get_build_road_actions(&self, player: Player) -> Vec<Action> {
        let slots = self.board.available_roads(player);
        slots.map(BuildRoad).collect()
    }

    pub(super) fn get_year_of_plenty_actions(&self) -> Vec<Action> {
        RESOURCES
            .into_iter()
            .filter_map(|res| {
                if self.bank.resources[res] > 0 {
                    Some(TakeFreeResource(res))
                } else {
                    None
                }
            })
            .collect()
    }

    pub(super) fn get_monopoly_actions(&self) -> Vec<Action> {
        RESOURCES.into_iter().map(Monopolize).collect()
    }

    pub(super) fn get_roll_actions(&self) -> (Vec<Action>, Option<Vec<f64>>) {
        (Vec::from(ROLL_ACTIONS), Some(Vec::from(ROLL_WEIGHTS)))
    }
}

const ROLL_ACTIONS: [Action; 11] = [
    Roll(2),
    Roll(3),
    Roll(4),
    Roll(5),
    Roll(6),
    Roll(7),
    Roll(8),
    Roll(9),
    Roll(10),
    Roll(11),
    Roll(12),
];
