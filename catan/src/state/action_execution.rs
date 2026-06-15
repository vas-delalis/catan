use enum_map::EnumMap;

use crate::{
    Phase, State,
    bundle::{BUY_COSTS, Bundle},
    common::*,
};

impl State {
    pub(super) fn roll_dice(&mut self) {
        self.phase = Phase::Rolling;
    }

    pub(super) fn build_settlement(&mut self, player: Player, vertex_id: VertexId) {
        self.give_to_bank(player, BUY_COSTS[Purchasable::Settlement]);
        self.bank.buildings[player][Purchasable::Settlement] -= 1;
        self.board.add_settlement(player, vertex_id);
    }

    pub(super) fn upgrade_settlement(&mut self, player: Player, vertex_id: VertexId) {
        self.give_to_bank(player, BUY_COSTS[Purchasable::City]);
        self.bank.buildings[player][Purchasable::Settlement] += 1;
        self.bank.buildings[player][Purchasable::City] -= 1;
        self.board.upgrade_settlement(vertex_id);
    }

    pub(super) fn build_road(&mut self, player: Player, edge_id: EdgeId) {
        self.board.add_road(player, edge_id);
        self.bank.buildings[player][Purchasable::Road] -= 1;
        self.phase = match self.phase {
            Phase::Normal => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::Road]);
                Phase::Normal
            }
            Phase::RoadBuilding(1) => Phase::Normal,
            Phase::RoadBuilding(remaining) => Phase::RoadBuilding(remaining - 1),
            Phase::Setup => Phase::Setup,
            _ => panic!("tried to build road in invalid phase"),
        }
    }

    pub(super) fn move_robber(&mut self, hex_id: HexId) {
        self.board.move_robber(hex_id);
        self.phase = if !self.get_hex_steal_actions(hex_id).is_empty() {
            Phase::StealingFromHex(hex_id)
        } else {
            Phase::Normal
        }
    }

    pub(super) fn steal_from(&mut self, target: Player) {
        self.phase = Phase::StealingFromPlayer(target);
    }

    pub(super) fn steal_resource_from(&mut self, player: Player, target: Player, res: Resource) {
        self.player_resources[target][res] -= 1;
        self.player_resources[player][res] += 1;
        self.phase = Phase::Normal;
    }

    pub(super) fn discard_resource(&mut self, player: Player, res: Resource) {
        let mut bundle = Bundle::splat(0);
        bundle[res] += 1;
        self.give_to_bank(player, bundle);
        self.phase = match self.phase {
            Phase::Discarding(mut remaining) => {
                remaining[player] -= 1;
                if remaining.reduce_sum() == 0 {
                    Phase::MovingRobber
                } else {
                    Phase::Discarding(remaining)
                }
            }
            _ => panic!("tried to discard in invalid phase"),
        };
    }

    pub(super) fn play_dev_card(&mut self, card: DevCard) {
        self.activate_dev_card(card);
        self.has_played_dev_card = true;
        self.dev_cards[self.whose_turn][card] -= 1;
    }

    pub(super) fn monopolize(&mut self, player: Player, res: Resource) {
        let mut total = 0;
        for p in PLAYERS {
            if p == player {
                continue;
            }
            let count = self.player_resources[p][res];
            total += count;
            self.player_resources[p][res] = 0;
        }
        self.player_resources[player][res] += total;
        self.phase = Phase::Normal;
    }

    pub(super) fn take_free_resource(&mut self, player: Player, res: Resource) {
        let mut bundle = Bundle::splat(0);
        bundle[res] += 1;
        self.take_from_bank(player, bundle);
        self.phase = match self.phase {
            Phase::YearOfPlenty(1) => Phase::Normal,
            Phase::YearOfPlenty(remaining) => Phase::YearOfPlenty(remaining - 1),
            _ => panic!("tried to take resource in invalid phase"),
        };
    }

    pub(super) fn exchange_resources(
        &mut self,
        player: Player,
        res1: Resource,
        cost: u8,
        res2: Resource,
    ) {
        let mut bundle = Bundle::splat(0);
        bundle[res1] = cost;
        self.give_to_bank(player, bundle);
        let mut bundle = Bundle::splat(0);
        bundle[res2] = 1;
        self.take_from_bank(player, bundle);
    }

    pub(super) fn buy_dev_card(&mut self, player: Player) {
        self.give_to_bank(player, BUY_COSTS[Purchasable::DevCard]);
        self.phase = Phase::BuyingDevCard;
    }

    pub(super) fn receive_dev_card(&mut self, player: Player, card: DevCard) {
        // You can only play one dev card per turn, so a card received this turn is locked
        // until next turn even if no other copy was held beforehand.
        if self.dev_cards[player][card] == 0 {
            self.locked_dev_cards[card] = true;
        }
        self.bank.take_dev_card(card);
        self.dev_cards[player][card] += 1;
        self.phase = Phase::Normal;
    }

    pub(super) fn end_turn(&mut self) {
        self.whose_turn = self.turn_order[(self.whose_turn as usize + 1) % 4];
        self.phase = Phase::Rolling;
        self.locked_dev_cards = EnumMap::from_fn(|_| false);
        self.has_played_dev_card = false;
    }
}
