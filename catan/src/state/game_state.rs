use common::{GameState, Outcome};
use enum_map::EnumMap;

use crate::{
    Action, PLAYERS,
    Phase::{self, StealingFromPlayer},
    Player, Purchasable, State,
    bundle::{BUY_COSTS, Bundle},
};

impl GameState for State {
    type Action = Action;
    type Player = Player;

    fn apply_action(&mut self, action: Self::Action) {
        use Action::*;

        let player = self.current_player();
        match action {
            RollDice => {
                self.phase = Phase::Rolling;
            }
            Roll(value) => {
                self.handle_dice_roll(value);
            }
            BuildSettlement(vertex_id) => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::Settlement]);
                self.bank.buildings[player][Purchasable::Settlement] -= 1;
                self.board.add_settlement(player, vertex_id);
            }
            UpgradeSettlement(vertex_id) => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::City]);
                self.bank.buildings[player][Purchasable::Settlement] += 1;
                self.bank.buildings[player][Purchasable::City] -= 1;
                self.board.upgrade_settlement(vertex_id);
            }
            BuildRoad(edge_id) => {
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
            MoveRobber(hex_id) => {
                self.board.move_robber(hex_id);
                self.phase = if !self.get_hex_steal_actions(hex_id).is_empty() {
                    Phase::StealingFromHex(hex_id)
                } else {
                    Phase::Normal
                }
            }
            StealFrom(target) => {
                self.phase = Phase::StealingFromPlayer(target);
            }
            StealResourceFrom(target, res) => {
                self.player_resources[target][res] -= 1;
                let player_bundle = &mut self.player_resources[player];
                player_bundle[res] += 1;
                self.phase = Phase::Normal;
            }
            DiscardResource(res) => {
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
            PlayDevCard(card) => {
                self.activate_dev_card(card);
                self.has_played_dev_card = true;
                self.dev_cards[self.whose_turn][card] -= 1;
            }
            Monopolize(res) => {
                // Take all of res from other players
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
            TakeFreeResource(res) => {
                let mut bundle = Bundle::splat(0);
                bundle[res] += 1;
                self.take_from_bank(player, bundle);
                self.phase = match self.phase {
                    Phase::YearOfPlenty(1) => Phase::Normal,
                    Phase::YearOfPlenty(remaining) => Phase::YearOfPlenty(remaining - 1),
                    _ => panic!("tried to take resource in invalid phase"),
                };
            }
            ExchangeResources(((res1, cost), res2)) => {
                let mut bundle = Bundle::splat(0);
                bundle[res1] = cost;
                self.give_to_bank(player, bundle);
                let mut bundle = Bundle::splat(0);
                bundle[res2] = 1;
                self.take_from_bank(player, bundle);
            }
            BuyDevCard => {
                self.give_to_bank(player, BUY_COSTS[Purchasable::DevCard]);
                self.phase = Phase::BuyingDevCard;
            }
            ReceiveDevCard(card) => {
                // You can only play one dev card per turn.
                // This means that we don't need to track how many cards of each type are locked.
                if self.dev_cards[player][card] == 0 {
                    self.locked_dev_cards[card] = true;
                }
                self.bank.take_dev_card(card);
                self.dev_cards[player][card] += 1;
                self.phase = Phase::Normal;
            }
            EndTurn => {
                self.whose_turn = self.turn_order[(self.whose_turn as usize + 1) % 4];
                // TODO: reset state variables
                self.phase = Phase::Rolling;
                self.locked_dev_cards = EnumMap::from_fn(|_| false);
                self.has_played_dev_card = false;
            }
        }
    }

    fn current_player(&self) -> Self::Player {
        match self.phase {
            Phase::Discarding(remaining) => *PLAYERS.iter().find(|&p| remaining[*p] > 0).unwrap(),
            Phase::Setup => {
                todo!()
            }
            _ => self.whose_turn,
        }
    }

    fn get_actions(&self, player: Self::Player) -> (Vec<Self::Action>, Option<Vec<f64>>) {
        if player != self.current_player() {
            return (vec![], None);
        }

        use Phase::*;
        let actions = match self.phase {
            Rolling => return self.get_roll_actions(),
            BuyingDevCard => return self.get_receive_dev_card_actions(),
            StealingFromPlayer(p) => return self.get_player_steal_actions(p),
            Setup => todo!(),
            Monopoly => self.get_monopoly_actions(),
            Normal => self.get_normal_actions(player),
            MovingRobber => self.get_robber_actions(),
            YearOfPlenty(_) => self.get_year_of_plenty_actions(),
            Discarding(_) => self.get_discard_actions(self.current_player()),
            StealingFromHex(hex_id) => self.get_hex_steal_actions(hex_id),
            RoadBuilding(remaining) => self.get_road_building_actions(player, remaining),
        };
        assert_ne!(actions.len(), 0);
        (actions, None)
    }

    fn is_random(&self) -> bool {
        match self.phase {
            Phase::Rolling | StealingFromPlayer(_) | Phase::BuyingDevCard => true,
            _ => false,
        }
    }

    fn is_terminal(&self) -> bool {
        self.victory_points(self.whose_turn) >= 10
    }

    fn name() -> String {
        String::from("Catan")
    }

    fn new() -> Self {
        Self::default()
    }

    fn outcome(&self, player: Self::Player) -> Option<(Outcome, f32)> {
        let current_player_wins = self.victory_points(self.whose_turn) >= 10;
        if !current_player_wins {
            return None;
        }
        if player == self.whose_turn {
            return Some((Outcome::Win, 1.0));
        }
        Some((Outcome::Loss, -0.3333))
    }

    fn pairwise_outcome(
        &self,
        player1: Self::Player,
        player2: Self::Player,
    ) -> Option<(Outcome, f32)> {
        use Outcome::*;
        if let Some((outcome1, _)) = self.outcome(player1)
            && let Some((outcome2, _)) = self.outcome(player2)
        {
            return match (outcome1, outcome2) {
                (Win, Loss) => Some((Win, 1.0)),
                (Loss, Win) => Some((Loss, -1.0)),
                _ => Some((Draw, 0.0)),
            };
        } else {
            return None;
        }
    }
}
