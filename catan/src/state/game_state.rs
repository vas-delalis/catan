use std::simd::cmp::SimdPartialOrd;

use common::{GameState, Outcome};
use enum_map::EnumMap;

use crate::{
    Action, DevCard, PLAYERS, Phase, Player, Purchasable, RESOURCES, State,
    bundle::{BUY_COSTS, Bundle},
};
use Action::*;

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
const ROLL_WEIGHTS: [f64; 11] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

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
                    Phase::Normal { has_rolled } => {
                        self.give_to_bank(player, BUY_COSTS[Purchasable::Road]);
                        Phase::Normal { has_rolled }
                    }
                    Phase::RoadBuilding(1) => Phase::Normal { has_rolled: true },
                    Phase::RoadBuilding(remaining) => Phase::RoadBuilding(remaining - 1),
                    Phase::Setup => Phase::Setup,
                    _ => panic!("tried to build road in invalid phase"),
                }
            }
            MoveRobber(hex_id) => {
                self.board.move_robber(hex_id);
                self.phase = if !self.get_steal_actions(hex_id).is_empty() {
                    Phase::StealingResources(hex_id)
                } else {
                    Phase::Normal { has_rolled: true }
                }
            }
            StealResource(target) => {
                self.steal_resource(self.current_player(), target);
                self.phase = Phase::Normal { has_rolled: true };
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
                    let count = self.player_data[p].resources[res];
                    total += count;
                    self.player_data[p].resources[res] = 0;
                }
                self.player_data[player].resources[res] += total;
                self.phase = Phase::Normal { has_rolled: true };
            }
            TakeFreeResource(res) => {
                let mut bundle = Bundle::splat(0);
                bundle[res] += 1;
                self.take_from_bank(player, bundle);
                self.phase = match self.phase {
                    Phase::YearOfPlenty(1) => Phase::Normal { has_rolled: true },
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
                let card = self.bank.draw_random_dev_card();
                // You can only play one dev card per turn.
                // This means that we don't need to track how many cards of each type are locked.
                if self.dev_cards[player][card] == 0 {
                    self.locked_dev_cards[card] = true;
                }
                self.dev_cards[player][card] += 1;
            }
            EndTurn => {
                self.whose_turn = self.turn_order[(self.whose_turn as usize + 1) % 4];
                // TODO: reset state variables
                self.phase = Phase::Normal { has_rolled: false }; // EndTurn can only be offered when Phase == Normal
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
            return (vec![], None); // TODO: allow concurrent actions (e.g. when multiple players must discard)
        }
        let player_data = &self.player_data[player];

        match self.phase {
            Phase::Normal { has_rolled } => {
                let mut actions = vec![];
                actions.push(if !has_rolled { RollDice } else { EndTurn });

                // BuyDevCard, BuildSettlement, UpgradeSettlement, BuildRoad
                for (item, cost) in BUY_COSTS.iter() {
                    if self.bank.purchasable_count(player, item) > 0
                        && cost.data.simd_le(self.bank.resources.data).all()
                        && cost.data.simd_le(player_data.resources.data).all()
                    {
                        match item {
                            Purchasable::DevCard => actions.push(BuyDevCard),
                            Purchasable::Settlement => {
                                let slots = self.board.available_settlements(player);
                                actions.extend(slots.map(|v_id| BuildSettlement(v_id)));
                            }
                            Purchasable::City => {
                                let settlements = self.board.settlements(player);
                                actions.extend(settlements.map(|v_id| UpgradeSettlement(v_id)));
                            }
                            Purchasable::Road => {
                                let slots = self.board.available_roads(player);
                                actions.extend(slots.map(|edge_id| BuildRoad(edge_id)));
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
                                if self.dev_cards[player][card] > 0 && !self.locked_dev_cards[card]
                                {
                                    Some(PlayDevCard(card))
                                } else {
                                    None
                                }
                            }),
                    );
                }

                // ExchangeResource (maritime trade)
                actions.append(&mut self.get_exchange_actions(player));
                (actions, None)
            }
            Phase::Rolling => (Vec::from(ROLL_ACTIONS), Some(Vec::from(ROLL_WEIGHTS))),
            Phase::Setup => todo!(),
            Phase::StealingResources(hex_id) => (self.get_steal_actions(hex_id), None),
            Phase::Discarding(_) => (self.get_discard_actions(self.current_player()), None),
            Phase::MovingRobber => (self.get_robber_actions(), None),
            Phase::RoadBuilding(remaining) => {
                let slots = self.board.available_roads(player);
                // dbg!(player);
                // dbg!(self.board.available_roads(player));
                // dbg!(PLAYERS
                //     .into_iter()
                //     .flat_map(|p| {
                //         let settlements = self
                //             .board
                //             .settlements(p)
                //             .map(move |v| (p, self.board.vertex(v), false));
                //         let cities = self
                //             .board
                //             .cities(p)
                //             .map(move |v| (p, self.board.vertex(v), true));
                //         settlements.chain(cities)
                //     })
                //     .collect::<Vec<(Player, Vertex, bool)>>());
                // dbg!(PLAYERS
                //     .into_iter()
                //     .flat_map(|p| self.board.roads(p).map(move |e| (p, self.board.edge(e))))
                //     .collect::<Vec<(Player, Edge)>>());
                assert!(slots.count_ones() > 0);
                (
                    slots
                        .take(remaining as usize)
                        .map(|edge_id| BuildRoad(edge_id))
                        .collect(),
                    None,
                )
            }
            Phase::YearOfPlenty(_) => (
                RESOURCES
                    .into_iter()
                    .filter_map(|res| {
                        if self.bank.resources[res] > 0 {
                            Some(TakeFreeResource(res))
                        } else {
                            None
                        }
                    })
                    .collect(),
                None,
            ),
            Phase::Monopoly => (RESOURCES.into_iter().map(|r| Monopolize(r)).collect(), None),
        }
    }

    fn is_random(&self) -> bool {
        self.phase == Phase::Rolling
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
        Some((Outcome::Loss, -1.0))
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
