use std::simd::cmp::SimdPartialOrd;

use common::{GameState, Outcome};

use crate::{
    bundle::BUY_COSTS, Action, DevCard, Phase, Player, Purchasable, State, PLAYERS, RESOURCES,
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

    fn apply_action(&mut self, _action: Self::Action) {
        todo!()
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
        todo!()
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
        todo!()
        // if let Some(outcome1) = self.outcome(player1) && let Some(outcome2) = self.outcome(player2) {

        // }
    }
}
