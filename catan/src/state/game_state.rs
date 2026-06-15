use common::{GameState, Outcome};

use crate::{
    Action, PLAYERS,
    Phase::{self, StealingFromPlayer},
    Player, State,
};

impl GameState for State {
    type Action = Action;
    type Player = Player;

    fn apply_action(&mut self, action: Self::Action) {
        use Action::*;

        let player = self.current_player();
        match action {
            RollDice => self.roll_dice(),
            Roll(value) => { self.handle_dice_roll(value); }
            BuildSettlement(vertex_id) => self.build_settlement(player, vertex_id),
            UpgradeSettlement(vertex_id) => self.upgrade_settlement(player, vertex_id),
            BuildRoad(edge_id) => self.build_road(player, edge_id),
            MoveRobber(hex_id) => self.move_robber(hex_id),
            StealFrom(target) => self.steal_from(target),
            StealResourceFrom(target, res) => self.steal_resource_from(player, target, res),
            DiscardResource(res) => self.discard_resource(player, res),
            PlayDevCard(card) => self.play_dev_card(card),
            Monopolize(res) => self.monopolize(player, res),
            TakeFreeResource(res) => self.take_free_resource(player, res),
            ExchangeResources(((res1, cost), res2)) => self.exchange_resources(player, res1, cost, res2),
            BuyDevCard => self.buy_dev_card(player),
            ReceiveDevCard(card) => self.receive_dev_card(player, card),
            EndTurn => self.end_turn(),
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
