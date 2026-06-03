use std::fmt;

use enum_map::{Enum, EnumMap};
use serde::{Deserialize, Serialize};

use crate::bundle::Bundle;
use common::Player as PlayerTrait;

pub use Player::*;
pub use Resource::*;

pub type HexId = usize;
pub type VertexId = usize;
pub type EdgeId = usize;

pub const N_HEXES: usize = 19;
pub const N_VERTICES: usize = 54;
pub const N_EDGES: usize = 72;
pub const N_ROLLS: usize = 11;

#[derive(Debug, Clone, Copy, Enum, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Resource {
    Brick,
    Grain,
    Lumber,
    Ore,
    Wool,
}

pub const RESOURCES: [Resource; 5] = [Brick, Grain, Lumber, Ore, Wool];

#[derive(Debug, Enum, Clone, Copy)]
pub enum Purchasable {
    Road,
    Settlement,
    City,
    DevCard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Action {
    RollDice,
    Roll(u8),
    BuildSettlement(VertexId),
    UpgradeSettlement(VertexId),
    BuildRoad(EdgeId),
    BuyDevCard,
    PlayDevCard(DevCard),
    MoveRobber(HexId),
    DiscardResource(Resource),
    StealResource(Player),
    Monopolize(Resource),
    TakeFreeResource(Resource),
    ExchangeResources(((Resource, u8), Resource)),
    EndTurn,
}

impl From<usize> for Action {
    fn from(n: usize) -> Self {
        let b0 = n & 0xFF;
        let b1 = (n >> 8) & 0xFF;
        let b2 = (n >> 16) & 0xFF;
        let b3 = (n >> 24) & 0xFF;
        match b0 {
            0 => Action::RollDice,
            1 => Action::Roll(b1 as u8),
            2 => Action::BuildSettlement(b1),
            3 => Action::UpgradeSettlement(b1),
            4 => Action::BuildRoad(b1),
            5 => Action::BuyDevCard,
            6 => Action::PlayDevCard(DevCard::from_usize(b1)),
            7 => Action::MoveRobber(b1),
            8 => Action::DiscardResource(Resource::from_usize(b1)),
            9 => Action::StealResource(Player::from_usize(b1)),
            10 => Action::Monopolize(Resource::from_usize(b1)),
            11 => Action::TakeFreeResource(Resource::from_usize(b1)),
            12 => Action::ExchangeResources((
                (Resource::from_usize(b1), b2 as u8),
                Resource::from_usize(b3),
            )),
            13 => Action::EndTurn,
            _ => panic!("invalid action index {n}"),
        }
    }
}

impl From<Action> for usize {
    fn from(value: Action) -> Self {
        match value {
            Action::RollDice => 0,
            Action::Roll(v) => 1 | ((v as usize) << 8),
            Action::BuildSettlement(v) => 2 | (v << 8),
            Action::UpgradeSettlement(v) => 3 | (v << 8),
            Action::BuildRoad(e) => 4 | (e << 8),
            Action::BuyDevCard => 5,
            Action::PlayDevCard(d) => 6 | (d.into_usize() << 8),
            Action::MoveRobber(h) => 7 | (h << 8),
            Action::DiscardResource(r) => 8 | (r.into_usize() << 8),
            Action::StealResource(p) => 9 | (p.into_usize() << 8),
            Action::Monopolize(r) => 10 | (r.into_usize() << 8),
            Action::TakeFreeResource(r) => 11 | (r.into_usize() << 8),
            Action::ExchangeResources(((from_r, qty), to_r)) => {
                12 | (from_r.into_usize() << 8) | ((qty as usize) << 16) | (to_r.into_usize() << 24)
            }
            Action::EndTurn => 13,
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(
    Debug, Clone, Copy, Enum, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash,
)]
pub enum Player {
    Blue,
    Orange,
    Red,
    White,
}

impl From<Player> for usize {
    fn from(value: Player) -> Self {
        value as usize
    }
}

impl PlayerTrait for Player {
    const LEN: usize = 4;
    fn list() -> Vec<Self> {
        vec![Blue, Orange, Red, White]
    }
}

impl Player {
    pub fn next(&self) -> Player {
        match *self {
            Blue => Orange,
            Orange => Red,
            Red => White,
            White => Blue,
        }
    }

    pub fn enemies(&self) -> [Player; 3] {
        match *self {
            Blue => [Orange, Red, White],
            Orange => [Blue, Red, White],
            Red => [Blue, Orange, White],
            White => [Blue, Orange, Red],
        }
    }
}

pub const PLAYERS: [Player; 4] = [Blue, Orange, Red, White];

#[derive(Debug, Enum, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum DevCard {
    Knight,
    VictoryPoint,
    RoadBuilding,
    YearOfPlenty,
    Monopoly,
}

pub const DEV_CARDS: [DevCard; 5] = [
    DevCard::Knight,
    DevCard::VictoryPoint,
    DevCard::RoadBuilding,
    DevCard::YearOfPlenty,
    DevCard::VictoryPoint,
];

#[derive(Serialize)]
pub enum ActionResult {
    DiceRolled(u8, Option<EnumMap<Player, Bundle>>),
    DevCardBought(DevCard),
    Monopolized(Resource, u8),
    ResourceStolen(Resource),
}

#[derive(Serialize)]
pub struct InitialObservation {
    pub resources: Vec<Option<Resource>>,
    pub rolls: Vec<Option<u8>>,
    // TODO: harbors
}

#[derive(Debug, Serialize)]
pub struct Observation {
    pub observer: Player,
    pub current_player: Player,
    pub is_terminal: bool,
    pub actions: Vec<Action>,
    pub observer_hand: ObserverHand,
    pub hidden_hands: Vec<HiddenHand>,
    pub robber: HexId,
    pub buildings: Vec<(Player, VertexId, bool)>,
    pub roads: Vec<(Player, EdgeId)>,
}

#[derive(Debug, Serialize)]
pub struct ObserverHand {
    pub resources: EnumMap<Resource, u8>,
    pub dev_cards: EnumMap<DevCard, u8>,
}

#[derive(Debug, Serialize)]
pub struct HiddenHand {
    pub player: Player,
    pub resources: u8,
    pub dev_cards: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_usize_conversion() {
        use Action::*;
        let mut actions = vec![];
        actions.push(RollDice);
        for roll in 2u8..=12 {
            actions.push(Roll(roll));
        }
        for v in 0..N_VERTICES {
            actions.push(BuildSettlement(v));
        }
        for v in 0..N_VERTICES {
            actions.push(UpgradeSettlement(v));
        }
        for e in 0..N_EDGES {
            actions.push(BuildRoad(e));
        }
        actions.push(BuyDevCard);
        for card in DEV_CARDS {
            actions.push(PlayDevCard(card));
        }
        for h in 0..N_HEXES {
            actions.push(MoveRobber(h));
        }
        for r in RESOURCES {
            actions.push(DiscardResource(r));
        }
        for p in PLAYERS {
            actions.push(StealResource(p));
        }
        for r in RESOURCES {
            actions.push(Monopolize(r));
        }
        for r in RESOURCES {
            actions.push(TakeFreeResource(r));
        }
        for from_r in RESOURCES {
            for to_r in RESOURCES {
                for qty in [2u8, 3, 4] {
                    actions.push(ExchangeResources(((from_r, qty), to_r)));
                }
            }
        }
        actions.push(EndTurn);

        for action in actions {
            let n: usize = action.into();
            assert_eq!(Action::from(n), action, "round-trip failed for {action:?}");
        }
    }
}
