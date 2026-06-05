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
    ReceiveDevCard(DevCard),
    PlayDevCard(DevCard),
    MoveRobber(HexId),
    DiscardResource(Resource),
    StealFrom(Player),
    StealResourceFrom(Player, Resource),
    Monopolize(Resource),
    TakeFreeResource(Resource),
    ExchangeResources(((Resource, u8), Resource)),
    EndTurn,
}

impl From<usize> for Action {
    fn from(n: usize) -> Self {
        use Action::*;
        let b0 = n & 0xFF;
        let b1 = (n >> 8) & 0xFF;
        let b2 = (n >> 16) & 0xFF;
        let b3 = (n >> 24) & 0xFF;
        match b0 {
            0 => RollDice,
            1 => Roll(b1 as u8),
            2 => BuildSettlement(b1),
            3 => UpgradeSettlement(b1),
            4 => BuildRoad(b1),
            5 => BuyDevCard,
            6 => ReceiveDevCard(DevCard::from_usize(b1)),
            7 => PlayDevCard(DevCard::from_usize(b1)),
            8 => MoveRobber(b1),
            9 => DiscardResource(Resource::from_usize(b1)),
            10 => StealFrom(Player::from_usize(b1)),
            11 => StealResourceFrom(Player::from_usize(b1), Resource::from_usize(b2)),
            12 => Monopolize(Resource::from_usize(b1)),
            13 => TakeFreeResource(Resource::from_usize(b1)),
            14 => ExchangeResources((
                (Resource::from_usize(b1), b2 as u8),
                Resource::from_usize(b3),
            )),
            15 => Action::EndTurn,
            _ => panic!("invalid action index {n}"),
        }
    }
}

impl From<Action> for usize {
    fn from(value: Action) -> Self {
        use Action::*;
        match value {
            RollDice => 0,
            Roll(v) => 1 | ((v as usize) << 8),
            BuildSettlement(v) => 2 | (v << 8),
            UpgradeSettlement(v) => 3 | (v << 8),
            BuildRoad(e) => 4 | (e << 8),
            BuyDevCard => 5,
            ReceiveDevCard(d) => 6 | (d.into_usize() << 8),
            PlayDevCard(d) => 7 | (d.into_usize() << 8),
            MoveRobber(h) => 8 | (h << 8),
            DiscardResource(r) => 9 | (r.into_usize() << 8),
            StealFrom(p) => 10 | (p.into_usize() << 8),
            StealResourceFrom(p, r) => 11 | (p.into_usize() << 8) | (r.into_usize() << 16),
            Monopolize(r) => 12 | (r.into_usize() << 8),
            TakeFreeResource(r) => 13 | (r.into_usize() << 8),
            ExchangeResources(((from_r, qty), to_r)) => {
                14 | (from_r.into_usize() << 8) | ((qty as usize) << 16) | (to_r.into_usize() << 24)
            }
            Action::EndTurn => 15,
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

pub const ROLL_WEIGHTS: [f64; 11] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

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
            actions.push(StealFrom(p));
        }
        for p in PLAYERS {
            for r in RESOURCES {
                actions.push(StealResourceFrom(p, r));
            }
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
