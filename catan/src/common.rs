use std::fmt;

use enum_map::{Enum, EnumMap};

pub type HexId = usize;
pub type VertexId = usize;
pub type EdgeId = usize;

pub const N_HEXES: usize = 19;
pub const N_VERTICES: usize = 54;
pub const N_EDGES: usize = 72;
pub const N_ROLLS: usize = 11;

use serde::{Deserialize, Serialize};
pub use Player::*;
pub use Resource::*;

use crate::bundle::Bundle;

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
    ProposeTrade(((Resource, u8), (Resource, u8))),
    AcceptTrade(Player),
    RejectTrade(Player),
    EndTurn,
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
