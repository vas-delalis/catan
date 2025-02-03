use enum_map::Enum;

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

#[derive(Debug, Clone, Copy, Enum, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, Enum, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Player {
    Blue,
    Orange,
    Red,
    White,
}

pub const PLAYERS: [Player; 4] = [Player::Blue, Player::Orange, Player::Red, Player::White];

#[derive(Debug, Enum, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevCard {
    Knight,
    VictoryPoint,
    RoadBuilding,
    YearOfPlenty,
    Monopoly,
}

#[derive(Serialize)]
pub enum ActionResult {
    DiceRolled(u8, Bundle),
    DevCardBought(DevCard),
    Monopolized(Resource, u8),
    ResourceStolen(Resource),
}
