use enum_map::Enum;
use Resource::*;

pub type HexId = usize;
pub type VertexId = usize;
pub type EdgeId = usize;

#[derive(Debug, Clone, Copy, Enum)]
pub enum Resource {
    Brick,
    Grain,
    Lumber,
    Ore,
    Wool,
}

pub static RESOURCE_TYPES: [Resource; 5] = [Brick, Grain, Lumber, Ore, Wool];

pub enum Building {
    Settlement,
    City,
}

pub enum Action {
    RollDice,
    BuildSettlement(VertexId),
    UpgradeSettlement(VertexId),
    BuildRoad(EdgeId),
    MoveRobber(HexId),
    ExchangeResources(((Resource, u8), (Resource, u8))),
    DiscardResources,
    EndTurn,
}

#[derive(Debug, Clone, Copy, Enum)]
pub enum Player {
    Blue,
    Orange,
    Red,
    White,
}

const PLAYERS: [Player; 4] = [Player::Blue, Player::Orange, Player::Red, Player::White];

#[derive(Debug, Enum)]
pub enum DevCard {
    Knight,
    VictoryPoint,
    RoadBuilding,
    YearOfPlenty,
    Monopoly,
}
