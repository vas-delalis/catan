use enum_map::Enum;

pub type HexId = usize;
pub type VertexId = usize;
pub type EdgeId = usize;

pub const N_HEXES: usize = 19;
pub const N_VERTICES: usize = 54;
pub const N_EDGES: usize = 72;
pub const N_ROLLS: usize = 11;

pub use Player::*;
pub use Resource::*;

#[derive(Debug, Clone, Copy, Enum)]
pub enum Resource {
    Brick,
    Grain,
    Lumber,
    Ore,
    Wool,
}

pub static RESOURCES: [Resource; 5] = [Brick, Grain, Lumber, Ore, Wool];

#[derive(Debug, Enum, Clone, Copy)]
pub enum Purchasable {
    Road,
    Settlement,
    City,
    DevCard,
}

pub enum Action {
    RollDice,
    BuildSettlement(VertexId),
    UpgradeSettlement(VertexId),
    BuildRoad(EdgeId),
    BuyDevCard,
    MoveRobber(HexId),
    ExchangeResources(((Resource, u8), (Resource, u8))),
    DiscardResources,
    EndTurn,
}

#[derive(Debug, Clone, Copy, Enum, PartialEq, Eq)]
pub enum Player {
    Blue,
    Orange,
    Red,
    White,
}

pub const PLAYERS: [Player; 4] = [Player::Blue, Player::Orange, Player::Red, Player::White];

#[derive(Debug, Enum, Clone, Copy)]
pub enum DevCard {
    Knight,
    VictoryPoint,
    RoadBuilding,
    YearOfPlenty,
    Monopoly,
}
