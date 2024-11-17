use enum_map::EnumMap;

use crate::{bundle::Bundle, EdgeId, Player, Resource, VertexId};

type VertexMap = u64;

pub struct Board {
    player_buildings: EnumMap<Player, VertexMap>,
    resources: EnumMap<Resource, VertexMap>,
    cities: VertexMap,
}

impl Board {
    pub fn add_settlement(&self, player: Player, vertex_id: VertexId) {}

    pub fn upgrade_settlement(&self, player: Player, vertex_id: VertexId) {}

    pub fn add_road(&self, player: Player, edge_id: EdgeId) {}

    pub fn resource_production(&self, roll: u8) -> EnumMap<Player, Bundle> {
        let bundles = EnumMap::default();
        let roll: VertexMap = 0; // TODO: get roll map

        // resources = [u64; 4];
        // a = [roll; 4]
        // b = a & resources
        for (i, resources) in self.resources.as_array().iter().enumerate() {
            let total = 0;
            let available: VertexMap = roll & resources;

            for (player, buildings) in self.player_buildings {
                // idea: pack

                let first_pass: VertexMap = available & buildings;
                // TODO: make sure popcnt is emitted
                let amount = first_pass.count_ones() + (first_pass & self.cities).count_ones();
                bundles[player] = amount;
                total += amount;
                // roll_map &
            }
        }
        bundles
    }
}
