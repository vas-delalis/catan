use std::sync::LazyLock;

use axum::{
    extract::Path,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use catan::*;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

static GAME: LazyLock<State> = LazyLock::new(State::default);

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(get_initial_observation))
        .route("/observation/{observer}", get(get_observation))
        .route("/action", post(apply_action))
        .layer(CorsLayer::new().allow_origin(Any));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:4000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn get_observation(Path(observer): Path<Player>) -> Json<Observation> {
    Json(GAME.observe(observer))
}

async fn get_initial_observation() -> Json<InitialObservation> {
    Json(InitialObservation {
        resources: vec![
            Ore, Wool, Lumber, Grain, Brick, Wool, Brick, Grain, Lumber, Ore, // <- Desert
            Lumber, Ore, Lumber, Ore, Grain, Wool, Brick, Grain, Wool,
        ],
        rolls: vec![
            10, 2, 9, 12, 6, 4, 10, 9, 11, 7, // <- Desert
            3, 8, 8, 3, 4, 5, 5, 6, 11,
        ],
        desert: 9 as HexId,
        robber: 9,
        settlements: vec![],
        roads: vec![],
    })
}

async fn apply_action(action: String) -> (StatusCode, Json<ActionResult>) {
    (StatusCode::OK, Json(ActionResult::Monopolized(Grain, 5)))
}

#[derive(Serialize)]
struct InitialObservation {
    resources: Vec<Resource>,
    rolls: Vec<u8>,
    desert: HexId,
    robber: HexId,
    settlements: Vec<(Player, VertexId)>,
    roads: Vec<(Player, EdgeId)>,
    // TODO: harbors
}
