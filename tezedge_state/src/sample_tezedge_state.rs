// Copyright (c) SimpleStaking, Viable Systems and Tezedge Contributors
// SPDX-License-Identifier: MIT

use slog::Drain;
use std::convert::TryFrom;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::shell_compatibility_version::ShellCompatibilityVersion;
use crate::*;
use crypto::{
    crypto_box::{CryptoKey, PublicKey, SecretKey},
    hash::{ChainId, CryptoboxPublicKeyHash, HashTrait},
    proof_of_work::ProofOfWork,
};
use hex::FromHex;
use tezos_identity::Identity;

pub fn identity(pkh: &[u8], pk: &[u8], sk: &[u8], pow: &[u8]) -> Identity {
    Identity {
        peer_id: CryptoboxPublicKeyHash::try_from_bytes(pkh).unwrap(),
        public_key: PublicKey::from_bytes(pk).unwrap(),
        secret_key: SecretKey::from_bytes(sk).unwrap(),
        proof_of_work_stamp: ProofOfWork::from_hex(hex::encode(pow)).unwrap(),
    }
}

pub fn identity_1() -> Identity {
    identity(
        &[
            86, 205, 231, 178, 152, 146, 2, 157, 213, 131, 90, 117, 83, 132, 177, 84,
        ],
        &[
            148, 73, 141, 148, 22, 20, 15, 188, 69, 132, 149, 51, 61, 170, 193, 180, 200, 126, 65,
            159, 87, 38, 113, 122, 84, 249, 182, 198, 116, 118, 174, 28,
        ],
        &[
            172, 122, 207, 58, 254, 215, 99, 123, 225, 15, 143, 199, 106, 46, 182, 179, 53, 156,
            120, 173, 177, 216, 19, 180, 28, 186, 179, 250, 233, 84, 244, 177,
        ],
        &[
            187, 194, 48, 1, 73, 36, 158, 28, 204, 132, 165, 67, 98, 35, 108, 60, 187, 194, 204,
            47, 251, 211, 182, 234,
        ],
    )
}

pub fn logger(level: slog::Level) -> slog::Logger {
    let drain = Arc::new(
        slog_async::Async::new(
            slog_term::FullFormat::new(slog_term::TermDecorator::new().build())
                .build()
                .fuse(),
        )
        .chan_size(32768)
        .overflow_strategy(slog_async::OverflowStrategy::Block)
        .build(),
    );

    slog::Logger::root(drain.filter_level(level).fuse(), slog::o!())
}

pub fn default_shell_compatibility_version() -> ShellCompatibilityVersion {
    ShellCompatibilityVersion::new("TEZOS_MAINNET".to_owned(), vec![0], vec![0, 1])
}

pub fn default_effects() -> DefaultEffects {
    DefaultEffects::default()
}

pub fn default_config() -> TezedgeConfig {
    TezedgeConfig {
        port: 9732,
        disable_mempool: false,
        private_node: false,
        disable_quotas: false,
        disable_blacklist: false,
        min_connected_peers: 1,
        max_connected_peers: 100,
        max_pending_peers: 100,
        max_potential_peers: 1000,
        periodic_react_interval: Duration::from_millis(250),
        reset_quotas_interval: Duration::from_secs(5),
        peer_blacklist_duration: Duration::from_secs(15 * 60),
        peer_timeout: Duration::from_secs(8),
        // use high number to speed up identity generation.
        pow_target: 0.0,
    }
}

pub fn main_chain_id() -> ChainId {
    ChainId::try_from("NetXgtSLGNJvNye").unwrap()
}

pub fn build<'a, Efs>(
    initial_time: Instant,
    config: TezedgeConfig,
    initial_effects: &'a mut Efs,
) -> TezedgeState
where
    Efs: Effects,
{
    let node_identity = identity_1();

    let tezedge_state = TezedgeState::new(
        // slog::Logger::root(slog::Discard, slog::o!()),
        logger(slog::Level::Trace),
        config,
        node_identity,
        default_shell_compatibility_version(),
        initial_effects,
        initial_time,
        main_chain_id(),
    );

    tezedge_state
}
