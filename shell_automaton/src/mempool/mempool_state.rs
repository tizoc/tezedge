// Copyright (c) SimpleStaking, Viable Systems and Tezedge Contributors
// SPDX-License-Identifier: MIT

use std::{collections::{HashMap, HashSet}, net::SocketAddr};

use serde::{Serialize, Deserialize};

use crypto::hash::{OperationHash, ChainId};
use tezos_messages::p2p::{
    encoding::{block_header::BlockHeader, operation::Operation},
};

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct MempoolState {
    // the current head
    pub head_state: Option<HeadState>,
    // let's track what our peers know, and what we waiting from them
    pub peer_state: HashMap<SocketAddr, PeerState>,
    // operations that passed basic checks 
    pub pending_operations: HashMap<OperationHash, Operation>,
    // operations that passed all checks and classified
    // can be applied in the current context
    pub applied_operations: HashMap<OperationHash, Operation>,
    // cannot be included in the next head of the chain, but it could be included in a descendant
    pub branch_delayed_operations: HashMap<OperationHash, Operation>,
    // might be applied on a different branch if a reorganization happens
    pub branch_refused_operations: HashMap<OperationHash, Operation>,
    // let's memorize a hash of a bad operation and do not spend time checking it again
    pub refused_operations: HashSet<OperationHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HeadState {
    pub chain_id: ChainId,
    pub current_block: BlockHeader,
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct PeerState {
    // we received mempool from the peer and gonna send GetOperations
    pub requesting_full_content: HashSet<OperationHash>,
    // we sent GetOperations and pending full content of those operations
    pub pending_full_content: HashSet<OperationHash>,
    // those operations are known to the peer, should not rebroadcast
    pub known_operations: HashSet<OperationHash>,
}
