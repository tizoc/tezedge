use std::sync::Arc;
use std::time::SystemTime;

use crypto::hash::CryptoboxPublicKeyHash;

use tezos_messages::p2p::encoding::peer::PeerMessageResponse;
use tezos_messages::p2p::encoding::prelude::{MetadataMessage, NetworkVersion};
pub use tla_sm::{GetRequests, Proposal};

use crate::state::TezedgeState;
use crate::PeerAddress;

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum RequestState {
    Idle { at: SystemTime },
    Pending { at: SystemTime },
    Success { at: SystemTime },
    // Error { at: SystemTime },
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum RetriableRequestState {
    Idle { at: SystemTime },
    Pending { at: SystemTime },
    Success { at: SystemTime },
    Retry { at: SystemTime },
}

/// Requests for [tezedge_state::TezedgeProposer].
#[derive(Debug, Clone)]
pub enum TezedgeRequest {
    StartListeningForNewPeers {
        req_id: usize,
    },
    StopListeningForNewPeers {
        req_id: usize,
    },
    ConnectPeer {
        req_id: usize,
        peer: PeerAddress,
    },
    DisconnectPeer {
        req_id: usize,
        peer: PeerAddress,
    },
    BlacklistPeer {
        req_id: usize,
        peer: PeerAddress,
    },
    PeerMessageReceived {
        req_id: usize,
        peer: PeerAddress,
        message: Arc<PeerMessageResponse>,
    },
    NotifyHandshakeSuccessful {
        req_id: usize,
        peer_address: PeerAddress,
        peer_public_key_hash: CryptoboxPublicKeyHash,
        metadata: MetadataMessage,
        network_version: NetworkVersion,
    },
}

#[derive(Debug, Clone)]
pub enum PendingRequest {
    StartListeningForNewPeers,
    StopListeningForNewPeers,
    ConnectPeer {
        peer: PeerAddress,
    },
    DisconnectPeer {
        peer: PeerAddress,
    },
    BlacklistPeer {
        peer: PeerAddress,
    },

    // -- TODO: temporary until everything is handled inside TezedgeState.
    PeerMessageReceived {
        peer: PeerAddress,
        message: Arc<PeerMessageResponse>,
    },
    NotifyHandshakeSuccessful {
        peer_address: PeerAddress,
        peer_public_key_hash: CryptoboxPublicKeyHash,
        metadata: MetadataMessage,
        network_version: NetworkVersion,
    },
}

/// State for the currently pending request.
#[derive(Debug, Clone)]
pub struct PendingRequestState {
    pub request: PendingRequest,
    pub status: RetriableRequestState,
}

impl GetRequests for TezedgeState {
    type Request = TezedgeRequest;

    fn get_requests(&self, buf: &mut Vec<TezedgeRequest>) -> usize {
        use RetriableRequestState::*;

        let buf_initial_len = buf.len();

        if self.requests.is_empty() {
            return 0;
        }

        for (req_id, req) in self.requests.iter() {
            match &req.status {
                Idle { .. } => {}
                // retry if we are `at` the point to retry the request.
                Retry { at } if self.time >= *at => {}
                _ => continue,
            }

            buf.push(match &req.request {
                PendingRequest::StartListeningForNewPeers => {
                    TezedgeRequest::StartListeningForNewPeers { req_id }
                }
                PendingRequest::StopListeningForNewPeers => {
                    TezedgeRequest::StopListeningForNewPeers { req_id }
                }
                PendingRequest::ConnectPeer { peer } => {
                    let peer = peer.clone();
                    TezedgeRequest::ConnectPeer { req_id, peer }
                }
                PendingRequest::DisconnectPeer { peer } => {
                    let peer = peer.clone();
                    TezedgeRequest::DisconnectPeer { req_id, peer }
                }
                PendingRequest::BlacklistPeer { peer } => {
                    let peer = peer.clone();
                    TezedgeRequest::BlacklistPeer { req_id, peer }
                }
                PendingRequest::PeerMessageReceived { peer, message } => {
                    TezedgeRequest::PeerMessageReceived {
                        req_id,
                        peer: peer.clone(),
                        message: message.clone(),
                    }
                }
                PendingRequest::NotifyHandshakeSuccessful {
                    peer_address,
                    peer_public_key_hash,
                    metadata,
                    network_version,
                } => TezedgeRequest::NotifyHandshakeSuccessful {
                    req_id,
                    peer_address: peer_address.clone(),
                    peer_public_key_hash: peer_public_key_hash.clone(),
                    metadata: metadata.clone(),
                    network_version: network_version.clone(),
                },
            })
        }

        buf.len() - buf_initial_len
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};
    use tla_sm::{Acceptor, GetRequests};

    use super::*;
    use crate::{
        proposals::{PendingRequestMsg, PendingRequestProposal, TickProposal},
        sample_tezedge_state,
    };

    fn get_requests(state: &mut TezedgeState) -> Vec<TezedgeRequest> {
        let mut requests = vec![];
        state.get_requests(&mut requests);
        requests
    }

    #[test]
    fn test_start_listening_for_new_peers_request_retries() {
        let mut effects = sample_tezedge_state::default_effects();
        let initial_time = SystemTime::now();

        let mut state = sample_tezedge_state::build(
            initial_time,
            sample_tezedge_state::default_config(),
            &mut effects,
        );

        let mut req_id = get_requests(&mut state)
            .into_iter()
            .filter_map(|x| match x {
                TezedgeRequest::StartListeningForNewPeers { req_id } => Some(req_id),
                _ => None,
            })
            .last()
            .unwrap();

        state.accept(PendingRequestProposal {
            effects: &mut effects,
            time_passed: Duration::new(0, 0),
            req_id,
            message: PendingRequestMsg::StartListeningForNewPeersError {
                error: std::io::ErrorKind::AddrInUse,
            },
        });

        match state.requests[req_id].status {
            RetriableRequestState::Retry { at } => {
                assert_eq!(initial_time + state.config.periodic_react_interval, at);
            }
            _ => panic!(),
        };
        assert!(get_requests(&mut state)
            .into_iter()
            .filter_map(|x| {
                match x {
                    TezedgeRequest::StartListeningForNewPeers { req_id } => Some(req_id),
                    _ => None,
                }
            })
            .last()
            .is_none());

        state.accept(TickProposal {
            effects: &mut effects,
            time_passed: state.config.periodic_react_interval,
        });

        assert!(get_requests(&mut state)
            .into_iter()
            .filter_map(|x| {
                match x {
                    TezedgeRequest::StartListeningForNewPeers { req_id } => Some(req_id),
                    _ => None,
                }
            })
            .last()
            .is_some());

        state.accept(PendingRequestProposal {
            effects: &mut effects,
            time_passed: Duration::new(0, 0),
            req_id,
            message: PendingRequestMsg::StartListeningForNewPeersSuccess,
        });

        assert!(get_requests(&mut state)
            .into_iter()
            .filter_map(|x| {
                match x {
                    TezedgeRequest::StartListeningForNewPeers { req_id } => Some(req_id),
                    _ => None,
                }
            })
            .last()
            .is_none());
    }
}
