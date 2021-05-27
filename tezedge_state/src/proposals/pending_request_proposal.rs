use std::time::Instant;
use tla_sm::Proposal;

#[derive(Debug, Clone)]
pub enum PendingRequestMsg {
    DisconnectPeerPending,
    DisconnectPeerSuccess,

    BlacklistPeerPending,
    BlacklistPeerSuccess,
}

#[derive(Debug, Clone)]
pub struct PendingRequestProposal {
    pub at: Instant,
    pub req_id: usize,
    pub message: PendingRequestMsg,
}

impl Proposal for PendingRequestProposal {
    fn time(&self) -> Instant {
        self.at
    }
}