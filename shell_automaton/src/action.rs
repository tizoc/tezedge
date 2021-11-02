// Copyright (c) SimpleStaking, Viable Systems and Tezedge Contributors
// SPDX-License-Identifier: MIT

use derive_more::From;
use enum_kinds::EnumKind;
use serde::{Deserialize, Serialize};
use storage::persistent::SchemaError;

use crate::event::{P2pPeerEvent, P2pServerEvent, WakeupEvent};

use crate::paused_loops::{
    PausedLoopsAddAction, PausedLoopsResumeAllAction, PausedLoopsResumeNextInitAction,
    PausedLoopsResumeNextSuccessAction,
};

use crate::peer::binary_message::read::*;
use crate::peer::binary_message::write::*;
use crate::peer::chunk::read::*;
use crate::peer::chunk::write::*;
use crate::peer::message::read::*;
use crate::peer::message::write::*;
use crate::peer::{
    PeerTryReadLoopFinishAction, PeerTryReadLoopStartAction, PeerTryWriteLoopFinishAction,
    PeerTryWriteLoopStartAction,
};

use crate::peer::connection::closed::PeerConnectionClosedAction;
use crate::peer::connection::incoming::accept::*;
use crate::peer::connection::incoming::{
    PeerConnectionIncomingErrorAction, PeerConnectionIncomingSuccessAction,
};
use crate::peer::connection::outgoing::{
    PeerConnectionOutgoingErrorAction, PeerConnectionOutgoingInitAction,
    PeerConnectionOutgoingPendingAction, PeerConnectionOutgoingRandomInitAction,
    PeerConnectionOutgoingSuccessAction,
};
use crate::peer::disconnection::{PeerDisconnectAction, PeerDisconnectedAction};

use crate::peer::handshaking::*;

use crate::peers::add::multi::PeersAddMultiAction;
use crate::peers::add::PeersAddIncomingPeerAction;
use crate::peers::check::timeouts::{
    PeersCheckTimeoutsCleanupAction, PeersCheckTimeoutsInitAction, PeersCheckTimeoutsSuccessAction,
};
use crate::peers::dns_lookup::{
    PeersDnsLookupCleanupAction, PeersDnsLookupErrorAction, PeersDnsLookupInitAction,
    PeersDnsLookupSuccessAction,
};
use crate::peers::graylist::{
    PeersGraylistAddressAction, PeersGraylistIpAddAction, PeersGraylistIpAddedAction,
    PeersGraylistIpRemoveAction, PeersGraylistIpRemovedAction,
};
use crate::peers::remove::PeersRemoveAction;
use crate::mempool::{
    MempoolRecvDoneAction, MempoolGetOperationsAction, MempoolGetOperationsPendingAction,
    MempoolOperationRecvDoneAction, MempoolBroadcastAction, MempoolBroadcastDoneAction,
    MempoolOperationInjectDoneAction,
};

use crate::storage::request::{
    StorageRequestCreateAction, StorageRequestErrorAction, StorageRequestFinishAction,
    StorageRequestInitAction, StorageRequestPendingAction, StorageRequestSuccessAction,
    StorageResponseReceivedAction,
};
use crate::storage::state_snapshot::create::{
    StorageStateSnapshotCreateErrorAction, StorageStateSnapshotCreateInitAction,
    StorageStateSnapshotCreatePendingAction, StorageStateSnapshotCreateSuccessAction,
};

pub use redux_rs::{ActionId, ActionWithId};

#[derive(
    EnumKind,
    strum_macros::AsRefStr,
    strum_macros::IntoStaticStr,
    From,
    Serialize,
    Deserialize,
    Debug,
    Clone,
)]
#[enum_kind(
    ActionKind,
    derive(
        strum_macros::EnumIter,
        strum_macros::Display,
        Serialize,
        Deserialize,
        Hash
    )
)]
#[serde(tag = "kind", content = "content")]
pub enum Action {
    Init,

    PausedLoopsAdd(PausedLoopsAddAction),
    PausedLoopsResumeAll(PausedLoopsResumeAllAction),
    PausedLoopsResumeNextInit(PausedLoopsResumeNextInitAction),
    PausedLoopsResumeNextSuccess(PausedLoopsResumeNextSuccessAction),

    PeersDnsLookupInit(PeersDnsLookupInitAction),
    PeersDnsLookupError(PeersDnsLookupErrorAction),
    PeersDnsLookupSuccess(PeersDnsLookupSuccessAction),
    PeersDnsLookupCleanup(PeersDnsLookupCleanupAction),

    PeersGraylistAddress(PeersGraylistAddressAction),
    PeersGraylistIpAdd(PeersGraylistIpAddAction),
    PeersGraylistIpAdded(PeersGraylistIpAddedAction),
    PeersGraylistIpRemove(PeersGraylistIpRemoveAction),
    PeersGraylistIpRemoved(PeersGraylistIpRemovedAction),

    PeersAddIncomingPeer(PeersAddIncomingPeerAction),
    PeersAddMulti(PeersAddMultiAction),
    PeersRemove(PeersRemoveAction),

    PeersCheckTimeoutsInit(PeersCheckTimeoutsInitAction),
    PeersCheckTimeoutsSuccess(PeersCheckTimeoutsSuccessAction),
    PeersCheckTimeoutsCleanup(PeersCheckTimeoutsCleanupAction),

    PeerConnectionIncomingAccept(PeerConnectionIncomingAcceptAction),
    PeerConnectionIncomingAcceptError(PeerConnectionIncomingAcceptErrorAction),
    PeerConnectionIncomingRejected(PeerConnectionIncomingRejectedAction),
    PeerConnectionIncomingAcceptSuccess(PeerConnectionIncomingAcceptSuccessAction),

    PeerConnectionIncomingError(PeerConnectionIncomingErrorAction),
    PeerConnectionIncomingSuccess(PeerConnectionIncomingSuccessAction),

    PeerConnectionOutgoingRandomInit(PeerConnectionOutgoingRandomInitAction),
    PeerConnectionOutgoingInit(PeerConnectionOutgoingInitAction),
    PeerConnectionOutgoingPending(PeerConnectionOutgoingPendingAction),
    PeerConnectionOutgoingError(PeerConnectionOutgoingErrorAction),
    PeerConnectionOutgoingSuccess(PeerConnectionOutgoingSuccessAction),

    PeerConnectionClosed(PeerConnectionClosedAction),

    PeerDisconnect(PeerDisconnectAction),
    PeerDisconnected(PeerDisconnectedAction),

    MioWaitForEvents,
    MioTimeoutEvent,
    P2pServerEvent(P2pServerEvent),
    P2pPeerEvent(P2pPeerEvent),
    WakeupEvent(WakeupEvent),

    PeerTryWriteLoopStart(PeerTryWriteLoopStartAction),
    PeerTryWriteLoopFinish(PeerTryWriteLoopFinishAction),
    PeerTryReadLoopStart(PeerTryReadLoopStartAction),
    PeerTryReadLoopFinish(PeerTryReadLoopFinishAction),

    // chunk read
    PeerChunkReadInit(PeerChunkReadInitAction),
    PeerChunkReadPart(PeerChunkReadPartAction),
    PeerChunkReadDecrypt(PeerChunkReadDecryptAction),
    PeerChunkReadReady(PeerChunkReadReadyAction),
    PeerChunkReadError(PeerChunkReadErrorAction),

    // chunk write
    PeerChunkWriteSetContent(PeerChunkWriteSetContentAction),
    PeerChunkWriteEncryptContent(PeerChunkWriteEncryptContentAction),
    PeerChunkWriteCreateChunk(PeerChunkWriteCreateChunkAction),
    PeerChunkWritePart(PeerChunkWritePartAction),
    PeerChunkWriteReady(PeerChunkWriteReadyAction),
    PeerChunkWriteError(PeerChunkWriteErrorAction),

    // binary message read
    PeerBinaryMessageReadInit(PeerBinaryMessageReadInitAction),
    PeerBinaryMessageReadChunkReady(PeerBinaryMessageReadChunkReadyAction),
    PeerBinaryMessageReadSizeReady(PeerBinaryMessageReadSizeReadyAction),
    PeerBinaryMessageReadReady(PeerBinaryMessageReadReadyAction),
    PeerBinaryMessageReadError(PeerBinaryMessageReadErrorAction),

    // binary message write
    PeerBinaryMessageWriteSetContent(PeerBinaryMessageWriteSetContentAction),
    PeerBinaryMessageWriteNextChunk(PeerBinaryMessageWriteNextChunkAction),
    PeerBinaryMessageWriteReady(PeerBinaryMessageWriteReadyAction),
    PeerBinaryMessageWriteError(PeerBinaryMessageWriteErrorAction),

    PeerMessageReadInit(PeerMessageReadInitAction),
    PeerMessageReadError(PeerMessageReadErrorAction),
    PeerMessageReadSuccess(PeerMessageReadSuccessAction),

    PeerMessageWriteNext(PeerMessageWriteNextAction),
    PeerMessageWriteInit(PeerMessageWriteInitAction),
    PeerMessageWriteError(PeerMessageWriteErrorAction),
    PeerMessageWriteSuccess(PeerMessageWriteSuccessAction),

    PeerHandshakingInit(PeerHandshakingInitAction),
    PeerHandshakingConnectionMessageInit(PeerHandshakingConnectionMessageInitAction),
    PeerHandshakingConnectionMessageEncode(PeerHandshakingConnectionMessageEncodeAction),
    PeerHandshakingConnectionMessageWrite(PeerHandshakingConnectionMessageWriteAction),
    PeerHandshakingConnectionMessageRead(PeerHandshakingConnectionMessageReadAction),
    PeerHandshakingConnectionMessageDecode(PeerHandshakingConnectionMessageDecodeAction),

    PeerHandshakingEncryptionInit(PeerHandshakingEncryptionInitAction),

    PeerHandshakingMetadataMessageInit(PeerHandshakingMetadataMessageInitAction),
    PeerHandshakingMetadataMessageEncode(PeerHandshakingMetadataMessageEncodeAction),
    PeerHandshakingMetadataMessageWrite(PeerHandshakingMetadataMessageWriteAction),
    PeerHandshakingMetadataMessageRead(PeerHandshakingMetadataMessageReadAction),
    PeerHandshakingMetadataMessageDecode(PeerHandshakingMetadataMessageDecodeAction),

    PeerHandshakingAckMessageInit(PeerHandshakingAckMessageInitAction),
    PeerHandshakingAckMessageEncode(PeerHandshakingAckMessageEncodeAction),
    PeerHandshakingAckMessageWrite(PeerHandshakingAckMessageWriteAction),
    PeerHandshakingAckMessageRead(PeerHandshakingAckMessageReadAction),
    PeerHandshakingAckMessageDecode(PeerHandshakingAckMessageDecodeAction),

    PeerHandshakingError(PeerHandshakingErrorAction),
    PeerHandshakingFinish(PeerHandshakingFinishAction),

    MempoolRecvDone(MempoolRecvDoneAction),
    MempoolGetOperations(MempoolGetOperationsAction),
    MempoolGetOperationsPending(MempoolGetOperationsPendingAction),
    MempoolOperationRecvDone(MempoolOperationRecvDoneAction),
    MempoolOperationInjectDone(MempoolOperationInjectDoneAction),
    MempoolBroadcast(MempoolBroadcastAction),
    MempoolBroadcastDone(MempoolBroadcastDoneAction),

    StorageRequestCreate(StorageRequestCreateAction),
    StorageRequestInit(StorageRequestInitAction),
    StorageRequestPending(StorageRequestPendingAction),
    StorageResponseReceived(StorageResponseReceivedAction),
    StorageRequestError(StorageRequestErrorAction),
    StorageRequestSuccess(StorageRequestSuccessAction),
    StorageRequestFinish(StorageRequestFinishAction),

    StorageStateSnapshotCreateInit(StorageStateSnapshotCreateInitAction),
    StorageStateSnapshotCreatePending(StorageStateSnapshotCreatePendingAction),
    StorageStateSnapshotCreateError(StorageStateSnapshotCreateErrorAction),
    StorageStateSnapshotCreateSuccess(StorageStateSnapshotCreateSuccessAction),
}

impl Action {
    #[inline(always)]
    pub fn kind(&self) -> ActionKind {
        ActionKind::from(self)
    }
}

// bincode decoding fails with: "Bincode does not support Deserializer::deserialize_identifier".
// So use json instead, which works.

// impl BincodeEncoded for Action {
//     fn decode(bytes: &[u8]) -> Result<Self, storage::persistent::SchemaError> {
//         // here it errors.
//         Ok(dbg!(bincode::deserialize(bytes)).unwrap())
//     }

//     fn encode(&self) -> Result<Vec<u8>, storage::persistent::SchemaError> {
//         Ok(bincode::serialize::<Self>(self).unwrap())
//     }
// }

impl storage::persistent::Encoder for Action {
    fn encode(&self) -> Result<Vec<u8>, SchemaError> {
        serde_json::to_vec(self).map_err(|_| SchemaError::EncodeError)
    }
}

impl storage::persistent::Decoder for Action {
    fn decode(bytes: &[u8]) -> Result<Self, SchemaError> {
        serde_json::from_slice(bytes).map_err(|_| SchemaError::DecodeError)
    }
}

impl<'a> From<&'a ActionWithId<Action>> for ActionKind {
    fn from(action: &'a ActionWithId<Action>) -> ActionKind {
        action.action.kind()
    }
}

impl From<ActionWithId<Action>> for ActionKind {
    fn from(action: ActionWithId<Action>) -> ActionKind {
        action.action.kind()
    }
}
