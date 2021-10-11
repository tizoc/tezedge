use std::sync::Arc;
use tokio::sync::mpsc;

pub type RequestTrySendError<T> = mpsc::error::TrySendError<T>;
pub type ResponseSendError<T> = mpsc::error::SendError<T>;

pub type ResponseTryRecvError = mpsc::error::TryRecvError;

/// Error while trying to receive next request from the requester.
///
/// Can only happen if requester is disconnected/shut down.
#[derive(Debug)]
pub struct RequestRecvError;

impl From<mpsc::error::RecvError> for RequestRecvError {
    fn from(_: mpsc::error::RecvError) -> Self {
        Self {}
    }
}

/// Requester half of the channel.
///
/// It is used to send requests to the worker.
#[derive(Debug)]
pub struct ServiceWorkerAsyncRequester<Req, Resp> {
    sender: mpsc::Sender<Req>,
    receiver: mpsc::Receiver<Resp>,
}

impl<Req, Resp> ServiceWorkerAsyncRequester<Req, Resp> {
    pub fn try_send(&self, req: Req) -> Result<(), RequestTrySendError<Req>> {
        Ok(self.sender.try_send(req)?)
    }

    pub fn try_recv(&mut self) -> Result<Resp, ResponseTryRecvError> {
        Ok(self.receiver.try_recv()?)
    }
}

#[inline(always)]
async fn responder_send<T>(
    sender: &mpsc::Sender<T>,
    mio_waker: &Arc<mio::Waker>,
    msg: T,
) -> Result<(), ResponseSendError<T>> {
    let _ = mio_waker.wake();
    Ok(sender.send(msg).await?)
}

pub struct ServiceWorkerAsyncResponderSender<Resp> {
    sender: mpsc::Sender<Resp>,
    mio_waker: Arc<mio::Waker>,
}

impl<Resp> ServiceWorkerAsyncResponderSender<Resp> {
    pub async fn send(&self, resp: Resp) -> Result<(), ResponseSendError<Resp>> {
        responder_send(&self.sender, &self.mio_waker, resp).await
    }
}

impl<Resp> Clone for ServiceWorkerAsyncResponderSender<Resp> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            mio_waker: self.mio_waker.clone(),
        }
    }
}

/// Responder half of the channel.
///
/// It is used by worker to send responses to the requester.
pub struct ServiceWorkerAsyncResponder<Req, Resp> {
    sender: mpsc::Sender<Resp>,
    receiver: mpsc::Receiver<Req>,
    mio_waker: Arc<mio::Waker>,
}

impl<Req, Resp> ServiceWorkerAsyncResponder<Req, Resp> {
    pub fn sender(&self) -> ServiceWorkerAsyncResponderSender<Resp> {
        ServiceWorkerAsyncResponderSender {
            sender: self.sender.clone(),
            mio_waker: self.mio_waker.clone(),
        }
    }

    pub async fn send(&self, resp: Resp) -> Result<(), ResponseSendError<Resp>> {
        responder_send(&self.sender, &self.mio_waker, resp).await
    }

    pub async fn recv(&mut self) -> Result<Req, RequestRecvError> {
        self.receiver.recv().await.ok_or(RequestRecvError {})
    }
}

pub fn worker_channel<Req, Resp>(
    mio_waker: Arc<mio::Waker>,
    bound: usize,
) -> (
    ServiceWorkerAsyncRequester<Req, Resp>,
    ServiceWorkerAsyncResponder<Req, Resp>,
) {
    let (requester_tx, responder_rx) = mpsc::channel(bound);
    let (responder_tx, requester_rx) = mpsc::channel(bound);

    (
        ServiceWorkerAsyncRequester {
            sender: requester_tx,
            receiver: requester_rx,
        },
        ServiceWorkerAsyncResponder {
            sender: responder_tx,
            receiver: responder_rx,
            mio_waker,
        },
    )
}