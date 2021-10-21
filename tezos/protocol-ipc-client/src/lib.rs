// Copyright (c) SimpleStaking, Viable Systems and Tezedge Contributors
// SPDX-License-Identifier: MIT

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use async_ipc::{IpcError, IpcReceiver, IpcSender};
use crypto::hash::{ChainId, ContextHash, ProtocolHash};
use nix::{
    sys::signal::{self, Signal},
    unistd::Pid,
};
use slog::{info, warn, Level, Logger};
use tezos_messages::p2p::encoding::operation::Operation;
use tezos_protocol_ipc_messages::*;
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{Child, Command},
    sync::RwLock,
    time::Instant,
};

use tezos_api::{environment::TezosEnvironmentConfiguration, ffi::*};
use tezos_context_api::{
    ContextKeyOwned, ContextValue, PatchContext, StringTreeObject, TezosContextStorageConfiguration,
};

/// Errors generated by `protocol_runner`.
#[derive(Error, Debug)]
pub enum ProtocolRunnerError {
    #[error("Failed to spawn tezos protocol wrapper sub-process: {reason}")]
    SpawnError {
        #[from]
        reason: tokio::io::Error,
    },
    #[error("Timeout when waiting for protocol runner connection socket")]
    SocketTimeout,
    #[error("Failed to terminate/kill tezos protocol wrapper sub-process, reason: {reason}")]
    TerminateError { reason: String },
}

impl slog::Value for ProtocolRunnerError {
    fn serialize(
        &self,
        _record: &slog::Record,
        key: slog::Key,
        serializer: &mut dyn slog::Serializer,
    ) -> slog::Result {
        serializer.emit_arguments(key, &format_args!("{}", self))
    }
}

/// Protocol configuration (transferred via IPC from tezedge node to protocol_runner.
#[derive(Clone, Debug)]
pub struct ProtocolRunnerConfiguration {
    pub runtime_configuration: TezosRuntimeConfiguration,
    pub environment: TezosEnvironmentConfiguration,
    pub enable_testchain: bool,
    pub storage: TezosContextStorageConfiguration,
    pub executable_path: PathBuf,
    pub log_level: Level,
}

impl ProtocolRunnerConfiguration {
    pub fn new(
        runtime_configuration: TezosRuntimeConfiguration,
        environment: TezosEnvironmentConfiguration,
        enable_testchain: bool,
        storage: TezosContextStorageConfiguration,
        executable_path: PathBuf,
        log_level: Level,
    ) -> Self {
        Self {
            runtime_configuration,
            environment,
            enable_testchain,
            storage,
            executable_path,
            log_level,
        }
    }
}

// TODO: differentiate between writable and readonly?
pub struct ProtocolRunnerInstance {
    configuration: ProtocolRunnerConfiguration,
    socket_path: PathBuf,
    endpoint_name: String,
    tokio_runtime: tokio::runtime::Handle,
    child_process_handle: Option<Child>,
    log: Logger,
}

impl Drop for ProtocolRunnerInstance {
    fn drop(&mut self) {
        self.shutdown()
    }
}

impl ProtocolRunnerInstance {
    pub const PROCESS_TERMINATE_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

    pub fn without_spawn(
        configuration: ProtocolRunnerConfiguration,
        socket_path: &Path,
        endpoint_name: String,
        tokio_runtime: &tokio::runtime::Handle,
        log: Logger,
    ) -> Result<Self, ProtocolRunnerError> {
        Ok(Self {
            configuration,
            socket_path: socket_path.to_path_buf(),
            endpoint_name,
            tokio_runtime: tokio_runtime.clone(),
            child_process_handle: None,
            log,
        })
    }

    pub fn spawn(
        configuration: ProtocolRunnerConfiguration,
        socket_path: &Path,
        endpoint_name: String,
        tokio_runtime: &tokio::runtime::Handle,
        log: Logger,
    ) -> Result<Self, ProtocolRunnerError> {
        let ProtocolRunnerConfiguration {
            executable_path,
            log_level,
            ..
        } = &configuration;
        let child_process_handle = Some(Self::spawn_process(
            executable_path,
            socket_path,
            &endpoint_name,
            log_level,
            log.clone(),
            tokio_runtime,
        )?);

        // TODO: duplicating stuff from configuration
        Ok(Self {
            configuration,
            socket_path: socket_path.to_path_buf(),
            endpoint_name,
            tokio_runtime: tokio_runtime.clone(),
            child_process_handle,
            log,
        })
    }

    pub async fn writable_connection(&self) -> Result<ProtocolRunnerConnection, IpcError> {
        // TODO: pool connections
        let ipc_client = async_ipc::IpcClient::new(&self.socket_path);
        let (rx, tx) = ipc_client.connect().await?;
        let io = IpcIO { rx, tx };

        Ok(ProtocolRunnerConnection {
            configuration: self.configuration.clone(),
            io,
        })
    }

    pub async fn wait_for_socket(
        &self,
        timeout: Option<Duration>,
    ) -> Result<(), ProtocolRunnerError> {
        let start = Instant::now();
        let timeout = timeout.unwrap_or(Duration::from_secs(3));

        loop {
            if self.socket_path.exists() {
                break;
            }

            if start.elapsed() > timeout {
                return Err(ProtocolRunnerError::SocketTimeout);
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(())
    }

    fn spawn_process(
        executable_path: &PathBuf,
        socket_path: &Path,
        endpoint_name: &str,
        log_level: &Level,
        log: Logger,
        tokio_runtime: &tokio::runtime::Handle,
    ) -> Result<tokio::process::Child, ProtocolRunnerError> {
        let _guard = tokio_runtime.enter();
        let mut process = Command::new(executable_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .arg("--socket-path")
            .arg(socket_path)
            .arg("--endpoint")
            .arg(endpoint_name)
            .arg("--log-level")
            .arg(log_level.as_str().to_lowercase())
            .spawn()?;

        Self::log_subprocess_output(tokio_runtime, &mut process, log.clone());

        Ok(process)
    }

    /// Give [`wait_timeout`] time to stop process, and after that if tries to terminate/kill it
    pub async fn wait_and_terminate_ref(
        process: &mut tokio::process::Child,
        wait_timeout: Duration,
        log: &Logger,
    ) -> Result<(), ProtocolRunnerError> {
        match tokio::time::timeout(wait_timeout, process.wait()).await {
            Ok(Ok(exit_status)) => {
                if exit_status.success() {
                    info!(log, "Exited successfuly");
                } else {
                    warn!(log, "Exited with status code: {}", exit_status);
                }
                Ok(())
            }
            Ok(Err(err)) => Self::terminate_or_kill(process, format!("{:?}", err)).await,
            Err(_) => Self::terminate_or_kill(process, "wait timeout exceeded".to_string()).await,
        }
    }

    /// Checks if process is running
    pub fn is_running(process: &mut tokio::process::Child) -> bool {
        matches!(process.try_wait(), Ok(None))
    }

    /// Logs exit status
    pub fn log_exit_status(process: &mut tokio::process::Child, log: &Logger) {
        match process.try_wait() {
            Ok(None) => (),
            Ok(Some(status)) => {
                if status.success() {
                    info!(log, "protocol-runner was closed normally");
                } else {
                    warn!(log, "protocol-runner exited with status code: {}", status);
                }
            }
            Err(err) => warn!(
                log,
                "failed to obtain protocol-runner exit status code: {:?}", err
            ),
        }
    }

    pub fn shutdown(&mut self) {
        let mut child_process_handle =
            if let Some(child) = std::mem::take(&mut self.child_process_handle) {
                child
            } else {
                return;
            };

        info!(
            self.log,
            "Shutting down protocol runner: {}", self.endpoint_name
        );
        if let Err(err) = tokio::task::block_in_place(|| {
            let tokio_runtime = self.tokio_runtime.clone();
            tokio_runtime.block_on(async {
                if let Ok(mut conn) = self.writable_connection().await {
                    if let Err(err) = conn.shutdown().await {
                        warn!(
                            self.log,
                            "Failed to send shutdown message to protocol runner: {}", err
                        );
                    }
                }
                Self::wait_and_terminate_ref(
                    &mut child_process_handle,
                    Duration::from_secs(3),
                    &self.log,
                )
                .await
            })
        }) {
            warn!(self.log, "Failed to stop protocol runner: {}", err);
        }
        Self::log_exit_status(&mut child_process_handle, &self.log);
    }

    async fn terminate_or_kill(
        process: &mut Child,
        reason: String,
    ) -> Result<(), ProtocolRunnerError> {
        // try to send SIGINT (ctrl-c)
        if let Some(pid) = process.id() {
            let pid = Pid::from_raw(pid as i32);
            match signal::kill(pid, Signal::SIGINT) {
                Ok(_) => Ok(()),
                Err(sigint_error) => {
                    // (fallback) if SIGINT failed, we just kill process
                    match process.kill().await {
                        Ok(_) => Ok(()),
                        Err(kill_error) => Err(ProtocolRunnerError::TerminateError {
                            reason: format!(
                                "Reason for termination: {}, sigint_error: {}, kill_error: {}",
                                reason, sigint_error, kill_error
                            ),
                        }),
                    }
                }
            }
        } else {
            Ok(())
        }
    }

    /// Spawns a tokio task that will forward STDOUT and STDERR from the child
    /// process to slog's output
    fn log_subprocess_output(
        tokio_runtime: &tokio::runtime::Handle,
        process: &mut Child,
        log: Logger,
    ) {
        // Only launch logging task if the output port if present, otherwise log a warning.
        macro_rules! handle_output {
            ($tag:expr, $name:expr, $io:expr, $log:expr) => {{
                if let Some(out) = $io.take() {
                    let log = $log;
                    tokio_runtime.spawn(async move {
                        let reader = BufReader::new(out);
                        let mut lines = reader.lines();
                        loop {
                            match lines.next_line().await {
                                Ok(Some(line)) => info!(log, "[{}] {}", $tag, line),
                                Ok(None) => {
                                    info!(log, "[{}] {} closed.", $tag, $name);
                                    break;
                                }
                                Err(err) => {
                                    warn!(log, "[{}] {} closed with error: {:?}", $tag, $name, err);
                                    break;
                                }
                            }
                        }
                    });
                } else {
                    warn!(
                        log,
                        "Expected child process to have {}, but it was None", $name
                    );
                };
            }};
        }

        handle_output!("OCaml-out", "STDOUT", process.stdout, log.clone());
        handle_output!("OCaml-err", "STDERR", process.stderr, log.clone());
    }
}

struct IpcIO {
    rx: IpcReceiver<NodeMessage>,
    tx: IpcSender<ProtocolMessage>,
}

impl IpcIO {
    pub async fn send(&mut self, value: &ProtocolMessage) -> Result<(), async_ipc::IpcError> {
        self.tx.send(value).await?;
        Ok(())
    }

    pub async fn try_receive(
        &mut self,
        read_timeout: Option<Duration>,
    ) -> Result<NodeMessage, async_ipc::IpcError> {
        if let Some(read_timeout) = read_timeout {
            Ok(self.rx.try_receive(read_timeout).await?)
        } else {
            self.rx.receive().await
        }
    }
}

pub struct ProtocolRunnerApi {
    pub(crate) writable_instance: RwLock<ProtocolRunnerInstance>,
    pub(crate) tokio_runtime: tokio::runtime::Handle,
}

pub struct ProtocolRunnerApiShutdownGuard {
    api: Arc<ProtocolRunnerApi>,
}

impl Drop for ProtocolRunnerApiShutdownGuard {
    fn drop(&mut self) {
        tokio::task::block_in_place(|| self.api.tokio_runtime.block_on(self.api.shutdown()))
    }
}

impl ProtocolRunnerApi {
    pub fn new(
        writable_instance: ProtocolRunnerInstance,
        tokio_runtime: &tokio::runtime::Handle,
    ) -> Self {
        Self {
            writable_instance: RwLock::new(writable_instance),
            tokio_runtime: tokio_runtime.clone(),
        }
    }

    pub async fn shutdown(&self) {
        self.writable_instance.write().await.shutdown();
    }

    pub fn shutdown_on_drop(self: &Arc<Self>) -> ProtocolRunnerApiShutdownGuard {
        ProtocolRunnerApiShutdownGuard {
            api: Arc::clone(&self),
        }
    }

    pub async fn writable_connection(&self) -> Result<ProtocolRunnerConnection, IpcError> {
        // TODO: pool connections
        self.writable_instance
            .read()
            .await
            .writable_connection()
            .await
    }

    pub async fn readable_connection(&self) -> Result<ProtocolRunnerConnection, IpcError> {
        // TODO: reimplement once readonly instances have been added
        self.writable_connection().await
    }
}

pub struct ProtocolRunnerConnection {
    configuration: ProtocolRunnerConfiguration,
    io: IpcIO,
}

macro_rules! handle_request {
    ($io:expr, $msg:ident $(($($arg:ident),+))?, $resp:ident($result:ident), $error:ident, $timeout:expr $(,)?) => {{
        $io.send(&ProtocolMessage::$msg $(($($arg),+))?).await?;

        match $io.try_receive($timeout).await? {
            NodeMessage::$resp($result) => {
                $result.map_err(|err| ProtocolError::$error { reason: err }.into())
            }
            message => Err(ProtocolServiceError::UnexpectedMessage {
                message: message.into(),
            }),
        }
    }};

    ($io:expr, $msg:ident $(($($arg:ident),+))?, $resp:ident $(($result:ident))? => $result_expr:expr, $timeout:expr $(,)?) => {{
        $io.send(&ProtocolMessage::$msg $(($($arg),+))?).await?;

        match $io.try_receive($timeout).await? {
            NodeMessage::$resp $(($result))? => $result_expr,
            message => Err(ProtocolServiceError::UnexpectedMessage {
                message: message.into(),
            }),
        }
    }};
}

impl ProtocolRunnerConnection {
    const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
    const DEFAULT_TIMEOUT_LONG: Duration = Duration::from_secs(300);
    const APPLY_BLOCK_TIMEOUT: Duration = Duration::from_secs(60 * 60 * 2);
    const INIT_PROTOCOL_CONTEXT_TIMEOUT: Duration = Duration::from_secs(60);
    const BEGIN_APPLICATION_TIMEOUT: Duration = Duration::from_secs(120);
    const BEGIN_CONSTRUCTION_TIMEOUT: Duration = Duration::from_secs(120);
    const VALIDATE_OPERATION_TIMEOUT: Duration = Duration::from_secs(120);
    const CALL_PROTOCOL_RPC_TIMEOUT: Duration = Duration::from_secs(30);
    const CALL_PROTOCOL_HEAVY_RPC_TIMEOUT: Duration = Duration::from_secs(600);
    const COMPUTE_PATH_TIMEOUT: Duration = Duration::from_secs(30);
    const JSON_ENCODE_DATA_TIMEOUT: Duration = Duration::from_secs(30);
    const ASSERT_ENCODING_FOR_PROTOCOL_DATA_TIMEOUT: Duration = Duration::from_secs(15);

    /// Apply block
    pub async fn apply_block(
        &mut self,
        request: ApplyBlockRequest,
    ) -> Result<ApplyBlockResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            ApplyBlockCall(request),
            ApplyBlockResult(result),
            ApplyBlockError,
            Some(Self::APPLY_BLOCK_TIMEOUT),
        )
    }

    pub async fn assert_encoding_for_protocol_data(
        &mut self,
        protocol_hash: ProtocolHash,
        protocol_data: RustBytes,
    ) -> Result<(), ProtocolServiceError> {
        handle_request!(
            self.io,
            AssertEncodingForProtocolDataCall(protocol_hash, protocol_data),
            AssertEncodingForProtocolDataResult(result),
            AssertEncodingForProtocolDataError,
            Some(Self::ASSERT_ENCODING_FOR_PROTOCOL_DATA_TIMEOUT),
        )
    }

    /// Begin application
    pub async fn begin_application(
        &mut self,
        request: BeginApplicationRequest,
    ) -> Result<BeginApplicationResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            BeginApplicationCall(request),
            BeginApplicationResult(result),
            BeginApplicationError,
            Some(Self::BEGIN_APPLICATION_TIMEOUT),
        )
    }

    /// Begin construction (for prevalidation, doesn't accumulate)
    pub async fn begin_construction_for_prevalidation(
        &mut self,
        request: BeginConstructionRequest,
    ) -> Result<PrevalidatorWrapper, ProtocolServiceError> {
        handle_request!(
            self.io,
            BeginConstructionForPrevalidationCall(request),
            BeginConstructionResult(result),
            BeginConstructionError,
            Some(Self::BEGIN_CONSTRUCTION_TIMEOUT),
        )
    }

    /// Validate operation (for prevalidation, doesn't accumulate)
    pub async fn validate_operation_for_prevalidation(
        &mut self,
        request: ValidateOperationRequest,
    ) -> Result<ValidateOperationResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            ValidateOperationForPrevalidationCall(request),
            ValidateOperationResponse(result),
            ValidateOperationError,
            Some(Self::VALIDATE_OPERATION_TIMEOUT),
        )
    }

    /// Begin construction (for mempool, accumulates)
    pub async fn begin_construction_for_mempool(
        &mut self,
        request: BeginConstructionRequest,
    ) -> Result<PrevalidatorWrapper, ProtocolServiceError> {
        handle_request!(
            self.io,
            BeginConstructionForMempoolCall(request),
            BeginConstructionResult(result),
            BeginConstructionError,
            Some(Self::BEGIN_CONSTRUCTION_TIMEOUT),
        )
    }

    /// Validate operation (for mempool, accumulates)
    pub async fn validate_operation_for_mempool(
        &mut self,
        request: ValidateOperationRequest,
    ) -> Result<ValidateOperationResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            ValidateOperationForMempoolCall(request),
            ValidateOperationResponse(result),
            ValidateOperationError,
            Some(Self::VALIDATE_OPERATION_TIMEOUT),
        )
    }

    /// ComputePath
    pub async fn compute_path(
        &mut self,
        request: ComputePathRequest,
    ) -> Result<ComputePathResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            ComputePathCall(request),
            ComputePathResponse(result),
            ComputePathError,
            Some(Self::COMPUTE_PATH_TIMEOUT),
        )
    }

    pub async fn apply_block_result_metadata(
        &mut self,
        context_hash: ContextHash,
        metadata_bytes: RustBytes,
        max_operations_ttl: i32,
        protocol_hash: ProtocolHash,
        next_protocol_hash: ProtocolHash,
    ) -> Result<String, ProtocolServiceError> {
        let params = JsonEncodeApplyBlockResultMetadataParams {
            context_hash,
            max_operations_ttl,
            metadata_bytes,
            protocol_hash,
            next_protocol_hash,
        };

        handle_request!(
            self.io,
            JsonEncodeApplyBlockResultMetadata(params),
            JsonEncodeApplyBlockResultMetadataResponse(result) => result.map_err(|err| {
                ProtocolError::FfiJsonEncoderError {
                    caller: "apply_block_result_metadata".to_owned(),
                    reason: err,
                }
                .into()
            }),
            Some(Self::JSON_ENCODE_DATA_TIMEOUT),
        )
    }

    pub async fn apply_block_operations_metadata(
        &mut self,
        chain_id: ChainId,
        operations: Vec<Vec<Operation>>,
        operations_metadata_bytes: Vec<Vec<RustBytes>>,
        protocol_hash: ProtocolHash,
        next_protocol_hash: ProtocolHash,
    ) -> Result<String, ProtocolServiceError> {
        let params = JsonEncodeApplyBlockOperationsMetadataParams {
            chain_id,
            operations,
            operations_metadata_bytes,
            protocol_hash,
            next_protocol_hash,
        };

        handle_request!(
            self.io,
            JsonEncodeApplyBlockOperationsMetadata(params),
            JsonEncodeApplyBlockOperationsMetadata(result) => result.map_err(|err| {
                ProtocolError::FfiJsonEncoderError {
                    caller: "apply_block_operations_metadata".to_owned(),
                    reason: err,
                }
                .into()
            }),
            Some(Self::JSON_ENCODE_DATA_TIMEOUT),
        )
    }

    /// Call protocol  rpc - internal
    async fn call_protocol_rpc_internal(
        &mut self,
        request_path: String,
        request: ProtocolRpcRequest,
    ) -> Result<ProtocolRpcResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            ProtocolRpcCall(request),
            RpcResponse(result) => result.map_err(|err| {
                ProtocolError::ProtocolRpcError {
                    reason: err,
                    request_path,
                }
                .into()
            }),
            Some(Self::CALL_PROTOCOL_HEAVY_RPC_TIMEOUT),
        )
    }

    /// Call protocol rpc
    pub async fn call_protocol_rpc(
        &mut self,
        request: ProtocolRpcRequest,
    ) -> Result<ProtocolRpcResponse, ProtocolServiceError> {
        self.call_protocol_rpc_internal(request.request.context_path.clone(), request)
            .await
    }

    /// Call helpers_preapply_operations shell service
    pub async fn helpers_preapply_operations(
        &mut self,
        request: ProtocolRpcRequest,
    ) -> Result<HelpersPreapplyResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            HelpersPreapplyOperationsCall(request),
            HelpersPreapplyResponse(result),
            HelpersPreapplyError,
            Some(Self::CALL_PROTOCOL_RPC_TIMEOUT),
        )
    }

    /// Call helpers_preapply_block shell service
    pub async fn helpers_preapply_block(
        &mut self,
        request: HelpersPreapplyBlockRequest,
    ) -> Result<HelpersPreapplyResponse, ProtocolServiceError> {
        handle_request!(
            self.io,
            HelpersPreapplyBlockCall(request),
            HelpersPreapplyResponse(result),
            HelpersPreapplyError,
            Some(Self::CALL_PROTOCOL_RPC_TIMEOUT),
        )
    }

    /// Change tezos runtime configuration
    pub async fn change_runtime_configuration(
        &mut self,
        settings: TezosRuntimeConfiguration,
    ) -> Result<(), ProtocolServiceError> {
        handle_request!(
            self.io,
            ChangeRuntimeConfigurationCall(settings),
            ChangeRuntimeConfigurationResult => Ok(()),
            Some(Self::DEFAULT_TIMEOUT),
        )
    }

    /// Command tezos ocaml code to initialize context and protocol.
    /// CommitGenesisResult is returned only if commit_genesis is set to true
    async fn init_protocol_context(
        &mut self,
        storage: TezosContextStorageConfiguration,
        tezos_environment: &TezosEnvironmentConfiguration,
        commit_genesis: bool,
        enable_testchain: bool,
        readonly: bool,
        patch_context: Option<PatchContext>,
        context_stats_db_path: Option<PathBuf>,
    ) -> Result<InitProtocolContextResult, ProtocolServiceError> {
        let params = InitProtocolContextParams {
            storage,
            genesis: tezos_environment.genesis.clone(),
            genesis_max_operations_ttl: tezos_environment
                .genesis_additional_data()
                .map_err(|error| ProtocolServiceError::InvalidDataError {
                    message: format!("{:?}", error),
                })?
                .max_operations_ttl,
            protocol_overrides: tezos_environment.protocol_overrides.clone(),
            commit_genesis,
            enable_testchain,
            readonly,
            turn_off_context_raw_inspector: true, // TODO - TE-261: remove later, new context doesn't use it
            patch_context,
            context_stats_db_path,
        };

        handle_request!(
            self.io,
            InitProtocolContextCall(params),
            InitProtocolContextResult(result),
            OcamlStorageInitError,
            Some(Self::INIT_PROTOCOL_CONTEXT_TIMEOUT),
        )
    }

    /// Gracefully shutdown protocol runner
    pub async fn shutdown(&mut self) -> Result<(), ProtocolServiceError> {
        handle_request!(
            self.io,
            ShutdownCall,
            ShutdownResult => Ok(()),
            Some(Self::DEFAULT_TIMEOUT),
        )
    }

    /// Initialize protocol environment from default configuration (writeable).
    pub async fn init_protocol_for_write(
        &mut self,
        commit_genesis: bool,
        patch_context: &Option<PatchContext>,
        context_stats_db_path: Option<PathBuf>,
    ) -> Result<InitProtocolContextResult, ProtocolServiceError> {
        self.change_runtime_configuration(self.configuration.runtime_configuration.clone())
            .await?;
        let environment = self.configuration.environment.clone();
        self.init_protocol_context(
            self.configuration.storage.clone(),
            &environment,
            commit_genesis,
            self.configuration.enable_testchain,
            false,
            patch_context.clone(),
            context_stats_db_path,
        )
        .await
    }

    /// Initialize protocol environment from default configuration (readonly).
    pub async fn init_protocol_for_read(
        &mut self,
    ) -> Result<InitProtocolContextResult, ProtocolServiceError> {
        // TODO - TE-261: should use a different message exchange for readonly contexts?
        self.change_runtime_configuration(self.configuration.runtime_configuration.clone())
            .await?;
        let environment = self.configuration.environment.clone();
        self.init_protocol_context(
            self.configuration.storage.clone(),
            &environment,
            false,
            self.configuration.enable_testchain,
            true,
            None,
            None,
        )
        .await
    }

    // TODO - TE-261: this requires more descriptive errors.

    /// Initializes server to listen for readonly context clients through IPC.
    ///
    /// Must be called after the writable context has been initialized.
    pub async fn init_context_ipc_server(&mut self) -> Result<(), ProtocolServiceError> {
        if self.configuration.storage.get_ipc_socket_path().is_some() {
            let cfg = self.configuration.storage.clone();
            handle_request!(
                self.io,
                InitProtocolContextIpcServer(cfg),
                InitProtocolContextIpcServerResult(result) => {
                    result.map_err(|err| ProtocolServiceError::ContextIpcServerError {
                        message: format!("Failure when starting context IPC server: {}", err),
                    })
                },
                Some(Self::DEFAULT_TIMEOUT),
            )
        } else {
            Ok(())
        }
    }

    /// Gets data for genesis.
    pub async fn genesis_result_data(
        &mut self,
        genesis_context_hash: &ContextHash,
    ) -> Result<CommitGenesisResult, ProtocolServiceError> {
        let tezos_environment = self.configuration.environment.clone();
        let main_chain_id = tezos_environment.main_chain_id().map_err(|e| {
            ProtocolServiceError::InvalidDataError {
                message: format!("{:?}", e),
            }
        })?;
        let protocol_hash = tezos_environment.genesis_protocol().map_err(|e| {
            ProtocolServiceError::InvalidDataError {
                message: format!("{:?}", e),
            }
        })?;
        let params = GenesisResultDataParams {
            genesis_context_hash: genesis_context_hash.clone(),
            chain_id: main_chain_id,
            genesis_protocol_hash: protocol_hash,
            genesis_max_operations_ttl: tezos_environment
                .genesis_additional_data()
                .map_err(|error| ProtocolServiceError::InvalidDataError {
                    message: format!("{:?}", error),
                })?
                .max_operations_ttl,
        };

        handle_request!(
            self.io,
            GenesisResultDataCall(params),
            CommitGenesisResultData(result),
            GenesisResultDataError,
            Some(Self::DEFAULT_TIMEOUT),
        )
    }

    pub async fn get_context_key_from_history(
        &mut self,
        context_hash: &ContextHash,
        key: ContextKeyOwned,
    ) -> Result<Option<ContextValue>, ProtocolServiceError> {
        let params = ContextGetKeyFromHistoryRequest {
            context_hash: context_hash.clone(),
            key,
        };

        handle_request!(
            self.io,
            ContextGetKeyFromHistory(params),
            ContextGetKeyFromHistoryResult(result),
            ContextGetKeyFromHistoryError,
            Some(Self::DEFAULT_TIMEOUT),
        )
    }

    pub async fn get_context_key_values_by_prefix(
        &mut self,
        context_hash: &ContextHash,
        prefix: ContextKeyOwned,
    ) -> Result<Option<Vec<(ContextKeyOwned, ContextValue)>>, ProtocolServiceError> {
        let params = ContextGetKeyValuesByPrefixRequest {
            context_hash: context_hash.clone(),
            prefix,
        };

        handle_request!(
            self.io,
            ContextGetKeyValuesByPrefix(params),
            ContextGetKeyValuesByPrefixResult(result),
            ContextGetKeyValuesByPrefixError,
            Some(Self::DEFAULT_TIMEOUT_LONG),
        )
    }

    pub async fn get_context_tree_by_prefix(
        &mut self,
        context_hash: &ContextHash,
        prefix: ContextKeyOwned,
        depth: Option<usize>,
    ) -> Result<StringTreeObject, ProtocolServiceError> {
        let params = ContextGetTreeByPrefixRequest {
            context_hash: context_hash.clone(),
            prefix,
            depth,
        };

        handle_request!(
            self.io,
            ContextGetTreeByPrefix(params),
            ContextGetTreeByPrefixResult(result),
            ContextGetKeyValuesByPrefixError,
            Some(Self::DEFAULT_TIMEOUT_LONG),
        )
    }
}

// Errors

/// Errors generated by `protocol_runner`.
#[derive(Error, Debug)]
pub enum ProtocolServiceError {
    /// Generic IPC communication error. See `reason` for more details.
    #[error("IPC error: {reason}")]
    IpcError {
        #[from]
        reason: IpcError,
    },
    /// Tezos protocol error.
    #[error("Protocol error: {reason}")]
    ProtocolError {
        #[from]
        reason: ProtocolError,
    },
    /// Unexpected message was received from IPC channel
    #[error("Received unexpected message: {message}")]
    UnexpectedMessage { message: &'static str },
    /// Invalid data error
    #[error("Invalid data error: {message}")]
    InvalidDataError { message: String },
    /// Lock error
    #[error("Lock error: {message:?}")]
    LockPoisonError { message: String },
    /// Context IPC server error
    #[error("Context IPC server error: {message:?}")]
    ContextIpcServerError { message: String },
}

impl<T> From<std::sync::PoisonError<T>> for ProtocolServiceError {
    fn from(source: std::sync::PoisonError<T>) -> Self {
        Self::LockPoisonError {
            message: source.to_string(),
        }
    }
}

impl slog::Value for ProtocolServiceError {
    fn serialize(
        &self,
        _record: &slog::Record,
        key: slog::Key,
        serializer: &mut dyn slog::Serializer,
    ) -> slog::Result {
        serializer.emit_arguments(key, &format_args!("{}", self))
    }
}

pub fn handle_protocol_service_error<LC: Fn(ProtocolServiceError)>(
    error: ProtocolServiceError,
    log_callback: LC,
) -> Result<(), ProtocolServiceError> {
    match error {
        ProtocolServiceError::IpcError { .. } | ProtocolServiceError::UnexpectedMessage { .. } => {
            // we need to refresh protocol runner endpoint, so propagate error
            Err(error)
        }
        _ => {
            // just log error
            log_callback(error);
            Ok(())
        }
    }
}
