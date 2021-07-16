// Copyright (c) SimpleStaking, Viable Systems and Tezedge Contributors
// SPDX-License-Identifier: MIT

use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::{Child, Command};

use failure::Fail;
use nix::sys::signal;
use nix::sys::signal::Signal;
use nix::unistd::Pid;
use slog::{info, warn, Level, Logger};

use crate::ProtocolEndpointConfiguration;

/// Errors generated by `protocol_runner`.
#[derive(Fail, Debug)]
pub enum ProtocolRunnerError {
    #[fail(
        display = "Failed to spawn tezos protocol wrapper sub-process: {}",
        reason
    )]
    SpawnError { reason: io::Error },
    #[fail(
        display = "Failed to terminate/kill tezos protocol wrapper sub-process, reason: {}",
        reason
    )]
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

/// Control protocol runner sub-process.
#[derive(Clone)]
pub struct ExecutableProtocolRunner {
    sock_cmd_path: PathBuf,
    executable_path: PathBuf,
    endpoint_name: String,
    tokio_runtime: tokio::runtime::Handle,
    log_level: Level,
}

impl ExecutableProtocolRunner {
    /// Send SIGINT signal to the sub-process, which is cheking for this ctrl-c signal and shuts down gracefully if recieved
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
    fn log_subprocess_output(&self, process: &mut Child, log: Logger) {
        // Only launch logging task if the output port if present, otherwise log a warning.
        macro_rules! handle_output {
            ($tag:expr, $name:expr, $io:expr, $log:expr) => {{
                if let Some(out) = $io.take() {
                    let log = $log;
                    self.tokio_runtime.spawn(async move {
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

impl ProtocolRunner for ExecutableProtocolRunner {
    type Subprocess = Child;
    const PROCESS_TERMINATE_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

    fn new(
        configuration: ProtocolEndpointConfiguration,
        sock_cmd_path: &Path,
        endpoint_name: String,
        tokio_runtime: tokio::runtime::Handle,
    ) -> Self {
        let ProtocolEndpointConfiguration {
            executable_path,
            log_level,
            ..
        } = configuration;
        ExecutableProtocolRunner {
            sock_cmd_path: sock_cmd_path.to_path_buf(),
            executable_path,
            endpoint_name,
            tokio_runtime,
            log_level,
        }
    }

    fn spawn(&self, log: Logger) -> Result<Self::Subprocess, ProtocolRunnerError> {
        let _guard = self.tokio_runtime.enter();
        let mut process = Command::new(&self.executable_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .arg("--sock-cmd")
            .arg(&self.sock_cmd_path)
            .arg("--endpoint")
            .arg(&self.endpoint_name)
            .arg("--log-level")
            .arg(&self.log_level.as_str().to_lowercase())
            .spawn()
            .map_err(|err| ProtocolRunnerError::SpawnError { reason: err })?;

        self.log_subprocess_output(&mut process, log.clone());

        Ok(process)
    }

    fn wait_and_terminate_ref(
        tokio_runtime: tokio::runtime::Handle,
        process: &mut Self::Subprocess,
        wait_timeout: Duration,
        log: &Logger,
    ) -> Result<(), ProtocolRunnerError> {
        tokio_runtime.block_on(async move {
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
                Err(_) => {
                    Self::terminate_or_kill(process, "wait timeout exceeded".to_string()).await
                }
            }
        })
    }

    fn is_running(process: &mut Self::Subprocess) -> bool {
        matches!(process.try_wait(), Ok(None))
    }
}

pub trait ProtocolRunner: Clone + Send + Sync {
    type Subprocess: Send;
    const PROCESS_TERMINATE_WAIT_TIMEOUT: Duration;

    fn new(
        configuration: ProtocolEndpointConfiguration,
        sock_cmd_path: &Path,
        endpoint_name: String,
        tokio_runtime: tokio::runtime::Handle,
    ) -> Self;

    fn spawn(&self, log: Logger) -> Result<Self::Subprocess, ProtocolRunnerError>;

    /// Give [`wait_timeout`] time to stop process, and after that if tries to terminate/kill it
    fn wait_and_terminate_ref(
        tokio_runtime: tokio::runtime::Handle,
        process: &mut Self::Subprocess,
        wait_timeout: Duration,
        log: &Logger,
    ) -> Result<(), ProtocolRunnerError>;

    /// Checks if process is running
    fn is_running(process: &mut Self::Subprocess) -> bool;
}
