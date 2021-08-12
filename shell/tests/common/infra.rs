// Copyright (c) SimpleStaking, Viable Systems and Tezedge Contributors
// SPDX-License-Identifier: MIT

//! Module which runs actor's very similar than real node runs

#[cfg(test)]
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use riker::actors::*;
use riker::system::SystemBuilder;
use slog::{info, warn, Level, Logger};
use tezos_api::ffi::TezosContextTezEdgeStorageConfiguration;
use tokio::runtime::Runtime;

use common::contains_all_keys;
use crypto::hash::{BlockHash, OperationHash};
use networking::ShellCompatibilityVersion;
use shell::chain_current_head_manager::ChainCurrentHeadManager;
use shell::chain_feeder::{ChainFeeder, ChainFeederRef};
use shell::chain_manager::ChainManager;
use shell::mempool::{
    init_mempool_state_storage, CurrentMempoolStateStorageRef, MempoolPrevalidatorFactory,
};
use shell::shell_channel::{ShellChannel, ShellChannelRef, ShellChannelTopic, ShuttingDown};
use shell::state::head_state::init_current_head_state;
use shell::state::synchronization_state::{
    init_synchronization_bootstrap_state_storage, SynchronizationBootstrapStateRef,
};
use shell::subscription::*;
use shell::tezedge_state_manager::{P2p, TezedgeStateManager};
use shell::PeerConnectionThreshold;
use storage::chain_meta_storage::ChainMetaStorageReader;
use storage::tests_common::TmpStorage;
use storage::{resolve_storage_init_chain_data, ChainMetaStorage};
use tezos_api::environment::TezosEnvironmentConfiguration;
use tezos_api::ffi::{
    PatchContext, TezosContextIrminStorageConfiguration, TezosContextStorageConfiguration,
    TezosRuntimeConfiguration,
};
use tezos_identity::Identity;
use tezos_wrapper::ProtocolEndpointConfiguration;
use tezos_wrapper::{TezosApiConnectionPool, TezosApiConnectionPoolConfiguration};

use crate::common;

pub struct NodeInfrastructure {
    name: String,
    pub log: Logger,
    pub tezedge_state_manager: TezedgeStateManager,
    pub block_applier: ChainFeederRef,
    pub shell_channel: ShellChannelRef,
    pub actor_system: ActorSystem,
    pub tmp_storage: TmpStorage,
    pub current_mempool_state_storage: CurrentMempoolStateStorageRef,
    pub bootstrap_state: SynchronizationBootstrapStateRef,
    pub tezos_env: TezosEnvironmentConfiguration,
    pub tokio_runtime: Runtime,
    pub mempool_prevalidator_factory: Arc<MempoolPrevalidatorFactory>,
}

impl NodeInfrastructure {
    pub fn start(
        tmp_storage: TmpStorage,
        context_db_path: &str,
        name: &str,
        tezos_env: &TezosEnvironmentConfiguration,
        patch_context: Option<PatchContext>,
        p2p: Option<(P2p, ShellCompatibilityVersion)>,
        identity: Identity,
        pow_target: f64,
        (log, log_level): (Logger, Level),
        (record_also_readonly_context_action, compute_context_action_tree_hashes): (bool, bool),
        network_event_listner: Option<common::infra::test_actor::NetworkEventListener>,
    ) -> Result<Self, failure::Error> {
        warn!(log, "[NODE] Starting node infrastructure"; "name" => name);

        // environement
        let is_sandbox = false;
        let (p2p_threshold, p2p_disable_mempool) = match p2p.as_ref() {
            Some((p2p, _)) => (p2p.peer_threshold.clone(), p2p.disable_mempool),
            None => (PeerConnectionThreshold::try_new(1, 1, Some(0))?, false),
        };
        let identity = Arc::new(identity);

        // storage
        let persistent_storage = tmp_storage.storage();
        let context_db_path = if !PathBuf::from(context_db_path).exists() {
            common::prepare_empty_dir(context_db_path)
        } else {
            context_db_path.to_string()
        };

        let ipc_socket_path = Some(ipc::temp_sock().to_string_lossy().as_ref().to_owned());

        let context_storage_configuration = TezosContextStorageConfiguration::Both(
            TezosContextIrminStorageConfiguration {
                data_dir: context_db_path,
            },
            TezosContextTezEdgeStorageConfiguration {
                backend: tezos_api::ffi::ContextKvStoreConfiguration::InMem,
                ipc_socket_path,
            },
        );

        let init_storage_data = resolve_storage_init_chain_data(
            &tezos_env,
            &tmp_storage.path(),
            &context_storage_configuration,
            &patch_context,
            &None,
            &None,
            &log,
        )
        .expect("Failed to resolve init storage chain data");

        let tokio_runtime = create_tokio_runtime();

        // create pool for ffi protocol runner connections (used just for readonly context)
        let tezos_readonly_api_pool = Arc::new(TezosApiConnectionPool::new_with_readonly_context(
            String::from(&format!("{}_readonly_runner_pool", name)),
            TezosApiConnectionPoolConfiguration {
                min_connections: 0,
                max_connections: 2,
                connection_timeout: Duration::from_secs(3),
                max_lifetime: Duration::from_secs(60),
                idle_timeout: Duration::from_secs(60),
            },
            ProtocolEndpointConfiguration::new(
                TezosRuntimeConfiguration {
                    log_enabled: common::is_ocaml_log_enabled(),
                    debug_mode: false,
                    compute_context_action_tree_hashes: false,
                },
                tezos_env.clone(),
                false,
                context_storage_configuration.readonly(),
                &common::protocol_runner_executable_path(),
                log_level,
            ),
            tokio_runtime.handle().clone(),
            log.clone(),
        )?);

        // create pool for ffi protocol runner connections (used just for readonly context)
        let tezos_writeable_api = Arc::new(TezosApiConnectionPool::new_without_context(
            String::from(&format!("{}_writeable_runner_pool", name)),
            TezosApiConnectionPoolConfiguration {
                min_connections: 0,
                max_connections: 1,
                connection_timeout: Duration::from_secs(3),
                max_lifetime: Duration::from_secs(60),
                idle_timeout: Duration::from_secs(60),
            },
            ProtocolEndpointConfiguration::new(
                TezosRuntimeConfiguration {
                    log_enabled: common::is_ocaml_log_enabled(),
                    debug_mode: record_also_readonly_context_action,
                    compute_context_action_tree_hashes,
                },
                tezos_env.clone(),
                false,
                context_storage_configuration,
                &common::protocol_runner_executable_path(),
                log_level,
            ),
            tokio_runtime.handle().clone(),
            log.clone(),
        )?);

        let local_current_head_state = init_current_head_state();
        let remote_current_head_state = init_current_head_state();
        let current_mempool_state_storage = init_mempool_state_storage();
        let bootstrap_state = init_synchronization_bootstrap_state_storage(
            p2p_threshold.num_of_peers_for_bootstrap_threshold(),
        );

        // run actor's
        let actor_system = SystemBuilder::new()
            .name(name)
            .log(log.clone())
            .create()
            .expect("Failed to create actor system");
        let shell_channel =
            ShellChannel::actor(&actor_system).expect("Failed to create shell channel");
        let mut new_current_head_notifier = Notifier::<NewCurrentHeadNotificationRef>::new(log.clone());
        let mut network_event_notifier = Notifier::<NetworkEventNotificationRef>::new(log.clone());
        let mut shell_event_notifier = Notifier::<ShellEventNotificationRef>::new(log.clone());
        let mut shell_command_notifier = Notifier::<ShellCommandNotificationRef>::new(log.clone());

        // initialize tezedge state
        let mut tezedge_state_manager = if let Some((p2p_config, shell_compatibility_version)) = p2p
        {
            TezedgeStateManager::new(
                log.clone(),
                identity.clone(),
                Arc::new(shell_compatibility_version),
                p2p_config,
                pow_target,
                init_storage_data.chain_id.clone(),
            )
        } else {
            panic!("P2p cfg is required")
        };

        // initialize actors
        let mempool_prevalidator_factory = Arc::new(MempoolPrevalidatorFactory::new(
            actor_system.clone(),
            shell_channel.clone(),
            tezedge_state_manager.proposer_handle(),
            persistent_storage.clone(),
            current_mempool_state_storage.clone(),
            tezos_readonly_api_pool.clone(),
            p2p_disable_mempool,
        ));

        let chain_current_head_manager = ChainCurrentHeadManager::actor(
            &actor_system,
            tezedge_state_manager.proposer_handle(),
            new_current_head_notifier,
            persistent_storage.clone(),
            init_storage_data.clone(),
            local_current_head_state.clone(),
            remote_current_head_state.clone(),
            current_mempool_state_storage.clone(),
            bootstrap_state.clone(),
            mempool_prevalidator_factory.clone(),
        )
        .expect("Failed to create chain current head manager");
        let block_applier = ChainFeeder::actor(
            &actor_system,
            chain_current_head_manager,
            shell_channel.clone(),
            persistent_storage.clone(),
            tezos_writeable_api,
            init_storage_data.clone(),
            tezos_env.clone(),
            log.clone(),
        )
        .expect("Failed to create chain feeder");
        let chain_manager_wrapper = ChainManager::actor(
            &actor_system,
            block_applier.clone(),
            tezedge_state_manager.proposer_handle(),
            shell_channel.clone(),
            shell_event_notifier,
            persistent_storage.clone(),
            tezos_readonly_api_pool.clone(),
            log.clone(),
            init_storage_data.clone(),
            is_sandbox,
            local_current_head_state,
            remote_current_head_state,
            current_mempool_state_storage.clone(),
            bootstrap_state.clone(),
            mempool_prevalidator_factory.clone(),
            identity.clone(),
        )
        .expect("Failed to create chain manager");
        chain_manager_wrapper.subscribe(&mut network_event_notifier);

        // network event listener
        if let Some(network_event_listner) = network_event_listner {
            network_event_listner.subscribe(&mut network_event_notifier);
        }

        // and than open p2p and others - if configured
        if let Err(e) = tezedge_state_manager.start(network_event_notifier, shell_command_notifier) {
            panic!("Failed to start tezedge state/proposer, reason: {:?}", e);
        }

        Ok(NodeInfrastructure {
            name: String::from(name),
            log,
            tezedge_state_manager,
            block_applier,
            shell_channel,
            tokio_runtime,
            actor_system,
            tmp_storage,
            current_mempool_state_storage,
            bootstrap_state,
            tezos_env: tezos_env.clone(),
            mempool_prevalidator_factory,
        })
    }

    fn stop(&mut self) {
        let NodeInfrastructure {
            log,
            shell_channel,
            actor_system,
            tokio_runtime,
            tezedge_state_manager,
            ..
        } = self;
        warn!(log, "[NODE] Stopping node infrastructure"; "name" => self.name.clone());

        // clean up + shutdown events listening
        shell_channel.tell(
            Publish {
                msg: ShuttingDown.into(),
                topic: ShellChannelTopic::ShellShutdown.into(),
            },
            None,
        );

        // close tezedge state explicitly
        drop(tezedge_state_manager);

        let _ = tokio_runtime.block_on(async move {
            tokio::time::timeout(Duration::from_secs(10), actor_system.shutdown()).await
        });

        warn!(log, "[NODE] Node infrastructure stopped"; "name" => self.name.clone());
    }

    // TODO: refactor with async/condvar, not to block main thread
    pub fn wait_for_new_current_head(
        &self,
        marker: &str,
        tested_head: BlockHash,
        (timeout, delay): (Duration, Duration),
    ) -> Result<(), failure::Error> {
        let start = Instant::now();
        let tested_head = Some(tested_head).map(|th| th.to_base58_check());

        let chain_meta_data = ChainMetaStorage::new(self.tmp_storage.storage());
        let result = loop {
            let current_head = chain_meta_data
                .get_current_head(&self.tezos_env.main_chain_id()?)?
                .map(|ch| ch.block_hash().to_base58_check());

            if current_head.eq(&tested_head) {
                info!(self.log, "[NODE] Expected current head detected"; "head" => tested_head, "marker" => marker);
                break Ok(());
            }

            // kind of simple retry policy
            if start.elapsed().le(&timeout) {
                thread::sleep(delay);
            } else {
                break Err(failure::format_err!("wait_for_new_current_head({:?}) - timeout (timeout: {:?}, delay: {:?}) exceeded! marker: {}", tested_head, timeout, delay, marker));
            }
        };
        result
    }

    // TODO: refactor with async/condvar, not to block main thread
    pub fn wait_for_mempool_on_head(
        &self,
        marker: &str,
        tested_head: BlockHash,
        (timeout, delay): (Duration, Duration),
    ) -> Result<(), failure::Error> {
        let start = Instant::now();
        let tested_head = Some(tested_head).map(|th| th.to_base58_check());

        let result = loop {
            let mempool_state = self
                .current_mempool_state_storage
                .read()
                .expect("Failed to obtain lock");
            let current_head = mempool_state.head().map(|ch| ch.to_base58_check());

            if current_head.eq(&tested_head) {
                info!(self.log, "[NODE] Expected mempool head detected"; "head" => tested_head, "marker" => marker);
                break Ok(());
            }

            // kind of simple retry policy
            if start.elapsed().le(&timeout) {
                thread::sleep(delay);
            } else {
                break Err(failure::format_err!("wait_for_mempool_on_head({:?}) - timeout (timeout: {:?}, delay: {:?}) exceeded! marker: {}", tested_head, timeout, delay, marker));
            }
        };
        result
    }

    // TODO: refactor with async/condvar, not to block main thread
    pub fn wait_for_mempool_contains_operations(
        &self,
        marker: &str,
        expected_operations: &HashSet<OperationHash>,
        (timeout, delay): (Duration, Duration),
    ) -> Result<(), failure::Error> {
        let start = Instant::now();

        let result = loop {
            let mempool_state = self
                .current_mempool_state_storage
                .read()
                .expect("Failed to obtain lock");
            if contains_all_keys(mempool_state.operations(), expected_operations) {
                info!(self.log, "[NODE] All expected operations found in mempool"; "marker" => marker);
                break Ok(());
            }
            drop(mempool_state);

            // kind of simple retry policy
            if start.elapsed().le(&timeout) {
                thread::sleep(delay);
            } else {
                break Err(failure::format_err!("wait_for_mempool_contains_operations() - timeout (timeout: {:?}, delay: {:?}) exceeded! marker: {}", timeout, delay, marker));
            }
        };
        result
    }

    // TODO: refactor with async/condvar, not to block main thread
    pub fn wait_for_bootstrapped(
        &self,
        marker: &str,
        (timeout, delay): (Duration, Duration),
    ) -> Result<(), failure::Error> {
        let start = Instant::now();

        let result = loop {
            let is_bootstrapped = self
                .bootstrap_state
                .read()
                .expect("Failed to get lock")
                .is_bootstrapped();

            if is_bootstrapped {
                info!(self.log, "[NODE] Expected node is bootstrapped"; "marker" => marker);
                break Ok(());
            }

            // kind of simple retry policy
            if start.elapsed().le(&timeout) {
                thread::sleep(delay);
            } else {
                break Err(failure::format_err!("wait_for_bootstrapped - timeout (timeout: {:?}, delay: {:?}) exceeded! marker: {}", timeout, delay, marker));
            }
        };
        result
    }

    pub fn whitelist_all(&self) {
        // TODO: TE-652 - whitelist scenario
        // if let Some(peer_manager) = &self.peer_manager {
        //     peer_manager.tell(WhitelistAllIpAddresses, None);
        // }
    }
}

impl Drop for NodeInfrastructure {
    fn drop(&mut self) {
        warn!(self.log, "[NODE] Dropping node infrastructure"; "name" => self.name.clone());
        self.stop();
    }
}

pub fn create_tokio_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
}

pub mod test_actor {
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock, mpsc};
    use std::time::{Duration, Instant};

    use crypto::hash::CryptoboxPublicKeyHash;
    use shell::subscription::*;

    use crate::common::test_node_peer::TestNodePeer;
    use std::net::IpAddr;

    #[derive(PartialEq, Eq, Debug)]
    pub enum PeerConnectionStatus {
        Connected,
        Blacklisted,
        Stalled,
    }

    pub type PeersMirror = Arc<
        RwLock<(
            HashMap<CryptoboxPublicKeyHash, PeerConnectionStatus>,
            HashMap<IpAddr, PeerConnectionStatus>,
            HashMap<CryptoboxPublicKeyHash, IpAddr>,
        )>,
    >;

    pub enum NotifyNetworkEventListenerMsg {
        NetworkEvent(NetworkEventNotificationRef),
        Shutdown,
    }

    pub struct NetworkEventListener {
        peers_mirror: PeersMirror,
        notify_tx: mpsc::SyncSender<NotifyNetworkEventListenerMsg>,
    }

    impl NetworkEventListener {

        pub fn create_and_start(
            peers_mirror: PeersMirror,
            log: slog::Logger,
        ) -> Result<NetworkEventListener, std::io::Error> {

            // start temporariry new network event listener thread
            let (notify_tx, notify_rx) =
                mpsc::sync_channel(1000);

            // spawn mpsc queue receiver for notifications
            {
                let peers_mirror = peers_mirror.clone();
                std::thread::Builder::new()
                    .name("test-network-event-listener".to_owned())
                    .spawn(move || {
                        Self::run_notify_network_event_listener(
                            notify_rx,
                            peers_mirror,
                            log,
                        );
                    })?;
            };

            Ok(NetworkEventListener {
                peers_mirror,
                notify_tx,
            })
        }

        fn process_network_event_message(
            peers_mirror: &PeersMirror,
            msg: NetworkEventNotificationRef,
        ) {
            match msg.as_ref() {
                NetworkEventNotification::PeerMessageReceived(_) => {}
                NetworkEventNotification::PeerDisconnected(_) => {}
                NetworkEventNotification::PeerBootstrapped(peer_id, _, _) => {
                    peers_mirror.write().unwrap().0.insert(
                        peer_id.public_key_hash.clone(),
                        PeerConnectionStatus::Connected,
                    );
                    peers_mirror
                        .write()
                        .unwrap()
                        .1
                        .insert(peer_id.address.ip(), PeerConnectionStatus::Connected);
                    peers_mirror
                        .write()
                        .unwrap()
                        .2
                        .insert(peer_id.public_key_hash.clone(), peer_id.address.ip());
                }
                NetworkEventNotification::PeerBlacklisted(peer_address) => {
                    peers_mirror
                        .write()
                        .unwrap()
                        .1
                        .insert(peer_address.ip(), PeerConnectionStatus::Blacklisted);
                }
            }
        }

        pub fn verify_connected(
            peer: &TestNodePeer,
            peers_mirror: PeersMirror,
        ) -> Result<(), failure::Error> {
            Self::verify_state(
                PeerConnectionStatus::Connected,
                peer,
                peers_mirror,
                (Duration::from_secs(5), Duration::from_millis(250)),
            )
        }

        pub fn verify_blacklisted(
            peer: &TestNodePeer,
            peers_mirror: PeersMirror,
        ) -> Result<(), failure::Error> {
            Self::verify_state(
                PeerConnectionStatus::Blacklisted,
                peer,
                peers_mirror,
                (Duration::from_secs(5), Duration::from_millis(250)),
            )
        }

        // TODO: refactor with async/condvar, not to block main thread
        fn verify_state(
            expected_state: PeerConnectionStatus,
            peer: &TestNodePeer,
            peers_mirror: PeersMirror,
            (timeout, delay): (Duration, Duration),
        ) -> Result<(), failure::Error> {
            let start = Instant::now();
            let public_key_hash = &peer.identity.public_key.public_key_hash()?;

            let result = loop {
                let peers_mirror = peers_mirror.read().unwrap();
                if let Some(peer_state) = peers_mirror.0.get(public_key_hash) {
                    if peer_state == &expected_state {
                        break Ok(());
                    }

                    // if we know peer we check also blacklisted IPs
                    if let Some(peer_address) = peers_mirror.2.get(public_key_hash) {
                        // check ips
                        if let Some(peer_address_state) = peers_mirror.1.get(peer_address) {
                            if peer_address_state == &expected_state {
                                break Ok(());
                            }
                        }
                    }
                }

                // kind of simple retry policy
                if start.elapsed().le(&timeout) {
                    std::thread::sleep(delay);
                } else {
                    break Err(
                        failure::format_err!(
                            "[{}] verify_state - peer_public_key({}) - (expected_state: {:?}) - timeout (timeout: {:?}, delay: {:?}) exceeded!",
                            peer.name, public_key_hash.to_base58_check(), expected_state, timeout, delay
                        )
                    );
                }
            };
            result
        }

        fn run_notify_network_event_listener(
            rx: mpsc::Receiver<NotifyNetworkEventListenerMsg>,
            peers_mirror: PeersMirror,
            log: slog::Logger,
        ) {
            // Read and handle messages incoming from actor system or `PeerManager`.
            loop {
                match rx.recv() {
                    Ok(NotifyNetworkEventListenerMsg::NetworkEvent(notification)) => {
                        Self::process_network_event_message(&peers_mirror, notification)
                    }
                    Ok(NotifyNetworkEventListenerMsg::Shutdown) => {
                        slog::info!(log, "Network event listener notification listener finished");
                        break;
                    }
                    Err(mpsc::RecvError) => {
                        slog::info!(log, "Network event listener notification listener finished (by disconnect)");
                        return;
                    }
                }
            }
        }
    }

    impl NotifyCallbackRegistrar<NetworkEventNotificationRef> for NetworkEventListener {
        fn subscribe(&self, notifier: &mut Notifier<NetworkEventNotificationRef>) {
            notifier.register(
                Box::new(NetworkEventListenerNotifyCallback(
                    self.notify_tx.clone(),
                )),
                "chain-manager-network-event-listener".to_string(),
            );
        }
    }

    struct NetworkEventListenerNotifyCallback(mpsc::SyncSender<NotifyNetworkEventListenerMsg>);

    impl NotifyCallback<NetworkEventNotificationRef> for NetworkEventListenerNotifyCallback {
        fn notify(
            &self,
            notification: NetworkEventNotificationRef,
        ) -> Result<(), SendNotificationError> {
            self.0
                .send(NotifyNetworkEventListenerMsg::NetworkEvent(notification))
                .map_err(|e| SendNotificationError {
                    reason: format!("{}", e),
                })
        }

        fn shutdown(&self) -> Result<(), ShutdownCallbackError> {
            self.0
                .try_send(NotifyNetworkEventListenerMsg::Shutdown)
                .map_err(|e| ShutdownCallbackError {
                    reason: format!("{}", e),
                })
        }
    }
}
