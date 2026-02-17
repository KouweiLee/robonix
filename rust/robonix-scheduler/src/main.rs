// SPDX-License-Identifier: MulanPSL-2.0
// Robonix Scheduler - Policy Governor Service

use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use ros2_client::{
    Context as RosContext, NodeOptions, ServiceMapping, ServiceTypeName, Name, NodeName,
    AService, rustdds::{QosPolicyBuilder, policy::Reliability, Duration as RustddsDuration},
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use futures::StreamExt;

mod ros_idl;
use ros_idl::{
    AdjustPriorityRequest, AdjustPriorityResponse,
    RegisterProcessRequest, RegisterProcessResponse,
};

/// Resolve home directory. When running with sudo (home=/root), use SUDO_USER or cwd.
fn resolve_home_dir() -> PathBuf {
    let mut home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
    if home_dir.to_str() == Some("/root") {
        if let Ok(sudo_user) = std::env::var("SUDO_USER") {
            let path = PathBuf::from("/home").join(&sudo_user);
            if path.exists() {
                home_dir = path;
            }
        } else if let Ok(cwd) = std::env::current_dir() {
            let parts: Vec<_> = cwd.components().collect();
            if parts.len() >= 3 && parts[1].as_os_str() == "home" {
                home_dir = PathBuf::from("/home").join(parts[2]);
            }
        }
    }
    home_dir
}

/// xsched configuration for GPU scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
struct XschedConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_xcli_path")]
    pub xcli_path: String,
    #[serde(default = "default_xsched_addr")]
    pub server_addr: String,
    #[serde(default = "default_xsched_port")]
    pub server_port: u16,
    #[serde(default = "default_high_priority")]
    pub high_priority: i32,
    #[serde(default)]
    pub normal_priority: i32,
}

fn default_true() -> bool {
    true
}
fn default_xcli_path() -> String {
    "~/.robonix/bin/xcli".to_string()
}

/// Expand ~ to HOME in path. Uses resolve_home_dir() for consistency when running as root.
fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        resolve_home_dir().join(path.trim_start_matches("~/"))
    } else if path == "~" {
        resolve_home_dir()
    } else {
        PathBuf::from(path)
    }
}
fn default_xsched_addr() -> String {
    "127.0.0.1".to_string()
}
fn default_xsched_port() -> u16 {
    50000
}
fn default_high_priority() -> i32 {
    10
}

impl Default for XschedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            xcli_path: default_xcli_path(),
            server_addr: "127.0.0.1".to_string(),
            server_port: 50000,
            high_priority: 10,
            normal_priority: 0,
        }
    }
}

/// Scheduler configuration loaded from ~/.robonix/scheduler.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SchedulerConfig {
    #[serde(default = "default_skill_dependencies")]
    pub skill_dependencies: HashMap<String, Vec<String>>,
    #[serde(default = "default_infrastructure_patterns")]
    pub infrastructure_patterns: Vec<String>,
    #[serde(default)]
    pub xpu_components: Vec<String>,
    #[serde(default)]
    pub xsched: XschedConfig,
}

fn default_skill_dependencies() -> HashMap<String, Vec<String>> {
    let mut m = HashMap::new();
    m.insert(
        "skl::move_to_object".to_string(),
        vec![
            "prm::base.navigate".to_string(),
            "prm::base.pose.cov".to_string(),
            "srv::semantic_map".to_string(),
            "prm::camera.rgb".to_string(),
            "prm::camera.depth".to_string(),
        ],
    );
    m.insert(
        "skl::wandering".to_string(),
        vec![
            "prm::base.navigate".to_string(),
            "prm::base.pose.cov".to_string(),
        ],
    );
    // Benchmark skills (scheduler_benchmark package)
    m.insert(
        "skl::bench_nav".to_string(),
        vec![
            "prm::base.navigate".to_string(),
            "prm::base.pose.cov".to_string(),
            "srv::bench_slam".to_string(),
        ],
    );
    m.insert(
        "skl::bench_grasp".to_string(),
        vec![
            "prm::camera.rgb".to_string(),
            "prm::camera.depth".to_string(),
            "srv::bench_perception".to_string(),
        ],
    );
    m.insert(
        "skl::bench_inspect".to_string(),
        vec![
            "prm::camera.rgb".to_string(),
            "prm::camera.depth".to_string(),
            "srv::bench_perception".to_string(),
        ],
    );
    m
}

fn default_infrastructure_patterns() -> Vec<String> {
    vec![
        "webots".to_string(),
        "Webots".to_string(),
        "webots_ros2".to_string(),
        "robot_launch".to_string(),
        "xserver".to_string(),
    ]
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            skill_dependencies: default_skill_dependencies(),
            infrastructure_patterns: default_infrastructure_patterns(),
            xpu_components: Vec::new(),
            xsched: XschedConfig::default(),
        }
    }
}

impl SchedulerConfig {
    /// Config is in ~/.robonix/scheduler.yaml. When running with sudo, resolve real user's home.
    fn config_path() -> Result<PathBuf> {
        let home_dir = resolve_home_dir();
        Ok(home_dir.join(".robonix").join("scheduler.yaml"))
    }

    fn load() -> Self {
        let path = match Self::config_path() {
            Ok(p) => p,
            Err(_) => return Self::default(),
        };
        if !path.exists() {
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => match serde_yaml::from_str(&content) {
                Ok(cfg) => {
                    info!("Loaded scheduler config from {}", path.display());
                    cfg
                }
                Err(e) => {
                    warn!("Failed to parse scheduler config {}: {}. Using defaults.", path.display(), e);
                    Self::default()
                }
            },
            Err(e) => {
                warn!("Failed to read scheduler config {}: {}. Using defaults.", path.display(), e);
                Self::default()
            }
        }
    }
}

/// Information about a running process (mirrors robonix-cli/src/process.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessInfo {
    pub package_name: String,
    pub std_name: String,
    pub package_type: String, // "cap" or "skl"
    pub pid: u32,
    pub log_file: PathBuf,
    pub hostname: String,
}

/// Priority level for a component
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum PriorityLevel {
    /// Latency critical: use SCHED_RR (Real-Time)
    LatencyCritical,
    /// Throughput critical: use Nice -10 (High Priority CFS)
    ThroughputCritical,
}

fn send_xcli_hint(xsched: &XschedConfig, pid: u32, name: &str, priority: i32) {
    let xcli_path = expand_tilde(&xsched.xcli_path);
    let output = std::process::Command::new(&xcli_path)
        .args([
            "-a",
            &xsched.server_addr,
            "-p",
            &xsched.server_port.to_string(),
            "hint",
            "--pid",
            &pid.to_string(),
            "-p",
            &priority.to_string(),
        ])
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                info!("xsched hint: {} (PID {}) -> priority {}", name, pid, priority);
            } else {
                let stderr = String::from_utf8_lossy(&result.stderr);
                debug!("xsched hint failed for {} (PID {}): {}", name, pid, stderr);
            }
        }
        Err(e) => {
            debug!("xsched xcli exec failed for {}: {}", name, e);
        }
    }
}

pub struct PolicyGovernor {
    /// Mapping of skill name -> (component name, priority level)
    dependencies: HashMap<String, Vec<(String, PriorityLevel)>>,
    /// Process name patterns for infrastructure (Webots, webots_ros2_driver, etc.)
    infrastructure_patterns: Vec<String>,
    /// In-memory process registry (std_name -> PID), populated via scheduler_register service.
    /// This is checked FIRST during PID resolution, avoiding file I/O entirely when populated.
    process_registry: Arc<RwLock<HashMap<String, u32>>>,
    /// File-based process cache (fallback when registry has no entry).
    /// Loaded lazily on first need, then never re-read (file content is static at runtime).
    process_cache: Arc<RwLock<HashMap<String, u32>>>,
    file_cache_loaded: AtomicBool,
    // Reference count now tracks the level
    priority_refs: Arc<RwLock<HashMap<String, (usize, PriorityLevel)>>>,
    /// Number of skills currently requesting high priority. When > 0, infra is boosted.
    infrastructure_refs: Arc<RwLock<usize>>,
    /// PIDs of infrastructure processes we've boosted (for restore)
    infrastructure_pids: Arc<RwLock<Vec<u32>>>,
    state_file: PathBuf,
    /// Components that use GPU (std_name), for xsched scheduling
    xpu_components: HashSet<String>,
    xsched: XschedConfig,
    xpu_priority_refs: Arc<RwLock<HashMap<String, usize>>>,
}

impl PolicyGovernor {
    pub fn new() -> Self {
        let config = SchedulerConfig::load();

        // Build dependencies: skill -> [(component, ThroughputCritical), ...]
        let dependencies: HashMap<String, Vec<(String, PriorityLevel)>> = config
            .skill_dependencies
            .into_iter()
            .map(|(skill, components)| {
                let deps = components
                    .into_iter()
                    .map(|c| (c, PriorityLevel::ThroughputCritical))
                    .collect();
                (skill, deps)
            })
            .collect();

        let home_dir = resolve_home_dir();
        let state_file = home_dir.join(".robonix").join("processes.json");
        let xpu_components: HashSet<String> = config.xpu_components.into_iter().collect();
        info!("Using process state file: {}", state_file.display());
        info!(
            "Scheduler: {} skills, {} infra patterns, {} xpu components",
            dependencies.len(),
            config.infrastructure_patterns.len(),
            xpu_components.len()
        );

        Self {
            dependencies,
            infrastructure_patterns: config.infrastructure_patterns,
            process_registry: Arc::new(RwLock::new(HashMap::new())),
            process_cache: Arc::new(RwLock::new(HashMap::new())),
            file_cache_loaded: AtomicBool::new(false),
            priority_refs: Arc::new(RwLock::new(HashMap::new())),
            infrastructure_refs: Arc::new(RwLock::new(0)),
            infrastructure_pids: Arc::new(RwLock::new(Vec::new())),
            state_file,
            xpu_components,
            xsched: config.xsched,
            xpu_priority_refs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load processes.json into the file cache. Only reads from disk once;
    /// subsequent calls are a no-op (the file content is effectively static
    /// at runtime — written once by rbnx-daemon at startup).
    pub async fn ensure_file_cache_loaded(&self) {
        if self.file_cache_loaded.load(Ordering::Acquire) {
            return;
        }

        if !self.state_file.exists() {
            error!("Process state file not found at {} (not an error if using registry)",
                   self.state_file.display());
            self.file_cache_loaded.store(true, Ordering::Release);
            return;
        }

        match tokio::fs::read_to_string(&self.state_file).await {
            Ok(content) => {
                match serde_json::from_str::<Vec<ProcessInfo>>(&content) {
                    Ok(processes) => {
                        let mut cache = self.process_cache.write().await;
                        for p in processes {
                            if Self::is_process_running(p.pid) {
                                cache.insert(p.std_name.clone(), p.pid);
                            } else {
                                debug!("Skipping stale process {} (PID {})", p.std_name, p.pid);
                            }
                        }
                        info!("Loaded file cache with {} processes", cache.len());
                    }
                    Err(e) => warn!("Failed to parse process state file: {}", e),
                }
            }
            Err(e) => warn!("Failed to read process state file {}: {}", self.state_file.display(), e),
        }

        self.file_cache_loaded.store(true, Ordering::Release);
    }

    fn is_process_running(pid: u32) -> bool {
        unsafe {
            // Signal 0 is used to check for existence
            libc::kill(pid as libc::pid_t, 0) == 0
        }
    }

    /// Register a process in the in-memory registry. Returns true on success.
    pub async fn register_process(&self, std_name: &str, pid: u32) -> bool {
        if !Self::is_process_running(pid) {
            warn!("Cannot register {} (PID {}): process not running", std_name, pid);
            return false;
        }
        let mut registry = self.process_registry.write().await;
        info!("Registered process: {} -> PID {}", std_name, pid);
        registry.insert(std_name.to_string(), pid);
        true
    }

    /// Unregister a process from the in-memory registry. Returns true if it was present.
    pub async fn unregister_process(&self, std_name: &str) -> bool {
        let mut registry = self.process_registry.write().await;
        if registry.remove(std_name).is_some() {
            info!("Unregistered process: {}", std_name);
            true
        } else {
            debug!("Unregister: {} was not in registry", std_name);
            false
        }
    }

    /// Resolve a std_name to a PID. Checks in-memory registry first (zero-overhead),
    /// then falls back to the file-based process cache.
    /// Note: no liveness check here — the downstream syscall (sched_setscheduler /
    /// setpriority) will return ESRCH if the process is gone, which is already handled.
    async fn resolve_pid(&self, std_name: &str) -> Option<u32> {
        // Fast path: in-memory registry (pure HashMap lookup, no syscall)
        {
            let registry = self.process_registry.read().await;
            if let Some(&pid) = registry.get(std_name) {
                return Some(pid);
            }
        }
        // Slow path: file-based cache
        let cache = self.process_cache.read().await;
        cache.get(std_name).copied()
    }

    pub async fn adjust_priorities(&self, skill_name: &str, high_priority: bool) {
        // Only refresh file cache if the in-memory registry doesn't cover all targets.
        // This is a lazy refresh: we check if we need it after collecting targets.
        let mut targets = HashMap::new();
        let full_skill_name = if skill_name.starts_with("skl::") {
            skill_name.to_string()
        } else {
            format!("skl::{}", skill_name)
        };
        
        // Skill itself defaults to ThroughputCritical
        targets.insert(full_skill_name.clone(), PriorityLevel::ThroughputCritical);

        if let Some(deps) = self.dependencies.get(&full_skill_name) {
            for (dep, level) in deps {
                targets.insert(dep.clone(), *level);
            }
        }

        // Lazily load the file cache if any target is not in the in-memory registry.
        // The file is read at most once for the scheduler's entire lifetime.
        {
            let registry = self.process_registry.read().await;
            let need_file = targets.keys().any(|t| !registry.contains_key(t));
            drop(registry);
            if need_file {
                self.ensure_file_cache_loaded().await;
            }
        }

        let mut refs = self.priority_refs.write().await;
        let mut xpu_refs = self.xpu_priority_refs.write().await;
        let xpu_enabled = self.xsched.enabled
            && !self.xpu_components.is_empty()
            && !self.xsched.xcli_path.is_empty();

        for (target, level) in targets {
            info!("Adjusting priority for {}: level={:?}", target, level);
            let (count, _) = refs.entry(target.clone()).or_insert((0, level));

            if high_priority {
                *count += 1;
                if *count == 1 {
                    if let Some(pid) = self.resolve_pid(&target).await {
                        Self::set_linux_priority(pid, target.as_str(), Some(level));
                    }
                }
            } else {
                if *count > 0 {
                    *count -= 1;
                    if *count == 0 {
                        if let Some(pid) = self.resolve_pid(&target).await {
                            Self::set_linux_priority(pid, target.as_str(), None);
                        }
                    }
                }
            }

            // GPU (xsched): apply to targets in xpu_components
            if xpu_enabled && self.xpu_components.contains(&target) {
                let xpu_count = xpu_refs.entry(target.clone()).or_insert(0);
                if high_priority {
                    *xpu_count += 1;
                    if *xpu_count == 1 {
                        if let Some(pid) = self.resolve_pid(&target).await {
                            send_xcli_hint(
                                &self.xsched,
                                pid,
                                &target,
                                self.xsched.high_priority,
                            );
                        }
                    }
                } else {
                    if *xpu_count > 0 {
                        *xpu_count -= 1;
                        if *xpu_count == 0 {
                            if let Some(pid) = self.resolve_pid(&target).await {
                                send_xcli_hint(
                                    &self.xsched,
                                    pid,
                                    &target,
                                    self.xsched.normal_priority,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Infrastructure: ensure Webots/webots_ros2_driver etc. have priority >= any skill
        drop(refs);
        drop(xpu_refs);
        let mut infra_refs = self.infrastructure_refs.write().await;
        if high_priority {
            *infra_refs += 1;
            if *infra_refs == 1 {
                self.boost_infrastructure().await;
            }
        } else if *infra_refs > 0 {
            *infra_refs -= 1;
            if *infra_refs == 0 {
                self.restore_infrastructure().await;
            }
        }
    }

    /// Discover and boost simulation infrastructure processes (Webots, webots_ros2_driver).
    async fn boost_infrastructure(&self) {
        let pids = self.discover_infrastructure_pids();
        if pids.is_empty() {
            debug!("No infrastructure processes found");
            return;
        }
        info!("Boosting {} infrastructure process(es): {:?}", pids.len(), pids);
        for &pid in &pids {
            Self::set_linux_priority(pid, "infra", Some(PriorityLevel::ThroughputCritical));
        }
        *self.infrastructure_pids.write().await = pids;
    }

    /// Restore infrastructure processes to normal priority.
    async fn restore_infrastructure(&self) {
        let pids: Vec<u32> = {
            let mut guard = self.infrastructure_pids.write().await;
            std::mem::take(&mut *guard)
        };
        if pids.is_empty() {
            return;
        }
        info!("Restoring {} infrastructure process(es) to normal priority", pids.len());
        for pid in pids {
            Self::set_linux_priority(pid, "infra", None);
        }
    }

    /// Find PIDs of processes matching infrastructure patterns (webots, webots_ros2, robot_launch).
    fn discover_infrastructure_pids(&self) -> Vec<u32> {
        let my_pid = std::process::id();
        let mut seen = HashSet::new();
        for pattern in &self.infrastructure_patterns {
            if let Ok(output) = std::process::Command::new("pgrep")
                .arg("-f")
                .arg(pattern)
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        if let Ok(pid) = line.trim().parse::<u32>() {
                            if pid != my_pid && Self::is_process_running(pid) {
                                seen.insert(pid);
                            }
                        }
                    }
                }
            }
        }
        seen.into_iter().collect()
    }

    /// Helper to get all thread IDs (TIDs) for a given process PID from /proc
    fn get_thread_ids(pid: u32) -> Vec<u32> {
        let mut tids = Vec::new();
        // Always include the main PID itself (which is the main thread ID)
        tids.push(pid);

        let path = format!("/proc/{}/task", pid);
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Ok(fname) = entry.file_name().into_string() {
                    if let Ok(tid) = fname.parse::<u32>() {
                        if tid != pid {
                            tids.push(tid);
                        }
                    }
                }
            }
        }
        tids
    }

    /// Set Linux scheduling priority for a process and all its threads.
    /// Iterates over /proc/<pid>/task/ to find all TIDs.
    fn set_linux_priority(pid: u32, name: &str, level: Option<PriorityLevel>) {
        let tids = Self::get_thread_ids(pid);
        if tids.len() > 1 {
            debug!("Setting priority for {} (PID {}) and {} threads", name, pid, tids.len() - 1);
        }

        for tid in tids {
            Self::set_linux_priority_for_tid(tid, name, level);
        }
    }

    /// Apply priority to a specific thread/task ID.
    fn set_linux_priority_for_tid(tid: u32, name: &str, level: Option<PriorityLevel>) {
        unsafe {
            match level {
                Some(PriorityLevel::LatencyCritical) => {
                    // RT scheduling is still thread-based in Linux
                    let mut param: libc::sched_param = std::mem::zeroed();
                    param.sched_priority = 5; 
                    if libc::sched_setscheduler(tid as libc::pid_t, libc::SCHED_RR, &param) == 0 {
                        debug!("Set {} (TID {}) to RT (SCHED_RR) priority 5", name, tid);
                    } else {
                        let err = std::io::Error::last_os_error();
                        if err.raw_os_error() == Some(libc::ESRCH) {
                            return;
                        }
                        debug!("RT failed for {} (TID {}): {}. Fallback to nice.", name, tid, err);
                        let _ = libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, -15);
                    }
                }
                Some(PriorityLevel::ThroughputCritical) => {
                    // Use PRIO_PROCESS for each TID
                    if libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, -10) != 0 {
                        let err = std::io::Error::last_os_error();
                        if err.raw_os_error() != Some(libc::ESRCH) {
                            debug!("Failed nice for {} (TID {}): {}", name, tid, err);
                        }
                    } else {
                        debug!("Adjusted {} (TID {}) nice to -10", name, tid);
                    }
                }
                None => {
                    // Restore priority
                    let mut param: libc::sched_param = std::mem::zeroed();
                    param.sched_priority = 0;
                    let _ = libc::sched_setscheduler(tid as libc::pid_t, libc::SCHED_OTHER, &param);
                    let _ = libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, 0);
                    debug!("Restored {} (TID {}) to Normal", name, tid);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check command line arguments manually since we don't use clap yet
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "--help" | "-h" => {
                println!("Robonix Scheduler - Policy Governor Service");
                println!("Usage: robonix-scheduler");
                println!("\nConfiguration is loaded from ~/.robonix/scheduler.yaml");
                return Ok(());
            }
            "--version" | "-v" => {
                println!("robonix-scheduler v{}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            _ => {
                eprintln!("Error: Unknown argument '{}'", args[1]);
                eprintln!("Usage: robonix-scheduler [--help|--version]");
                std::process::exit(1);
            }
        }
    }

    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info,robonix_scheduler=debug,rustdds=off"));
    info!("robonix scheduler starting...");

    // Set own priority to high (nice -10) permanently at startup.
    // This ensures the scheduler itself is never starved by background noise.
    unsafe {
        let pid = libc::getpid();
        if libc::setpriority(libc::PRIO_PROCESS, pid as libc::id_t, -10) == 0 {
            info!("Robonix scheduler set own priority to high (nice -10)");
        } else {
            let err = std::io::Error::last_os_error();
            warn!("Failed to set own priority: {}. (Are you running with sudo?)", err);
        }
    }

    let governor = Arc::new(PolicyGovernor::new());

    // Source ROS2 environment
    let ros_context = RosContext::new().context("Failed to create ROS2 context")?;
    let mut node = ros_context
        .new_node(
            NodeName::new("/rbnx", "scheduler").unwrap(),
            NodeOptions::new().enable_rosout(true),
        )
        .context("Failed to create ROS2 node")?;

    let service_qos = QosPolicyBuilder::new()
        .reliability(Reliability::Reliable {
            max_blocking_time: RustddsDuration::from_millis(100),
        })
        .build();

    // Service 1: scheduler_policy (priority adjustment)
    let policy_name = Name::parse("scheduler_policy").unwrap();
    let policy_type = ServiceTypeName::new("robonix_sdk", "AdjustPriority");

    let policy_server = node
        .create_server::<AService<AdjustPriorityRequest, AdjustPriorityResponse>>(
            ServiceMapping::Enhanced,
            &policy_name,
            &policy_type,
            service_qos.clone(),
            service_qos.clone(),
        )
        .context("Failed to create priority adjustment service")?;

    // Service 2: scheduler_register (in-memory PID registration)
    let register_name = Name::parse("scheduler_register").unwrap();
    let register_type = ServiceTypeName::new("robonix_sdk", "RegisterProcess");

    let register_server = node
        .create_server::<AService<RegisterProcessRequest, RegisterProcessResponse>>(
            ServiceMapping::Enhanced,
            &register_name,
            &register_type,
            service_qos.clone(),
            service_qos,
        )
        .context("Failed to create process registration service")?;

    info!("robonix scheduler ready, services: scheduler_policy, scheduler_register");

    // Run both services concurrently
    let gov_policy = governor.clone();
    let policy_task = tokio::spawn(async move {
        let stream = policy_server.receive_request_stream();
        stream.for_each(|result| async {
            match result {
                Ok((req_id, req)) => {
                    info!("Received adjustment request: skill={}, high={}", req.skill_name, req.high_priority);

                    gov_policy.adjust_priorities(&req.skill_name, req.high_priority).await;

                    let response = AdjustPriorityResponse { ok: true };
                    if let Err(e) = policy_server.async_send_response(req_id, response).await {
                        warn!("Failed to send policy response: {:?}", e);
                    }
                }
                Err(e) => warn!("Policy request error: {:?}", e),
            }
        }).await;
    });

    let gov_register = governor.clone();
    let register_task = tokio::spawn(async move {
        let stream = register_server.receive_request_stream();
        stream.for_each(|result| async {
            match result {
                Ok((req_id, req)) => {
                    let ok = if req.do_register {
                        info!("Register request: {} -> PID {}", req.std_name, req.pid);
                        gov_register.register_process(&req.std_name, req.pid).await
                    } else {
                        info!("Unregister request: {}", req.std_name);
                        gov_register.unregister_process(&req.std_name).await
                    };

                    let response = RegisterProcessResponse { ok };
                    if let Err(e) = register_server.async_send_response(req_id, response).await {
                        warn!("Failed to send register response: {:?}", e);
                    }
                }
                Err(e) => warn!("Register request error: {:?}", e),
            }
        }).await;
    });

    // Wait for both services (they run forever)
    tokio::select! {
        _ = policy_task => warn!("Policy service exited unexpectedly"),
        _ = register_task => warn!("Register service exited unexpectedly"),
    }

    Ok(())
}
