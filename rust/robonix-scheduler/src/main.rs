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
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use futures::StreamExt;

mod ros_idl;
use ros_idl::{AdjustPriorityRequest, AdjustPriorityResponse};

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
    process_cache: Arc<RwLock<HashMap<String, u32>>>,
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
            process_cache: Arc::new(RwLock::new(HashMap::new())),
            priority_refs: Arc::new(RwLock::new(HashMap::new())),
            infrastructure_refs: Arc::new(RwLock::new(0)),
            infrastructure_pids: Arc::new(RwLock::new(Vec::new())),
            state_file,
            xpu_components,
            xsched: config.xsched,
            xpu_priority_refs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn update_process_cache(&self) {
        if !self.state_file.exists() {
            error!("Process state file not found at {}", self.state_file.display());
            return;
        }

        match tokio::fs::read_to_string(&self.state_file).await {
            Ok(content) => {
                match serde_json::from_str::<Vec<ProcessInfo>>(&content) {
                    Ok(processes) => {
                        let mut cache = self.process_cache.write().await;
                        cache.clear();
                        for p in processes {
                            if Self::is_process_running(p.pid) {
                                cache.insert(p.std_name.clone(), p.pid);
                            } else {
                                debug!("Skipping stale process {} (PID {})", p.std_name, p.pid);
                            }
                        }
                        debug!("Updated process cache with {} processes", cache.len());
                    }
                    Err(e) => warn!("Failed to parse process state file: {}", e),
                }
            }
            Err(e) => warn!("Failed to read process state file {}: {}", self.state_file.display(), e),
        }
    }

    fn is_process_running(pid: u32) -> bool {
        unsafe {
            // Signal 0 is used to check for existence
            libc::kill(pid as libc::pid_t, 0) == 0
        }
    }

    pub async fn adjust_priorities(&self, skill_name: &str, high_priority: bool) {
        self.update_process_cache().await;

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

        let cache = self.process_cache.read().await;
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
                    if let Some(&pid) = cache.get(&target) {
                        Self::set_linux_priority(pid, target.as_str(), Some(level));
                    }
                }
            } else {
                if *count > 0 {
                    *count -= 1;
                    if *count == 0 {
                        if let Some(&pid) = cache.get(&target) {
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
                        if let Some(&pid) = cache.get(&target) {
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
                            if let Some(&pid) = cache.get(&target) {
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
        drop(cache);
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
                            if Self::is_process_running(pid) {
                                seen.insert(pid);
                            }
                        }
                    }
                }
            }
        }
        seen.into_iter().collect()
    }

    fn set_linux_priority(pid: u32, name: &str, level: Option<PriorityLevel>) {
        if !Self::is_process_running(pid) {
            warn!("Skip priority adjustment for {} (PID {}): Process not found", name, pid);
            return;
        }

        unsafe {
            // Get actual PGID (pid may be a child process resolved from the group)
            let pgid = libc::getpgid(pid as libc::pid_t);
            let pgid = if pgid > 0 { pgid as libc::id_t } else { pid as libc::id_t };
            match level {
                Some(PriorityLevel::LatencyCritical) => {
                    // RT scheduling is still thread-based in Linux, no group setting
                    // We set RT for the main PID, and high priority Nice for the group as fallback
                    let mut param: libc::sched_param = std::mem::zeroed();
                    param.sched_priority = 5; 
                    if libc::sched_setscheduler(pid as libc::pid_t, libc::SCHED_RR, &param) == 0 {
                        info!("Set {} (PID {}) to RT (SCHED_RR) priority 5", name, pid);
                    } else {
                        let err = std::io::Error::last_os_error();
                        if err.raw_os_error() == Some(libc::ESRCH) {
                            warn!("RT failed for {} (PID {}): No such process", name, pid);
                        } else {
                            error!("RT failed for {} (PID {}): {}. Fallback to nice.", name, pid, err);
                        }
                        let _ = libc::setpriority(libc::PRIO_PGRP, pgid, -15);
                    }
                }
                Some(PriorityLevel::ThroughputCritical) => {
                    // Use PRIO_PGRP to adjust nice for the entire process group (including children)
                    if libc::setpriority(libc::PRIO_PGRP, pgid, -10) != 0 {
                        let err = std::io::Error::last_os_error();
                        if err.raw_os_error() == Some(libc::ESRCH) {
                            warn!("Failed nice for {} (PGID {}): No such process", name, pid);
                        } else {
                            error!("Failed nice for {} (PGID {}): {}", name, pid, err);
                        }
                    } else {
                        info!("Adjusted {} (PGID {}) nice to -10", name, pid);
                    }
                }
                None => {
                    // Restore priority for the entire process group
                    let mut param: libc::sched_param = std::mem::zeroed();
                    param.sched_priority = 0;
                    let _ = libc::sched_setscheduler(pid as libc::pid_t, libc::SCHED_OTHER, &param);
                    let _ = libc::setpriority(libc::PRIO_PGRP, pgid, 0);
                    info!("Restored {} (PGID {}) to Normal", name, pid);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info,robonix_scheduler=debug,rustdds=off"));
    info!("robonix scheduler starting...");

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

    let service_name = Name::parse("scheduler_policy").unwrap();
    let service_type = ServiceTypeName::new("robonix_sdk", "AdjustPriority");

    let server = node
        .create_server::<AService<AdjustPriorityRequest, AdjustPriorityResponse>>(
            ServiceMapping::Enhanced,
            &service_name,
            &service_type,
            service_qos.clone(),
            service_qos,
        )
        .context("Failed to create priority adjustment service")?;

    info!("robonix scheduler ready, service: scheduler_policy");

    let stream = server.receive_request_stream();
    stream.for_each(|result| async {
        match result {
            Ok((req_id, req)) => {
                info!("Received adjustment request: skill={}, high={}", req.skill_name, req.high_priority);
                
                let gov = governor.clone();
                let skill_name = req.skill_name.clone();
                let high_priority = req.high_priority;

                gov.adjust_priorities(&skill_name, high_priority).await;
                
                let response = AdjustPriorityResponse { ok: true };
                if let Err(e) = server.async_send_response(req_id, response).await {
                    warn!("Failed to send response: {:?}", e);
                }
            }
            Err(e) => warn!("Receive request error: {:?}", e),
        }
    }).await;

    Ok(())
}
