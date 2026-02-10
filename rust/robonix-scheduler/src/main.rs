// SPDX-License-Identifier: MulanPSL-2.0
// Robonix Scheduler - Policy Governor Service

use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use ros2_client::{
    Context as RosContext, NodeOptions, ServiceMapping, ServiceTypeName, Name, NodeName,
    AService, rustdds::{QosPolicyBuilder, policy::Reliability, Duration as RustddsDuration},
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use futures::StreamExt;

mod ros_idl;
use ros_idl::{AdjustPriorityRequest, AdjustPriorityResponse};

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

pub struct XpuScheduler {
    // Reserved for xsched integration
}

impl XpuScheduler {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn adjust_xpu_scheduling(&self, _skill_name: &str, _high_priority: bool) {
        // Placeholder for xsched integration
    }
}

pub struct PolicyGovernor {
    // Mapping of skill name -> (component name, priority level)
    dependencies: HashMap<String, Vec<(String, PriorityLevel)>>,
    process_cache: Arc<RwLock<HashMap<String, u32>>>,
    // Reference count now tracks the level
    priority_refs: Arc<RwLock<HashMap<String, (usize, PriorityLevel)>>>,
    state_file: PathBuf,
}

impl PolicyGovernor {
    pub fn new() -> Self {
        let mut dependencies = HashMap::new();
        
        // 关键改进：区分实时(RR)和高优先级(Nice)
        // 只有涉及到物理运动控制的 prm::base.navigate 给予实时权限
        dependencies.insert(
            "skl::move_to_object".to_string(),
            vec![
                ("prm::base.navigate".to_string(), PriorityLevel::ThroughputCritical),
                ("prm::base.pose.cov".to_string(), PriorityLevel::ThroughputCritical),
                ("srv::semantic_map".to_string(), PriorityLevel::ThroughputCritical),
                ("prm::camera.rgb".to_string(), PriorityLevel::ThroughputCritical),
                ("prm::camera.depth".to_string(), PriorityLevel::ThroughputCritical),
            ],
        );

        dependencies.insert(
            "skl::wandering".to_string(),
            vec![
                ("prm::base.navigate".to_string(), PriorityLevel::ThroughputCritical),
                ("prm::base.pose.cov".to_string(), PriorityLevel::ThroughputCritical),
            ],
        );

        let home_dir = dirs::home_dir().expect("Failed to get home directory");
        let state_file = home_dir.join(".robonix").join("processes.json");

        Self {
            dependencies,
            process_cache: Arc::new(RwLock::new(HashMap::new())),
            priority_refs: Arc::new(RwLock::new(HashMap::new())),
            state_file,
        }
    }

    pub async fn update_process_cache(&self) {
        if !self.state_file.exists() {
            debug!("Process state file not found at {}", self.state_file.display());
            return;
        }

        match tokio::fs::read_to_string(&self.state_file).await {
            Ok(content) => {
                match serde_json::from_str::<Vec<ProcessInfo>>(&content) {
                    Ok(processes) => {
                        let mut cache = self.process_cache.write().await;
                        cache.clear();
                        for p in processes {
                            cache.insert(p.std_name.clone(), p.pid);
                        }
                        debug!("Updated process cache with {} processes", cache.len());
                    }
                    Err(e) => warn!("Failed to parse process state file: {}", e),
                }
            }
            Err(e) => warn!("Failed to read process state file {}: {}", self.state_file.display(), e),
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
        
        // 技能本身默认属于 ThroughputCritical
        targets.insert(full_skill_name.clone(), PriorityLevel::ThroughputCritical);

        if let Some(deps) = self.dependencies.get(&full_skill_name) {
            for (dep, level) in deps {
                targets.insert(dep.clone(), *level);
            }
        }

        let cache = self.process_cache.read().await;
        let mut refs = self.priority_refs.write().await;

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
        }
    }

    fn set_linux_priority(pid: u32, name: &str, level: Option<PriorityLevel>) {
        unsafe {
            match level {
                Some(PriorityLevel::LatencyCritical) => {
                    // 对于 LatencyCritical，尝试设置为实时调度策略 (SCHED_RR)
                    let mut param: libc::sched_param = std::mem::zeroed();
                    param.sched_priority = 5; 
                    if libc::sched_setscheduler(pid as libc::pid_t, libc::SCHED_RR, &param) == 0 {
                        info!("Set {} (PID {}) to RT (SCHED_RR) priority 5", name, pid);
                    } else {
                        let err = std::io::Error::last_os_error();
                        error!("RT failed for {} (PID {}): {}. Fallback to nice.", name, pid, err);
                        let _ = libc::setpriority(libc::PRIO_PROCESS, pid as libc::id_t, -15);
                    }
                }
                Some(PriorityLevel::ThroughputCritical) => {
                    // 对于 ThroughputCritical，仅调整 nice 值，不进入实时调度
                    if libc::setpriority(libc::PRIO_PROCESS, pid as libc::id_t, -10) != 0 {
                        let err = std::io::Error::last_os_error();
                        error!("Failed nice for {} (PID {}): {}", name, pid, err);
                    } else {
                        info!("Adjusted {} (PID {}) nice to -10", name, pid);
                    }
                }
                None => {
                    // 还原优先级
                    let mut param: libc::sched_param = std::mem::zeroed();
                    param.sched_priority = 0;
                    let _ = libc::sched_setscheduler(pid as libc::pid_t, libc::SCHED_OTHER, &param);
                    let _ = libc::setpriority(libc::PRIO_PROCESS, pid as libc::id_t, 0);
                    info!("Restored {} (PID {}) to Normal", name, pid);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info,robonix_scheduler=debug,rustdds=error"));
    info!("robonix scheduler starting...");

    let governor = Arc::new(PolicyGovernor::new());
    let xpu_scheduler = Arc::new(XpuScheduler::new());
    
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
                let xpu = xpu_scheduler.clone();
                let skill_name = req.skill_name.clone();
                let high_priority = req.high_priority;
                
                // Process adjustment
                gov.adjust_priorities(&skill_name, high_priority).await;
                xpu.adjust_xpu_scheduling(&skill_name, high_priority).await;
                
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
