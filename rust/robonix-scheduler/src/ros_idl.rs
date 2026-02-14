// SPDX-License-Identifier: MulanPSL-2.0
// Scheduler Service ROS IDL Message Types

use serde::{Deserialize, Serialize};

/// AdjustPriority service request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustPriorityRequest {
    pub skill_name: String,
    pub high_priority: bool,
}

impl ros2_client::Message for AdjustPriorityRequest {}

/// AdjustPriority service response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustPriorityResponse {
    pub ok: bool,
}

impl ros2_client::Message for AdjustPriorityResponse {}

/// RegisterProcess service request — register or unregister a process by std_name + PID.
/// This allows callers to provide PID mappings directly in memory, avoiding
/// file-based processes.json lookups and reducing scheduler overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterProcessRequest {
    pub std_name: String,
    pub pid: u32,
    /// true = register, false = unregister
    pub do_register: bool,
}

impl ros2_client::Message for RegisterProcessRequest {}

/// RegisterProcess service response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterProcessResponse {
    pub ok: bool,
}

impl ros2_client::Message for RegisterProcessResponse {}
