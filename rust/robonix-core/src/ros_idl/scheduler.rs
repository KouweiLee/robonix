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
