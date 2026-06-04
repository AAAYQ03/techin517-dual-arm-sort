// Copyright 2024 SOA Teleop
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class TeleopNode : public rclcpp::Node
{
public:
  TeleopNode()
  : Node("teleop_node")
  {
    declare_parameter<std::vector<std::string>>(
      "arm_joints",
      {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"});
    declare_parameter<std::string>("gripper_joint", "gripper");

    arm_joints_ = get_parameter("arm_joints").as_string_array();
    gripper_joint_ = get_parameter("gripper_joint").as_string();

    arm_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/follower/arm_fwd_controller/commands", 10);
    gripper_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/follower/gripper_fwd_controller/commands", 10);

    leader_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/leader/joint_states", 10,
      std::bind(&TeleopNode::on_joint_states, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Teleop node started — relaying leader → follower");
  }

private:
  void on_joint_states(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    // Build name → position lookup
    std::unordered_map<std::string, double> pos_map;
    for (std::size_t i = 0; i < msg->name.size(); ++i) {
      if (i < msg->position.size()) {
        pos_map[msg->name[i]] = msg->position[i];
      }
    }

    // Arm command — ordered to match arm_fwd_controller joint list
    std_msgs::msg::Float64MultiArray arm_cmd;
    arm_cmd.data.reserve(arm_joints_.size());
    for (const auto & joint : arm_joints_) {
      auto it = pos_map.find(joint);
      if (it == pos_map.end()) {
        if (!warned_missing_arm_) {
          RCLCPP_WARN(get_logger(),
            "Arm joint '%s' not found in leader joint states — skipping message",
            joint.c_str());
          warned_missing_arm_ = true;
        }
        return;
      }
      arm_cmd.data.push_back(it->second);
    }

    // Gripper command
    auto git = pos_map.find(gripper_joint_);
    if (git == pos_map.end()) {
      if (!warned_missing_gripper_) {
        RCLCPP_WARN(get_logger(),
          "Gripper joint '%s' not found in leader joint states — skipping message",
          gripper_joint_.c_str());
        warned_missing_gripper_ = true;
      }
      return;
    }

    // All joints found — clear any previous warnings and publish
    warned_missing_arm_ = false;
    warned_missing_gripper_ = false;

    arm_pub_->publish(arm_cmd);

    std_msgs::msg::Float64MultiArray gripper_cmd;
    gripper_cmd.data.push_back(git->second);
    gripper_pub_->publish(gripper_cmd);
  }

  std::vector<std::string> arm_joints_;
  std::string gripper_joint_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr arm_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gripper_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr leader_sub_;

  bool warned_missing_arm_{false};
  bool warned_missing_gripper_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TeleopNode>());
  rclcpp::shutdown();
  return 0;
}
