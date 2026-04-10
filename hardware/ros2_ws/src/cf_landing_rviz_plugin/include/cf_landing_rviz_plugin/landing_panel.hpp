#ifndef CF_LANDING_RVIZ_PLUGIN__LANDING_PANEL_HPP_
#define CF_LANDING_RVIZ_PLUGIN__LANDING_PANEL_HPP_

#include <mutex>
#include <memory>

#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QProcess>
#include <QTimer>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <cf_landing_interfaces/msg/mission_status.hpp>
#include <cf_landing_interfaces/srv/command.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace cf_landing_rviz_plugin
{

// Widget displaying one agent's state (drone or rover)
class AgentStatusWidget : public QFrame
{
  Q_OBJECT
public:
  explicit AgentStatusWidget(const QString & name, bool is_drone = true, QWidget * parent = nullptr);
  void updateFromOdom(const nav_msgs::msg::Odometry & odom);
  void updateFromVelRaw(const geometry_msgs::msg::Twist & twist);
  void setConnected(bool connected);

private:
  QLabel * indicator_;
  QLabel * name_label_;
  QLabel * pos_label_;
  QLabel * vel_label_;
  QLabel * orient_label_;
  QLabel * body_vel_label_;  // rover only
  bool is_drone_;
  bool connected_ = false;
};

// Main rviz2 panel
class LandingPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit LandingPanel(QWidget * parent = nullptr);
  ~LandingPanel() override;

  void onInitialize() override;

private Q_SLOTS:
  void onTakeoffClicked();
  void onRunClicked();
  void onAbortClicked();
  void onOffClicked();
  void onResetOdomClicked();
  void onRecordClicked();
  void onRecordProcessFinished(int exit_code, QProcess::ExitStatus status);
  void updateUI();

private:
  void setupUI();
  void callCommand(uint8_t cmd);

  // ROS2
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<cf_landing_interfaces::msg::MissionStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr drone_odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr rover_odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr rover_vel_sub_;
  rclcpp::Client<cf_landing_interfaces::srv::Command>::SharedPtr command_client_;

  // Thread-safe state
  std::mutex mutex_;
  cf_landing_interfaces::msg::MissionStatus::SharedPtr latest_status_;
  nav_msgs::msg::Odometry::SharedPtr latest_drone_odom_;
  nav_msgs::msg::Odometry::SharedPtr latest_rover_odom_;
  geometry_msgs::msg::Twist::SharedPtr latest_rover_vel_;
  bool drone_connected_ = false;
  bool rover_connected_ = false;
  rclcpp::Time last_drone_odom_time_;
  rclcpp::Time last_rover_odom_time_;

  // UI elements
  QLabel * status_value_;
  QLabel * distance_value_;
  QLabel * rel_pos_value_;
  QPushButton * takeoff_btn_;
  QPushButton * run_btn_;
  QPushButton * abort_btn_;
  QPushButton * off_btn_;
  QPushButton * record_btn_;
  QPushButton * reset_odom_btn_;
  AgentStatusWidget * drone_widget_;
  AgentStatusWidget * rover_widget_;
  QTimer * ui_timer_;

  // TF for global position lookup
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Recording
  QProcess * record_process_ = nullptr;
};

}  // namespace cf_landing_rviz_plugin

#endif  // CF_LANDING_RVIZ_PLUGIN__LANDING_PANEL_HPP_
