#include "cf_landing_rviz_plugin/landing_panel.hpp"

#include <QScrollArea>
#include <QDateTime>
#include <QDir>

#include <rviz_common/display_context.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2/exceptions.h>
#include <robot_localization/srv/set_pose.hpp>
#include <cmath>

namespace cf_landing_rviz_plugin
{

// ─── AgentStatusWidget ─────────────────────────────────────────────────────

AgentStatusWidget::AgentStatusWidget(const QString & name, bool is_drone, QWidget * parent)
: QFrame(parent), is_drone_(is_drone)
{
  setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
  setLineWidth(1);

  auto * layout = new QVBoxLayout(this);
  layout->setSpacing(2);
  layout->setContentsMargins(6, 4, 6, 4);

  // Header: indicator + name
  auto * header = new QHBoxLayout();
  indicator_ = new QLabel("●");
  indicator_->setStyleSheet("color: #FF0000; font-size: 14px;");
  indicator_->setFixedWidth(20);
  name_label_ = new QLabel(name);
  name_label_->setStyleSheet("font-weight: bold; font-size: 11px;");
  header->addWidget(indicator_);
  header->addWidget(name_label_);
  header->addStretch();
  layout->addLayout(header);

  // State labels
  QFont mono("monospace", 9);

  pos_label_ = new QLabel("Pos:   --- --- --- [m]");
  pos_label_->setFont(mono);
  layout->addWidget(pos_label_);

  vel_label_ = new QLabel("Vel:   --- --- --- [m/s]");
  vel_label_->setFont(mono);
  layout->addWidget(vel_label_);

  orient_label_ = new QLabel("Quat:  --- --- --- ---");
  orient_label_->setFont(mono);
  layout->addWidget(orient_label_);

  if (!is_drone_) {
    body_vel_label_ = new QLabel("Body:  --- --- --- [m/s, rad/s]");
    body_vel_label_->setFont(mono);
    layout->addWidget(body_vel_label_);
  } else {
    body_vel_label_ = nullptr;
  }
}

void AgentStatusWidget::updateFromOdom(const nav_msgs::msg::Odometry & odom)
{
  auto & p = odom.pose.pose.position;
  auto & v = odom.twist.twist.linear;
  auto & q = odom.pose.pose.orientation;

  pos_label_->setText(QString("Pos:   %1 %2 %3 [m]")
    .arg(p.x, 7, 'f', 3).arg(p.y, 7, 'f', 3).arg(p.z, 7, 'f', 3));

  vel_label_->setText(QString("Vel:   %1 %2 %3 [m/s]")
    .arg(v.x, 7, 'f', 3).arg(v.y, 7, 'f', 3).arg(v.z, 7, 'f', 3));

  // Quaternion to euler conversion
  double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
  double roll = std::atan2(sinr_cosp, cosr_cosp);

  double sinp = 2.0 * (q.w * q.y - q.z * q.x);
  sinp = std::clamp(sinp, -1.0, 1.0);
  double pitch = std::asin(sinp);

  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  double yaw = std::atan2(siny_cosp, cosy_cosp);

  if (is_drone_) {
    // Drone: show roll, pitch, yaw in degrees
    double r_deg = roll * 180.0 / M_PI;
    double p_deg = pitch * 180.0 / M_PI;
    double y_deg = yaw * 180.0 / M_PI;
    orient_label_->setText(QString("RPY:   %1 %2 %3 [deg]")
      .arg(r_deg, 7, 'f', 2).arg(p_deg, 7, 'f', 2).arg(y_deg, 7, 'f', 2));
  } else {
    // Rover: show theta in degrees
    double theta_deg = yaw * 180.0 / M_PI;
    orient_label_->setText(QString("Theta: %1 [deg]").arg(theta_deg, 7, 'f', 2));
  }
}

void AgentStatusWidget::updateFromVelRaw(const geometry_msgs::msg::Twist & twist)
{
  if (body_vel_label_) {
    body_vel_label_->setText(QString("Body:  vx=%1 vy=%2 wz=%3")
      .arg(twist.linear.x, 6, 'f', 3)
      .arg(twist.linear.y, 6, 'f', 3)
      .arg(twist.angular.z, 6, 'f', 3));
  }
}

void AgentStatusWidget::setConnected(bool connected)
{
  connected_ = connected;
  indicator_->setStyleSheet(
    connected ? "color: #00FF00; font-size: 14px;" : "color: #FF0000; font-size: 14px;");
}

// ─── LandingPanel ──────────────────────────────────────────────────────────

LandingPanel::LandingPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  setupUI();
}

LandingPanel::~LandingPanel()
{
  if (record_process_ && record_process_->state() != QProcess::NotRunning) {
    record_process_->terminate();
    record_process_->waitForFinished(3000);
  }
}

void LandingPanel::setupUI()
{
  auto * scroll = new QScrollArea(this);
  scroll->setWidgetResizable(true);
  auto * container = new QWidget();
  auto * main_layout = new QVBoxLayout(container);
  main_layout->setSpacing(8);

  // ── System Status ──
  auto * status_group = new QGroupBox("SYSTEM STATUS");
  status_group->setStyleSheet("QGroupBox { font-weight: bold; }");
  auto * status_layout = new QHBoxLayout(status_group);

  auto * status_label = new QLabel("Status:");
  status_value_ = new QLabel("DISCONNECTED");
  status_value_->setStyleSheet("color: #666; font-weight: bold;");
  status_layout->addWidget(status_label);
  status_layout->addWidget(status_value_);
  status_layout->addStretch();

  auto * policy_label = new QLabel("Policy:");
  policy_value_ = new QLabel("NOT READY");
  policy_value_->setStyleSheet("color: #f44336; font-weight: bold;");
  status_layout->addWidget(policy_label);
  status_layout->addWidget(policy_value_);

  main_layout->addWidget(status_group);

  // ── Landing Info ──
  auto * info_group = new QGroupBox("LANDING INFO");
  info_group->setStyleSheet("QGroupBox { font-weight: bold; }");
  auto * info_layout = new QVBoxLayout(info_group);

  QFont mono("monospace", 9);
  distance_value_ = new QLabel("Distance:  ---  [m]");
  distance_value_->setFont(mono);
  rel_pos_value_ = new QLabel("Rel Pos:   --- --- --- [m]");
  rel_pos_value_->setFont(mono);
  info_layout->addWidget(distance_value_);
  info_layout->addWidget(rel_pos_value_);

  main_layout->addWidget(info_group);

  // ── Control Buttons ──
  auto * control_group = new QGroupBox("CONTROL");
  control_group->setStyleSheet("QGroupBox { font-weight: bold; }");
  auto * btn_layout = new QHBoxLayout(control_group);

  takeoff_btn_ = new QPushButton("Takeoff");
  takeoff_btn_->setMinimumHeight(40);
  takeoff_btn_->setStyleSheet(
    "QPushButton { background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px; }"
    "QPushButton:disabled { background-color: #555; color: #999; }");
  takeoff_btn_->setEnabled(false);

  run_btn_ = new QPushButton("Run");
  run_btn_->setMinimumHeight(40);
  run_btn_->setStyleSheet(
    "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 4px; }"
    "QPushButton:disabled { background-color: #555; color: #999; }");
  run_btn_->setEnabled(false);

  abort_btn_ = new QPushButton("Abort");
  abort_btn_->setMinimumHeight(40);
  abort_btn_->setStyleSheet(
    "QPushButton { background-color: #FF9800; color: white; font-weight: bold; border-radius: 4px; }"
    "QPushButton:disabled { background-color: #555; color: #999; }");
  abort_btn_->setEnabled(false);

  off_btn_ = new QPushButton("Off");
  off_btn_->setMinimumHeight(40);
  off_btn_->setStyleSheet(
    "QPushButton { background-color: #f44336; color: white; font-weight: bold; border-radius: 4px; }"
    "QPushButton:disabled { background-color: #555; color: #999; }");
  off_btn_->setEnabled(false);

  btn_layout->addWidget(takeoff_btn_);
  btn_layout->addWidget(run_btn_);
  btn_layout->addWidget(abort_btn_);
  btn_layout->addWidget(off_btn_);

  connect(takeoff_btn_, &QPushButton::clicked, this, &LandingPanel::onTakeoffClicked);
  connect(run_btn_, &QPushButton::clicked, this, &LandingPanel::onRunClicked);
  connect(abort_btn_, &QPushButton::clicked, this, &LandingPanel::onAbortClicked);
  connect(off_btn_, &QPushButton::clicked, this, &LandingPanel::onOffClicked);

  main_layout->addWidget(control_group);

  // ── Recording ──
  auto * record_group = new QGroupBox("RECORDING");
  record_group->setStyleSheet("QGroupBox { font-weight: bold; }");
  auto * record_layout = new QHBoxLayout(record_group);

  record_btn_ = new QPushButton("Record Rosbag");
  record_btn_->setCheckable(true);
  record_btn_->setMinimumHeight(35);
  record_btn_->setStyleSheet(
    "QPushButton { background-color: #607D8B; color: white; font-weight: bold; border-radius: 4px; }"
    "QPushButton:checked { background-color: #E91E63; }");
  connect(record_btn_, &QPushButton::clicked, this, &LandingPanel::onRecordClicked);
  record_layout->addWidget(record_btn_);

  main_layout->addWidget(record_group);

  // ── Reset Odom ──
  auto * reset_group = new QGroupBox("ROVER ODOM");
  reset_group->setStyleSheet("QGroupBox { font-weight: bold; }");
  auto * reset_layout = new QHBoxLayout(reset_group);

  reset_odom_btn_ = new QPushButton("Reset Odom to (0,0,0)");
  reset_odom_btn_->setMinimumHeight(35);
  reset_odom_btn_->setStyleSheet(
    "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; border-radius: 4px; }");
  connect(reset_odom_btn_, &QPushButton::clicked, this, &LandingPanel::onResetOdomClicked);
  reset_layout->addWidget(reset_odom_btn_);

  main_layout->addWidget(reset_group);

  // ── Drone Status ──
  auto * drone_group = new QGroupBox("DRONE");
  drone_group->setStyleSheet(
    "QGroupBox { font-weight: bold; color: #2196F3; }");
  auto * drone_layout = new QVBoxLayout(drone_group);
  drone_widget_ = new AgentStatusWidget("cf_1", true);
  drone_layout->addWidget(drone_widget_);
  main_layout->addWidget(drone_group);

  // ── Rover Status ──
  auto * rover_group = new QGroupBox("ROVER (X3)");
  rover_group->setStyleSheet(
    "QGroupBox { font-weight: bold; color: #f44336; }");
  auto * rover_layout = new QVBoxLayout(rover_group);
  rover_widget_ = new AgentStatusWidget("rover", false);
  rover_layout->addWidget(rover_widget_);
  main_layout->addWidget(rover_group);

  main_layout->addStretch();

  scroll->setWidget(container);
  auto * outer = new QVBoxLayout(this);
  outer->setContentsMargins(0, 0, 0, 0);
  outer->addWidget(scroll);

  // UI update timer
  ui_timer_ = new QTimer(this);
  connect(ui_timer_, &QTimer::timeout, this, &LandingPanel::updateUI);
}

void LandingPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  // Subscribe to mission status
  auto qos = rclcpp::QoS(10).best_effort();
  status_sub_ = node_->create_subscription<cf_landing_interfaces::msg::MissionStatus>(
    "/cf_landing/mission_status", qos,
    [this](cf_landing_interfaces::msg::MissionStatus::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_status_ = msg;
    });

  // Subscribe to drone odom
  drone_odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
    "/cf_1/odom", qos,
    [this](nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_drone_odom_ = msg;
      last_drone_odom_time_ = node_->now();
      drone_connected_ = true;
    });

  // Subscribe to rover odom (both /rover/odom for sim and /odom for hw)
  auto rover_odom_cb = [this](nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_rover_odom_ = msg;
    last_rover_odom_time_ = node_->now();
    rover_connected_ = true;
  };
  rover_odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
    "/rover/odom", qos, rover_odom_cb);
  // Also subscribe to /odom and /x3/odom for hw mode
  node_->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", qos, rover_odom_cb);
  node_->create_subscription<nav_msgs::msg::Odometry>(
    "/x3/odom", qos, rover_odom_cb);

  // Subscribe to rover body velocities (both /vel_raw for sim and /x3/vel_raw for hw)
  auto vel_cb = [this](geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_rover_vel_ = msg;
  };
  rover_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
    "/vel_raw", qos, vel_cb);
  node_->create_subscription<geometry_msgs::msg::Twist>(
    "/x3/vel_raw", qos, vel_cb);

  // TF buffer for global position lookup
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Command service client
  command_client_ = node_->create_client<cf_landing_interfaces::srv::Command>(
    "/cf_landing/command");

  ui_timer_->start(50);  // 20 Hz
}

void LandingPanel::updateUI()
{
  std::lock_guard<std::mutex> lock(mutex_);

  // Check connection timeouts (1 second)
  if (drone_connected_ && node_) {
    auto dt = (node_->now() - last_drone_odom_time_).seconds();
    if (dt > 1.0) drone_connected_ = false;
  }
  if (rover_connected_ && node_) {
    auto dt = (node_->now() - last_rover_odom_time_).seconds();
    if (dt > 1.0) rover_connected_ = false;
  }

  // Update mission status
  using MS = cf_landing_interfaces::msg::MissionStatus;
  if (latest_status_) {
    uint8_t s = latest_status_->status;
    QString text;
    QString color;
    switch (s) {
      case MS::STATUS_OFF:     text = "OFF";     color = "#666";    break;
      case MS::STATUS_TAKEOFF: text = "TAKEOFF"; color = "#2196F3"; break;
      case MS::STATUS_HOVER:   text = "HOVER";   color = "#FF9800"; break;
      case MS::STATUS_RUN:     text = "RUNNING"; color = "#4CAF50"; break;
      case MS::STATUS_LANDED:  text = "LANDED";  color = "#4CAF50"; break;
      case MS::STATUS_ABORT:   text = "ABORT";   color = "#f44336"; break;
      default:                 text = "UNKNOWN"; color = "#666";    break;
    }
    status_value_->setText(text);
    status_value_->setStyleSheet(QString("color: %1; font-weight: bold;").arg(color));

    // Policy readiness
    bool policies_ready = latest_status_->drone_policy_ready && latest_status_->rover_policy_ready;
    if (policies_ready) {
      policy_value_->setText("READY");
      policy_value_->setStyleSheet("color: #4CAF50; font-weight: bold;");
    } else {
      QString waiting;
      if (!latest_status_->drone_policy_ready && !latest_status_->rover_policy_ready)
        waiting = "NOT READY (drone, rover)";
      else if (!latest_status_->drone_policy_ready)
        waiting = "NOT READY (drone)";
      else
        waiting = "NOT READY (rover)";
      policy_value_->setText(waiting);
      policy_value_->setStyleSheet("color: #f44336; font-weight: bold;");
    }

    // Button enablement
    takeoff_btn_->setEnabled(s == MS::STATUS_OFF && policies_ready);
    run_btn_->setEnabled(s == MS::STATUS_HOVER);
    abort_btn_->setEnabled(s == MS::STATUS_RUN || s == MS::STATUS_TAKEOFF || s == MS::STATUS_HOVER);
    off_btn_->setEnabled(s != MS::STATUS_OFF);
  } else {
    status_value_->setText("DISCONNECTED");
    status_value_->setStyleSheet("color: #666; font-weight: bold;");
    policy_value_->setText("NOT READY");
    policy_value_->setStyleSheet("color: #f44336; font-weight: bold;");
    takeoff_btn_->setEnabled(false);
    run_btn_->setEnabled(false);
    abort_btn_->setEnabled(false);
    off_btn_->setEnabled(false);
  }

  // Update drone state
  drone_widget_->setConnected(drone_connected_);
  if (latest_drone_odom_) {
    drone_widget_->updateFromOdom(*latest_drone_odom_);
  }

  // Update rover state — try TF lookup for global position, fallback to odom
  rover_widget_->setConnected(rover_connected_);
  nav_msgs::msg::Odometry rover_global;
  bool have_rover_global = false;
  try {
    auto t = tf_buffer_->lookupTransform("world", "base_link", tf2::TimePointZero);
    rover_global.pose.pose.position.x = t.transform.translation.x;
    rover_global.pose.pose.position.y = t.transform.translation.y;
    rover_global.pose.pose.position.z = t.transform.translation.z;
    rover_global.pose.pose.orientation = t.transform.rotation;
    if (latest_rover_odom_) {
      rover_global.twist = latest_rover_odom_->twist;
    }
    have_rover_global = true;
    rover_widget_->updateFromOdom(rover_global);
  } catch (...) {
    if (latest_rover_odom_) {
      rover_widget_->updateFromOdom(*latest_rover_odom_);
      have_rover_global = true;
      rover_global = *latest_rover_odom_;
    }
  }
  if (latest_rover_vel_) {
    rover_widget_->updateFromVelRaw(*latest_rover_vel_);
  }

  // Update landing info (relative position to pad center, distance)
  if (latest_drone_odom_ && have_rover_global) {
    // Pad offset in rover body frame
    constexpr double pad_off_x = -0.0384;
    constexpr double pad_off_y = 0.0004;
    constexpr double rover_height = 0.213;  // pad surface above ground

    // Rover yaw from quaternion
    auto & rq = rover_global.pose.pose.orientation;
    double rover_yaw = std::atan2(2.0 * (rq.w * rq.z + rq.x * rq.y),
                                   1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z));
    double cr = std::cos(rover_yaw), sr = std::sin(rover_yaw);

    // Pad center in world frame
    double pad_x = rover_global.pose.pose.position.x + pad_off_x * cr - pad_off_y * sr;
    double pad_y = rover_global.pose.pose.position.y + pad_off_x * sr + pad_off_y * cr;

    double dx = latest_drone_odom_->pose.pose.position.x - pad_x;
    double dy = latest_drone_odom_->pose.pose.position.y - pad_y;
    double dz = latest_drone_odom_->pose.pose.position.z - rover_height;
    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    distance_value_->setText(QString("Distance:  %1  [m]").arg(dist, 6, 'f', 3));
    rel_pos_value_->setText(QString("Rel Pos:   %1 %2 %3 [m]")
      .arg(dx, 7, 'f', 3).arg(dy, 7, 'f', 3).arg(dz, 7, 'f', 3));
  }
}

// ── Button handlers ──

void LandingPanel::onTakeoffClicked()
{
  callCommand(cf_landing_interfaces::srv::Command::Request::CMD_TAKEOFF);
}

void LandingPanel::onRunClicked()
{
  callCommand(cf_landing_interfaces::srv::Command::Request::CMD_RUN);
}

void LandingPanel::onAbortClicked()
{
  callCommand(cf_landing_interfaces::srv::Command::Request::CMD_ABORT);
}

void LandingPanel::onOffClicked()
{
  callCommand(cf_landing_interfaces::srv::Command::Request::CMD_OFF);
}

void LandingPanel::callCommand(uint8_t cmd)
{
  if (!command_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(node_->get_logger(), "Command service not available");
    return;
  }
  auto request = std::make_shared<cf_landing_interfaces::srv::Command::Request>();
  request->command = cmd;
  command_client_->async_send_request(request);
}

// ── Reset Odom ──

void LandingPanel::onResetOdomClicked()
{
  // Call /set_pose service on robot_localization EKF to reset to (0,0,0)
  auto client = node_->create_client<robot_localization::srv::SetPose>("/x3/set_pose");
  if (!client->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(node_->get_logger(), "set_pose service not available");
    return;
  }
  auto request = std::make_shared<robot_localization::srv::SetPose::Request>();
  request->pose.header.frame_id = "odom";
  request->pose.header.stamp = node_->now();
  request->pose.pose.pose.position.x = 0.0;
  request->pose.pose.pose.position.y = 0.0;
  request->pose.pose.pose.position.z = 0.0;
  request->pose.pose.pose.orientation.w = 1.0;
  client->async_send_request(request);
  RCLCPP_INFO(node_->get_logger(), "Reset rover odom to (0,0,0)");
}

// ── Recording ──

void LandingPanel::onRecordClicked()
{
  if (record_btn_->isChecked()) {
    // Start recording
    QString timestamp = QDateTime::currentDateTime().toString("yyyy_MM_dd-HH_mm_ss");
    QString log_dir = QDir::homePath() + "/crazyflie-rover-landing/hardware/logs";
    QDir().mkpath(log_dir);
    QString bag_path = log_dir + "/rosbag_" + timestamp;

    record_process_ = new QProcess(this);
    connect(record_process_,
      QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
      this, &LandingPanel::onRecordProcessFinished);

    record_process_->start("ros2", QStringList() << "bag" << "record" << "-a" << "-o" << bag_path);
    record_btn_->setText("Stop Recording");

    // Brief disable to prevent double-click
    record_btn_->setEnabled(false);
    QTimer::singleShot(500, [this]() { record_btn_->setEnabled(true); });
  } else {
    // Stop recording
    if (record_process_ && record_process_->state() != QProcess::NotRunning) {
      record_process_->terminate();
    }
    record_btn_->setText("Record Rosbag");
  }
}

void LandingPanel::onRecordProcessFinished(int, QProcess::ExitStatus)
{
  record_btn_->setChecked(false);
  record_btn_->setText("Record Rosbag");
  if (record_process_) {
    record_process_->deleteLater();
    record_process_ = nullptr;
  }
}

}  // namespace cf_landing_rviz_plugin

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(cf_landing_rviz_plugin::LandingPanel, rviz_common::Panel)
