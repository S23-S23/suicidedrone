// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/Monitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__MONITORING__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__MONITORING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/monitoring__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_Monitoring_nav_state
{
public:
  explicit Init_Monitoring_nav_state(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::Monitoring nav_state(::px4_msgs::msg::Monitoring::_nav_state_type arg)
  {
    msg_.nav_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_rtk_d
{
public:
  explicit Init_Monitoring_rtk_d(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_nav_state rtk_d(::px4_msgs::msg::Monitoring::_rtk_d_type arg)
  {
    msg_.rtk_d = std::move(arg);
    return Init_Monitoring_nav_state(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_rtk_e
{
public:
  explicit Init_Monitoring_rtk_e(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_rtk_d rtk_e(::px4_msgs::msg::Monitoring::_rtk_e_type arg)
  {
    msg_.rtk_e = std::move(arg);
    return Init_Monitoring_rtk_d(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_rtk_n
{
public:
  explicit Init_Monitoring_rtk_n(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_rtk_e rtk_n(::px4_msgs::msg::Monitoring::_rtk_n_type arg)
  {
    msg_.rtk_n = std::move(arg);
    return Init_Monitoring_rtk_e(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_b
{
public:
  explicit Init_Monitoring_b(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_rtk_n b(::px4_msgs::msg::Monitoring::_b_type arg)
  {
    msg_.b = std::move(arg);
    return Init_Monitoring_rtk_n(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_g
{
public:
  explicit Init_Monitoring_g(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_b g(::px4_msgs::msg::Monitoring::_g_type arg)
  {
    msg_.g = std::move(arg);
    return Init_Monitoring_b(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_r
{
public:
  explicit Init_Monitoring_r(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_g r(::px4_msgs::msg::Monitoring::_r_type arg)
  {
    msg_.r = std::move(arg);
    return Init_Monitoring_g(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_battery
{
public:
  explicit Init_Monitoring_battery(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_r battery(::px4_msgs::msg::Monitoring::_battery_type arg)
  {
    msg_.battery = std::move(arg);
    return Init_Monitoring_r(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_rtk_nrover
{
public:
  explicit Init_Monitoring_rtk_nrover(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_battery rtk_nrover(::px4_msgs::msg::Monitoring::_rtk_nrover_type arg)
  {
    msg_.rtk_nrover = std::move(arg);
    return Init_Monitoring_battery(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_rtk_nbase
{
public:
  explicit Init_Monitoring_rtk_nbase(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_rtk_nrover rtk_nbase(::px4_msgs::msg::Monitoring::_rtk_nbase_type arg)
  {
    msg_.rtk_nbase = std::move(arg);
    return Init_Monitoring_rtk_nrover(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_status2
{
public:
  explicit Init_Monitoring_status2(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_rtk_nbase status2(::px4_msgs::msg::Monitoring::_status2_type arg)
  {
    msg_.status2 = std::move(arg);
    return Init_Monitoring_rtk_nbase(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_status1
{
public:
  explicit Init_Monitoring_status1(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_status2 status1(::px4_msgs::msg::Monitoring::_status1_type arg)
  {
    msg_.status1 = std::move(arg);
    return Init_Monitoring_status2(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_pitch
{
public:
  explicit Init_Monitoring_pitch(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_status1 pitch(::px4_msgs::msg::Monitoring::_pitch_type arg)
  {
    msg_.pitch = std::move(arg);
    return Init_Monitoring_status1(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_roll
{
public:
  explicit Init_Monitoring_roll(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_pitch roll(::px4_msgs::msg::Monitoring::_roll_type arg)
  {
    msg_.roll = std::move(arg);
    return Init_Monitoring_pitch(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_head
{
public:
  explicit Init_Monitoring_head(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_roll head(::px4_msgs::msg::Monitoring::_head_type arg)
  {
    msg_.head = std::move(arg);
    return Init_Monitoring_roll(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_ref_alt
{
public:
  explicit Init_Monitoring_ref_alt(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_head ref_alt(::px4_msgs::msg::Monitoring::_ref_alt_type arg)
  {
    msg_.ref_alt = std::move(arg);
    return Init_Monitoring_head(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_ref_lon
{
public:
  explicit Init_Monitoring_ref_lon(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_ref_alt ref_lon(::px4_msgs::msg::Monitoring::_ref_lon_type arg)
  {
    msg_.ref_lon = std::move(arg);
    return Init_Monitoring_ref_alt(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_ref_lat
{
public:
  explicit Init_Monitoring_ref_lat(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_ref_lon ref_lat(::px4_msgs::msg::Monitoring::_ref_lat_type arg)
  {
    msg_.ref_lat = std::move(arg);
    return Init_Monitoring_ref_lon(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_alt
{
public:
  explicit Init_Monitoring_alt(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_ref_lat alt(::px4_msgs::msg::Monitoring::_alt_type arg)
  {
    msg_.alt = std::move(arg);
    return Init_Monitoring_ref_lat(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_lon
{
public:
  explicit Init_Monitoring_lon(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_alt lon(::px4_msgs::msg::Monitoring::_lon_type arg)
  {
    msg_.lon = std::move(arg);
    return Init_Monitoring_alt(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_lat
{
public:
  explicit Init_Monitoring_lat(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_lon lat(::px4_msgs::msg::Monitoring::_lat_type arg)
  {
    msg_.lat = std::move(arg);
    return Init_Monitoring_lon(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_pos_z
{
public:
  explicit Init_Monitoring_pos_z(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_lat pos_z(::px4_msgs::msg::Monitoring::_pos_z_type arg)
  {
    msg_.pos_z = std::move(arg);
    return Init_Monitoring_lat(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_pos_y
{
public:
  explicit Init_Monitoring_pos_y(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_pos_z pos_y(::px4_msgs::msg::Monitoring::_pos_y_type arg)
  {
    msg_.pos_y = std::move(arg);
    return Init_Monitoring_pos_z(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_pos_x
{
public:
  explicit Init_Monitoring_pos_x(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_pos_y pos_x(::px4_msgs::msg::Monitoring::_pos_x_type arg)
  {
    msg_.pos_x = std::move(arg);
    return Init_Monitoring_pos_y(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_tow
{
public:
  explicit Init_Monitoring_tow(::px4_msgs::msg::Monitoring & msg)
  : msg_(msg)
  {}
  Init_Monitoring_pos_x tow(::px4_msgs::msg::Monitoring::_tow_type arg)
  {
    msg_.tow = std::move(arg);
    return Init_Monitoring_pos_x(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

class Init_Monitoring_timestamp
{
public:
  Init_Monitoring_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Monitoring_tow timestamp(::px4_msgs::msg::Monitoring::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_Monitoring_tow(msg_);
  }

private:
  ::px4_msgs::msg::Monitoring msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::Monitoring>()
{
  return px4_msgs::msg::builder::Init_Monitoring_timestamp();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__MONITORING__BUILDER_HPP_
