// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/F9pRtk.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__F9P_RTK__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__F9P_RTK__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/f9p_rtk__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_F9pRtk_acc_d
{
public:
  explicit Init_F9pRtk_acc_d(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::F9pRtk acc_d(::px4_msgs::msg::F9pRtk::_acc_d_type arg)
  {
    msg_.acc_d = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_acc_e
{
public:
  explicit Init_F9pRtk_acc_e(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_acc_d acc_e(::px4_msgs::msg::F9pRtk::_acc_e_type arg)
  {
    msg_.acc_e = std::move(arg);
    return Init_F9pRtk_acc_d(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_acc_n
{
public:
  explicit Init_F9pRtk_acc_n(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_acc_e acc_n(::px4_msgs::msg::F9pRtk::_acc_n_type arg)
  {
    msg_.acc_n = std::move(arg);
    return Init_F9pRtk_acc_e(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_v_d
{
public:
  explicit Init_F9pRtk_v_d(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_acc_n v_d(::px4_msgs::msg::F9pRtk::_v_d_type arg)
  {
    msg_.v_d = std::move(arg);
    return Init_F9pRtk_acc_n(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_v_e
{
public:
  explicit Init_F9pRtk_v_e(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_v_d v_e(::px4_msgs::msg::F9pRtk::_v_e_type arg)
  {
    msg_.v_e = std::move(arg);
    return Init_F9pRtk_v_d(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_v_n
{
public:
  explicit Init_F9pRtk_v_n(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_v_e v_n(::px4_msgs::msg::F9pRtk::_v_n_type arg)
  {
    msg_.v_n = std::move(arg);
    return Init_F9pRtk_v_e(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_d
{
public:
  explicit Init_F9pRtk_d(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_v_n d(::px4_msgs::msg::F9pRtk::_d_type arg)
  {
    msg_.d = std::move(arg);
    return Init_F9pRtk_v_n(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_e
{
public:
  explicit Init_F9pRtk_e(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_d e(::px4_msgs::msg::F9pRtk::_e_type arg)
  {
    msg_.e = std::move(arg);
    return Init_F9pRtk_d(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_n
{
public:
  explicit Init_F9pRtk_n(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_e n(::px4_msgs::msg::F9pRtk::_n_type arg)
  {
    msg_.n = std::move(arg);
    return Init_F9pRtk_e(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_satellites_used
{
public:
  explicit Init_F9pRtk_satellites_used(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_n satellites_used(::px4_msgs::msg::F9pRtk::_satellites_used_type arg)
  {
    msg_.satellites_used = std::move(arg);
    return Init_F9pRtk_n(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_fix_type
{
public:
  explicit Init_F9pRtk_fix_type(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_satellites_used fix_type(::px4_msgs::msg::F9pRtk::_fix_type_type arg)
  {
    msg_.fix_type = std::move(arg);
    return Init_F9pRtk_satellites_used(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_age_corr
{
public:
  explicit Init_F9pRtk_age_corr(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_fix_type age_corr(::px4_msgs::msg::F9pRtk::_age_corr_type arg)
  {
    msg_.age_corr = std::move(arg);
    return Init_F9pRtk_fix_type(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_tow
{
public:
  explicit Init_F9pRtk_tow(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_age_corr tow(::px4_msgs::msg::F9pRtk::_tow_type arg)
  {
    msg_.tow = std::move(arg);
    return Init_F9pRtk_age_corr(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_device_id
{
public:
  explicit Init_F9pRtk_device_id(::px4_msgs::msg::F9pRtk & msg)
  : msg_(msg)
  {}
  Init_F9pRtk_tow device_id(::px4_msgs::msg::F9pRtk::_device_id_type arg)
  {
    msg_.device_id = std::move(arg);
    return Init_F9pRtk_tow(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

class Init_F9pRtk_timestamp
{
public:
  Init_F9pRtk_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_F9pRtk_device_id timestamp(::px4_msgs::msg::F9pRtk::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_F9pRtk_device_id(msg_);
  }

private:
  ::px4_msgs::msg::F9pRtk msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::F9pRtk>()
{
  return px4_msgs::msg::builder::Init_F9pRtk_timestamp();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__F9P_RTK__BUILDER_HPP_
