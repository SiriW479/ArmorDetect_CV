#pragma once

#include "armor.hpp" // 你定义的自定义消息
// #include "armor_interfaces/msg/armor.hpp"
// #include "armor_interfaces/msg/armor_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class ROS2Manager : public rclcpp::Node
{
  public:
    ROS2Manager();

    // 从图像订阅中获取当前图像帧（深拷贝）
    bool get_img(cv::Mat &img);

    // // 向话题发布装甲板信息
    // void publish_armor_result(const ArmorArray &armor_list);

  private:
    // 图像订阅回调
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    // 成员变量
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    // rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr armor_pub_;

    std::mutex img_mutex_;
    cv::Mat current_frame_;
    std::atomic<bool> new_frame_ready_;
};