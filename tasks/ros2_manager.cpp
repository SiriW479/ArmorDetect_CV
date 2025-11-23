#include "ros2_manager.hpp"
#include "detector.hpp"
#include "pnp_solver.hpp"
#include "ros2_manager.hpp"
#include "tracker.hpp"

ROS2Manager::ROS2Manager() : Node("ros2_manager"), new_frame_ready_(false)
{
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 10, std::bind(&ROS2Manager::image_callback, this, std::placeholders::_1));

    // armor_pub_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("/armor/prediction", 10);

    RCLCPP_INFO(this->get_logger(), "ROS2Manager initialized.");
}

void ROS2Manager::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try
    {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        std::lock_guard<std::mutex> lock(img_mutex_);
        current_frame_ = frame.clone();
        new_frame_ready_ = true;
        // RCLCPP_INFO(this->get_logger(), "Image Received.");
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    }
}

bool ROS2Manager::get_img(cv::Mat &img)
{
    if (!new_frame_ready_) return false;

    std::lock_guard<std::mutex> lock(img_mutex_);
    img = current_frame_.clone();
    new_frame_ready_ = false;
    return true;
}

// void ROS2Manager::publish_armor_result(ArmorArray &armor_list)
// {
//     armor_interfaces::msg::ArmorArray msg_out;
//     for (const auto &armor : armor_list)
//     {
//         armor_interfaces::msg::Armor a;
//         a.confidence = armor.confidence;
//         a.detect_id = armor.detect_id;
//         a.car_num = armor.car_num;
//         a.yaw = armor.yaw;
//         a.p_camera.x = armor.p_camera.x;
//         a.p_camera.y = armor.p_camera.y;
//         a.p_camera.z = armor.p_camera.z;
//         a.cloar = armor.color;
//         msg_out.armors.push_back(a);
//     }
//     armor_pub_->publish(msg_out);
// }
