#include "draw.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <cmath>
#include "../../tasks/aimer.hpp"
#include "../../tasks/trajectory_normal.hpp"

namespace tools {

void draw_point(cv::Mat & img, const cv::Point & point, const cv::Scalar & color, int radius)
{
    cv::circle(img, point, radius, color, -1);
}

void draw_points(
    cv::Mat & img, const std::vector<cv::Point> & points, const cv::Scalar & color, int thickness)
{
    std::vector<std::vector<cv::Point>> contours = {points};
    cv::drawContours(img, contours, -1, color, thickness);
}

void draw_points(
    cv::Mat & img, const std::vector<cv::Point2f> & points, const cv::Scalar & color, int thickness)
{
    std::vector<cv::Point> int_points(points.begin(), points.end());
    draw_points(img, int_points, color, thickness);
}

void draw_text(
    cv::Mat & img, const std::string & text, const cv::Point & point, const cv::Scalar & color,
    double font_scale, int thickness)
{
    cv::putText(img, text, point, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

std::vector<cv::Point2f> project3DPointsTo2D(const std::vector<cv::Point3f> &points_3d, const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs)
{
    std::vector<cv::Point2f> points_2d;
    cv::projectPoints(points_3d, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix, distort_coeffs, points_2d);
    return points_2d;
}

void draw_info_box(cv::Mat &image, const std::string &text, int font_scale, int thickness)
{
    // 1. 拆分多行
    std::vector<std::string> lines;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line, '\n'))
    {
        lines.push_back(line);
    }

    // 2. 字体参数
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    int baseline = 0;
    int line_height = 0;
    int max_width = 0;

    // 3. 获取最大宽度和总高度
    for (const auto &l : lines)
    {
        cv::Size text_size = cv::getTextSize(l, font_face, font_scale, thickness, &baseline);
        line_height = std::max(line_height, text_size.height + baseline);
        max_width = std::max(max_width, text_size.width);
    }

    int padding = 10;
    int box_width = max_width + padding * 2;
    int box_height = line_height * lines.size() + padding * 2;

    // 4. 设置背景位置（右下角）
    int x = image.cols - box_width - 20;
    int y = image.rows - box_height - 20;

    // 5. 绘制背景矩形（半透明黑）
    cv::Mat roi = image(cv::Rect(x, y, box_width, box_height));
    cv::Mat overlay;
    roi.copyTo(overlay);
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(box_width, box_height), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);

    // 6. 绘制文字
    for (size_t i = 0; i < lines.size(); ++i)
    {
        cv::putText(image, lines[i], cv::Point(x + padding, y + padding + line_height * (i + 1) - baseline),
                   font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
    }
}

void plotArmors(const ArmorArray &armors, cv::Mat &img)
{
    if (armors.empty())
    {
        cv::imshow("Armors", img);
        cv::waitKey(1);
        return;
    }

    for (const auto &armor : armors)
    {
        // 绘制检测框
        cv::rectangle(img, armor.box, cv::Scalar(0, 255, 0), 2);

        // 绘制中心点
        draw_point(img, armor.center, cv::Scalar(0, 0, 255), 4);

        // 绘制灯条位置
        draw_point(img, armor.left_lightbar.top, cv::Scalar(255, 0, 0), 3);
        draw_point(img, armor.left_lightbar.bottom, cv::Scalar(255, 0, 0), 3);
        draw_point(img, armor.right_lightbar.top, cv::Scalar(0, 0, 255), 3);
        draw_point(img, armor.right_lightbar.bottom, cv::Scalar(0, 0, 255), 3);

        // 装甲板信息
        std::stringstream ss;
        ss << "ID:" << armor.car_num << " Conf:" << std::fixed << std::setprecision(2) << armor.confidence;
        std::string color_str = (armor.color == Color::red ? "RED" : armor.color == Color::blue ? "BLUE" : "PURPLE");
        ss << " " << color_str;
        ss << " Pos:[" << std::fixed << std::setprecision(1) << armor.p_camera[0] << "," 
           << armor.p_camera[1] << "," << armor.p_camera[2] << "]";
        ss << " Yaw:" << std::fixed << std::setprecision(1) << armor.yaw;

        draw_text(img, ss.str(), cv::Point(armor.box.x, std::max(0, armor.box.y - 10)),
                 cv::Scalar(0, 255, 255), 0.5, 1);
    }

    cv::imshow("Armors", img);
    cv::waitKey(1);
}

void plotSingleArmor(const Armor &target, const Armor &armor, cv::Mat &img)
{
    if (armor.detect_id == -1)
    {
        cv::imshow("SingleArmor", img);
        cv::waitKey(1);
        return;
    }

    // 绘制检测框
    cv::rectangle(img, armor.box, cv::Scalar(0, 255, 0), 2);

    // 绘制中心点
    draw_point(img, target.center, cv::Scalar(0, 0, 255), 4);
    draw_point(img, armor.center, cv::Scalar(255, 0, 0), 4);

    // 绘制灯条
    draw_point(img, armor.left_lightbar.top, cv::Scalar(0, 255, 0), 3);
    draw_point(img, armor.left_lightbar.bottom, cv::Scalar(0, 255, 0), 3);
    draw_point(img, armor.right_lightbar.top, cv::Scalar(0, 255, 0), 3);
    draw_point(img, armor.right_lightbar.bottom, cv::Scalar(0, 255, 0), 3);

    // 装甲板信息
    std::stringstream ss;
    ss << "ID:" << armor.car_num << " Conf:" << std::fixed << std::setprecision(2) << armor.confidence;
    std::string color_str = (armor.color == Color::red ? "RED" : armor.color == Color::blue ? "BLUE" : "PURPLE");
    ss << " " << color_str;
    ss << " Pos:[" << std::fixed << std::setprecision(1) << armor.p_camera[0] << "," 
       << armor.p_camera[1] << "," << armor.p_camera[2] << "]";
    ss << " Yaw:" << std::fixed << std::setprecision(1) << armor.yaw;

    draw_text(img, ss.str(), cv::Point(armor.box.x, std::max(0, armor.box.y - 10)),
             cv::Scalar(0, 255, 255), 0.5, 1);

    cv::imshow("SingleArmor", img);
    cv::waitKey(1);
}

} // namespace tools

// ============================================================================
// 弹道辅助计算（匿名命名空间，仅本文件使用）
// ============================================================================
namespace {
constexpr double kGravity = 9.7833;

// 计算物理弹道在世界系下的轨迹点
std::vector<Eigen::Vector3d> calculateTrajectoryPoints(const Eigen::Vector3d &start_pos,
                                                       const Eigen::Vector3d &target_pos,
                                                       double bullet_speed,
                                                       int num_points = 50)
{
    std::vector<Eigen::Vector3d> trajectory_points;

    Eigen::Vector3d diff = target_pos - start_pos;
    double d = std::sqrt(diff.x() * diff.x() + diff.y() * diff.y());
    double h = diff.z();

    ::tools::Trajectory traj(bullet_speed, d, h);
    if (traj.unsolvable)
    {
        return trajectory_points;
    }

    double yaw = std::atan2(diff.y(), diff.x());
    double v0_x = bullet_speed * std::cos(traj.pitch) * std::cos(yaw);
    double v0_y = bullet_speed * std::cos(traj.pitch) * std::sin(yaw);
    double v0_z = bullet_speed * std::sin(traj.pitch);

    double dt = traj.fly_time / static_cast<double>(num_points);
    for (int i = 0; i <= num_points; ++i)
    {
        double t = i * dt;
        if (t > traj.fly_time) t = traj.fly_time;

        double x = start_pos.x() + v0_x * t;
        double y = start_pos.y() + v0_y * t;
        double z = start_pos.z() + v0_z * t - 0.5 * kGravity * t * t;
        trajectory_points.emplace_back(x, y, z);
    }

    return trajectory_points;
}

// 加载外参矩阵（相机->云台，云台->世界）
void loadTransformMatrices(const std::string &config_path, Eigen::Matrix3d &R_camera2gimbal,
                           Eigen::Vector3d &t_camera2gimbal, Eigen::Matrix3d &R_gimbal2world)
{
    YAML::Node yaml = YAML::LoadFile(config_path);

    auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
    auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();

    R_camera2gimbal = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
    t_camera2gimbal = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

    // 当前环境缺少 IMU，先假设云台坐标系与世界坐标系重合
    R_gimbal2world = Eigen::Matrix3d::Identity();
}

// 将世界系轨迹投影到图像
std::vector<cv::Point2f> projectTrajectoryToImage(const std::vector<Eigen::Vector3d> &world_points,
                                                  const Eigen::Matrix3d &R_camera2gimbal,
                                                  const Eigen::Vector3d &t_camera2gimbal,
                                                  const Eigen::Matrix3d &R_gimbal2world,
                                                  const cv::Mat &camera_matrix,
                                                  const cv::Mat &distort_coeffs)
{
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> camera_points;

    for (const auto &pt_world : world_points)
    {
        Eigen::Vector3d pt_camera = R_camera2gimbal.transpose() * (R_gimbal2world.transpose() * pt_world - t_camera2gimbal);
        if (pt_camera.z() > 0.1)
        {
            camera_points.emplace_back(static_cast<float>(pt_camera.x()), static_cast<float>(pt_camera.y()), static_cast<float>(pt_camera.z()));
        }
    }

    if (camera_points.empty())
    {
        return image_points;
    }

    cv::projectPoints(camera_points, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F),
                      camera_matrix, distort_coeffs, image_points);
    return image_points;
}
} // namespace

// 从YAML配置文件加载相机参数
std::pair<cv::Mat, cv::Mat> loadCameraParameters(const std::string& config_path)
{
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;
    
    try {
        YAML::Node config = YAML::LoadFile(config_path);

        if (config["camera_matrix"]) {
            std::vector<double> cam_params = config["camera_matrix"].as<std::vector<double>>();
            camera_matrix = (cv::Mat_<double>(3, 3) <<
                cam_params[0], cam_params[1], cam_params[2],
                cam_params[3], cam_params[4], cam_params[5],
                cam_params[6], cam_params[7], cam_params[8]);
        }

        if (config["distort_coeffs"]) {
            std::vector<double> dist_params = config["distort_coeffs"].as<std::vector<double>>();
            distort_coeffs = (cv::Mat_<double>(1, 5) <<
                dist_params[0], dist_params[1], dist_params[2],
                dist_params[3], dist_params[4]);
        }

        std::cout << "Camera parameters loaded from: " << config_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading camera parameters from " << config_path << ": " << e.what() << std::endl;
        camera_matrix = (cv::Mat_<double>(3, 3) << 610, 0, 320, 0, 613, 240, 0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
        std::cout << "Using default camera parameters" << std::endl;
    }
    
    return std::make_pair(camera_matrix, distort_coeffs);
}

// 绘制装甲板检测结果
void drawArmorDetection(cv::Mat& img, const ArmorArray& armors) {
    static bool camera_params_initialized = false;
    static cv::Mat camera_matrix;
    static cv::Mat distort_coeffs;

    if (!camera_params_initialized) {
        auto camera_params = loadCameraParameters("../config/demo.yaml");
        camera_matrix = camera_params.first;
        distort_coeffs = camera_params.second;
        camera_params_initialized = true;
        std::cout << "[Debug] drawArmorDetection loaded camera parameters once." << std::endl;
    }

    for (const auto& armor : armors) {
        // 绘制装甲板中心点
        tools::draw_point(img, armor.center, cv::Scalar(0, 0, 255), 3);
        
        // 显示3D位置信息（如果已解算）
        if (armor.p_camera != Eigen::Vector3d::Zero()) {
            std::string pos_info = "Pos:(" + 
                                  std::to_string((int)armor.p_camera[0]) + "," +
                                  std::to_string((int)armor.p_camera[1]) + "," +
                                  std::to_string((int)armor.p_camera[2]) + ")mm";
            tools::draw_text(img, pos_info, 
                           cv::Point(armor.box.x, armor.box.y + armor.box.height + 15),
                           cv::Scalar(0, 255, 255), 0.4, 1);

            // 绘制装甲板角点
            if (!armor.corners.empty()) {
                tools::draw_points(img, armor.corners, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

// 绘制Target详细信息
void drawTargetInfo(cv::Mat& img, const std::vector<Target>& targets, const std::string& tracker_state, const PnpSolver& pnp_solver) {
    // 从demo.yaml中加载相机参数
    auto camera_params = loadCameraParameters("../config/demo.yaml");
    cv::Mat camera_matrix = camera_params.first;
    
    // 显示追踪器状态
    tools::draw_text(img, "Tracker State: " + tracker_state, 
                     cv::Point(10, 30), cv::Scalar(0, 255, 0), 0.7, 2);
    
    if (targets.empty()) {
        tools::draw_text(img, "No Target", cv::Point(10, 60), 
                        cv::Scalar(0, 0, 255), 0.6, 2);
        return;
    }
    
    const auto& target = targets.front();
    int y_offset = 60;
    int line_height = 20;
    
    // Target基本信息
    tools::draw_text(img, "=== Target Info ===", cv::Point(10, y_offset), 
                     cv::Scalar(255, 255, 255), 0.6, 2);
    y_offset += line_height;
    
    // 获取所有装甲板的预测位置和角度
    auto xyza_list = target.armor_xyza_list();
    
    // 遍历所有装甲板并重投影
    for (size_t i = 0; i < xyza_list.size(); ++i) {
        auto image_points = pnp_solver.reproject_armor(
            xyza_list[i].head<3>(), xyza_list[i][3], target.car_num, false);
        tools::draw_points(img, image_points, cv::Scalar(255, 0, 0), 2);  // 使用蓝色绘制预测装甲板
    }
    
    Eigen::VectorXd ekf_state = target.ekf_x();
    
    if (ekf_state.size() >= 5) {
        Eigen::Vector3d ekf_world(ekf_state[0], ekf_state[2], ekf_state[4]);
        Eigen::Matrix3d R_wc;
        R_wc << 0, -1,  0,
                0,  0, -1,
                1,  0,  0;
        Eigen::Vector3d ekf_camera = R_wc * ekf_world;
        
        if (ekf_camera[2] > 0) {
            cv::Point2f ekf_proj;
            ekf_proj.x = camera_matrix.at<double>(0, 0) * ekf_camera[0] / ekf_camera[2] + camera_matrix.at<double>(0, 2);
            ekf_proj.y = camera_matrix.at<double>(1, 1) * ekf_camera[1] / ekf_camera[2] + camera_matrix.at<double>(1, 2);
            
            if (ekf_proj.x >= 0 && ekf_proj.x < img.cols &&
                ekf_proj.y >= 0 && ekf_proj.y < img.rows) {
                tools::draw_point(img, ekf_proj, cv::Scalar(0, 0, 255), 5);
                tools::draw_text(img, "EKF", cv::Point(ekf_proj.x + 10, ekf_proj.y - 10),
                               cv::Scalar(0, 0, 255), 0.4, 1);
            }
        }
    }
    
    // 是否切换标志
    if (target.is_switch_) {
        tools::draw_text(img, "TARGET SWITCHED!", cv::Point(10, y_offset),
                        cv::Scalar(0, 0, 255), 0.6, 2);
        y_offset += line_height;
    }
    
    // 是否跳跃
    if (target.jumped) {
        tools::draw_text(img, "TARGET JUMPED!", cv::Point(10, y_offset),
                        cv::Scalar(0, 165, 255), 0.6, 2);
        y_offset += line_height;
    }
    
    // EKF状态信息
    tools::draw_text(img, "=== EKF State ===", cv::Point(10, y_offset),
                     cv::Scalar(255, 255, 255), 0.6, 2);
    y_offset += line_height;
    
    // 发散状态检查
    bool diverged = target.diverged();
    std::string diverge_status = diverged ? "DIVERGED!" : "Converged";
    cv::Scalar diverge_color = diverged ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    tools::draw_text(img, "Status: " + diverge_status, cv::Point(10, y_offset),
                     diverge_color, 0.5, 2);
    y_offset += line_height;
}

// 显示性能信息
void drawPerformanceInfo(cv::Mat& img, double fps, double detect_time, double track_time) {
    int x_pos = img.cols - 200;
    int y_offset = 30;
    int line_height = 20;
    
    // FPS信息
    std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 5);
    tools::draw_text(img, fps_text, cv::Point(x_pos, y_offset), 
                     cv::Scalar(0, 255, 0), 0.6, 2);
    y_offset += line_height;
    
    // 检测时间
    std::string detect_text = "Detect: " + std::to_string(detect_time).substr(0, 5) + "ms";
    tools::draw_text(img, detect_text, cv::Point(x_pos, y_offset), 
                     cv::Scalar(255, 255, 255), 0.5, 1);
    y_offset += line_height;
    
    // 追踪时间
    std::string track_text = "Track: " + std::to_string(track_time).substr(0, 5) + "ms";
    tools::draw_text(img, track_text, cv::Point(x_pos, y_offset), 
                     cv::Scalar(255, 255, 255), 0.5, 1);
}

void drawTrajectory(cv::Mat &img, const AimPoint &aim_point, double bullet_speed,
                    const std::string &config_path, const cv::Mat &camera_matrix,
                    const cv::Mat &distort_coeffs)
{
    if (!aim_point.valid) return;
    if (bullet_speed <= 0) bullet_speed = 23.0;

    Eigen::Matrix3d R_camera2gimbal;
    Eigen::Vector3d t_camera2gimbal;
    Eigen::Matrix3d R_gimbal2world;
    try
    {
        loadTransformMatrices(config_path, R_camera2gimbal, t_camera2gimbal, R_gimbal2world);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load transform matrices: " << e.what() << std::endl;
        return;
    }

    // 枪口位置（相机系固定偏移）
    Eigen::Vector3d muzzle_pos_cam(0.1, 0.1, 0.5);
    cv::Point3f muzzle_pos_cam_cv(static_cast<float>(muzzle_pos_cam.x()),
                                  static_cast<float>(muzzle_pos_cam.y()),
                                  static_cast<float>(muzzle_pos_cam.z()));
    std::vector<cv::Point3f> local_points = {muzzle_pos_cam_cv};
    std::vector<cv::Point2f> local_pixels;
    cv::projectPoints(local_points, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F),
                      camera_matrix, distort_coeffs, local_pixels);
    if (local_pixels.empty()) return;
    cv::Point2f fixed_start_pixel = local_pixels.front();

    // 计算物理弹道
    Eigen::Vector3d start_gimbal = R_camera2gimbal * muzzle_pos_cam + t_camera2gimbal;
    Eigen::Vector3d start_pos_world = R_gimbal2world * start_gimbal;
    Eigen::Vector3d target_pos_world = aim_point.xyza.head<3>();
    auto trajectory_points = calculateTrajectoryPoints(start_pos_world, target_pos_world, bullet_speed, 50);
    if (trajectory_points.empty()) return;

    // 投影到图像并绘制
    auto image_points = projectTrajectoryToImage(trajectory_points, R_camera2gimbal, t_camera2gimbal,
                                                 R_gimbal2world, camera_matrix, distort_coeffs);
    if (image_points.size() < 2) return;

    image_points[0] = fixed_start_pixel;

    if (fixed_start_pixel.x >= 0 && fixed_start_pixel.x < img.cols && fixed_start_pixel.y >= 0 && fixed_start_pixel.y < img.rows)
    {
        cv::circle(img, fixed_start_pixel, 6, cv::Scalar(0, 255, 0), 2);
    }

    cv::Scalar trajectory_color(0, 255, 255);
    for (size_t i = 0; i + 1 < image_points.size(); ++i)
    {
        bool p1_in = image_points[i].x >= 0 && image_points[i].x < img.cols && image_points[i].y >= 0 && image_points[i].y < img.rows;
        bool p2_in = image_points[i + 1].x >= 0 && image_points[i + 1].x < img.cols && image_points[i + 1].y >= 0 && image_points[i + 1].y < img.rows;
        if (p1_in && p2_in)
        {
            cv::line(img, image_points[i], image_points[i + 1], trajectory_color, 2, cv::LINE_AA);
        }
    }

    cv::circle(img, image_points.back(), 4, cv::Scalar(0, 0, 255), -1);
}

// 绘制Target详细信息 (deprecated)
[[deprecated("Use the other overload of drawTargetInfo instead")]]
void drawTargetInfo(cv::Mat& img, const std::vector<armor_task::Target>& targets, const std::string& state, const cv::Mat& /* unused */ camera_matrix) {
    int y_offset = 60;
    int line_height = 20;

    // 显示追踪状态
    std::string state_text = "State: " + state;
    cv::putText(img, state_text, cv::Point(10, y_offset),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    y_offset += line_height;

    // 显示目标数量
    std::string target_count = "Targets: " + std::to_string(targets.size());
    cv::putText(img, target_count, cv::Point(10, y_offset),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    y_offset += line_height;

    // 为每个目标绘制信息
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        Eigen::VectorXd ekf_x = target.ekf_x();

        // 显示目标位置
        std::string pos_text = "T" + std::to_string(i) + ": (" +
                              std::to_string((int)ekf_x[0]) + ", " +
                              std::to_string((int)ekf_x[2]) + ", " +
                              std::to_string((int)ekf_x[4]) + ")";
        cv::putText(img, pos_text, cv::Point(10, y_offset),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        y_offset += line_height;

        // 显示速度
        std::string vel_text = "V: (" +
                              std::to_string((int)ekf_x[1]) + ", " +
                              std::to_string((int)ekf_x[3]) + ", " +
                              std::to_string((int)ekf_x[5]) + ")";
        cv::putText(img, vel_text, cv::Point(10, y_offset),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        y_offset += line_height;
    }
}
