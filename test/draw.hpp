#ifndef DRAW_HPP
#define DRAW_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <utility>
#include "../include/armor.hpp"
#include "tracker.hpp"
#include "pnp_solver.hpp"
// Forward declaration to avoid heavy include; defined in tasks/aimer.hpp
struct AimPoint;

using namespace armor_task;

namespace tools {
// 基础绘图工具函数
void draw_point(cv::Mat & img, const cv::Point & point, const cv::Scalar & color, int radius = 2);
void draw_points(cv::Mat & img, const std::vector<cv::Point> & points, const cv::Scalar & color = cv::Scalar(0, 0, 255), int thickness = 2);
void draw_points(cv::Mat & img, const std::vector<cv::Point2f> & points, const cv::Scalar & color = cv::Scalar(0, 0, 255), int thickness = 2);
void draw_text(cv::Mat & img, const std::string & text, const cv::Point & point, const cv::Scalar & color, double font_scale = 0.5, int thickness = 1);

// 辅助函数
std::vector<cv::Point2f> project3DPointsTo2D(const std::vector<cv::Point3f> &points_3d, const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs);
void draw_info_box(cv::Mat &image, const std::string &text, int font_scale = 1, int thickness = 2);
void plotArmors(const ArmorArray &armors, cv::Mat &img);
void plotSingleArmor(const Armor &target, const Armor &armor, cv::Mat &img);
} // namespace tools

// 从YAML配置文件加载相机参数
std::pair<cv::Mat, cv::Mat> loadCameraParameters(const std::string& config_path);

// 主要绘图函数
void drawArmorDetection(cv::Mat& img, const ArmorArray& armors);
void drawTargetInfo(cv::Mat& img, const std::vector<Target>& targets, const std::string& tracker_state, const PnpSolver& pnp_solver);
void drawPerformanceInfo(cv::Mat& img, double fps, double detect_time, double track_time);
// 绘制弹道轨迹（基于 AimPoint）
void drawTrajectory(cv::Mat &img, const AimPoint &aim_point, double bullet_speed,
                    const std::string &config_path, const cv::Mat &camera_matrix,
                    const cv::Mat &distort_coeffs);

#endif // DRAW_HPP
