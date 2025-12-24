#include "../include/armor.hpp"
#include "detector.hpp"
#include "utils/draw.hpp"
#include "pnp_solver.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

using namespace armor_task;

// 相机内参
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 610, 0, 320, 0, 613, 240, 0, 0, 1);
cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

// 测试视频路径
std::string video_path = "/home/wxy/Downloads/circular1.avi";

// 配置路径
const std::string config_path = "/home/wxy/ArmorDetect_CV/config/demo.yaml";

/**
 * @brief 计算重投影误差（欧氏距离）
 */
double calculateReprojectionError(
  const Armor & armor, const std::vector<cv::Point2f> & reprojected_points)
{
  if (armor.corners.size() != reprojected_points.size()) {
    return -1.0;  // 错误
  }

  double total_error = 0.0;
  for (size_t i = 0; i < armor.corners.size(); ++i) {
    double dx = armor.corners[i].x - reprojected_points[i].x;
    double dy = armor.corners[i].y - reprojected_points[i].y;
    total_error += std::sqrt(dx * dx + dy * dy);
  }
  return total_error / armor.corners.size();  // 返回平均每个角点的误差
}

/**
 * @brief 不使用优化的PnP解算（baseline）
 * @param armor 装甲板信息
 * @param pnp_solver PnP求解器
 * @return 重投影误差（像素）
 */
double baseline_solvePnP(Armor & armor, PnpSolver & pnp_solver)
{
  // 调用内部的_solve_pnp但不调用optimize_yaw
  if (armor.left_lightbar.center == cv::Point2f(0, 0) || 
      armor.right_lightbar.center == cv::Point2f(0, 0)) {
    return -1.0;
  }

  // 获取装甲板角点
  pnp_solver.getArmorCorners(armor);
  if (armor.corners.size() != 4) {
    return -1.0;
  }

  // 判断大小装甲板
  bool islarge = (armor.car_num == 1 || armor.car_num == 4);
  const std::vector<cv::Point3f> & armor_points = 
    islarge ? pnp_solver.BIG_ARMOR_POINTS : pnp_solver.SMALL_ARMOR_POINTS;

  // 求解PnP
  cv::Mat rvec, tvec;
  bool success = cv::solvePnP(armor_points, armor.corners, camera_matrix, 
                              distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

  if (!success) {
    return -1.0;
  }

  // 计算重投影误差
  std::vector<cv::Point2f> reprojected_points;
  cv::projectPoints(armor_points, rvec, tvec, camera_matrix, distort_coeffs, reprojected_points);

  return calculateReprojectionError(armor, reprojected_points);
}

/**
 * @brief 使用优化的PnP解算（with SJTU_cost + optimize_yaw）
 */
double optimized_solvePnP(Armor & armor, PnpSolver & pnp_solver)
{
  // 调用完整的_solve_pnp方法（包含yaw优化）
  bool success = pnp_solver._solve_pnp(armor);
  
  if (!success) {
    return -1.0;
  }

  // 使用优化后的yaw角重新计算重投影
  auto reprojected_points = pnp_solver.reproject_armor(
    armor.p_world, armor.ypr_in_world[0], armor.car_num, armor.islarge);

  return calculateReprojectionError(armor, reprojected_points);
}

/**
 * @brief 统计数据结构
 */
struct AblationStats {
  std::vector<double> baseline_errors;
  std::vector<double> optimized_errors;
  std::vector<double> improvement_rates;  // (baseline - optimized) / baseline * 100

  double baseline_mean = 0.0;
  double optimized_mean = 0.0;
  double baseline_std = 0.0;
  double optimized_std = 0.0;
  double mean_improvement = 0.0;

  int total_samples = 0;
  int successful_samples = 0;
};

/**
 * @brief 计算统计量
 */
void computeStats(AblationStats & stats)
{
  if (stats.baseline_errors.empty()) {
    return;
  }

  stats.successful_samples = stats.baseline_errors.size();
  stats.total_samples = stats.successful_samples;

  // 计算均值
  double baseline_sum = 0.0, optimized_sum = 0.0, improvement_sum = 0.0;
  for (size_t i = 0; i < stats.baseline_errors.size(); ++i) {
    baseline_sum += stats.baseline_errors[i];
    optimized_sum += stats.optimized_errors[i];
    improvement_sum += stats.improvement_rates[i];
  }
  stats.baseline_mean = baseline_sum / stats.baseline_errors.size();
  stats.optimized_mean = optimized_sum / stats.optimized_errors.size();
  stats.mean_improvement = improvement_sum / stats.improvement_rates.size();

  // 计算标准差
  double baseline_var = 0.0, optimized_var = 0.0;
  for (size_t i = 0; i < stats.baseline_errors.size(); ++i) {
    baseline_var += (stats.baseline_errors[i] - stats.baseline_mean) * 
                   (stats.baseline_errors[i] - stats.baseline_mean);
    optimized_var += (stats.optimized_errors[i] - stats.optimized_mean) * 
                    (stats.optimized_errors[i] - stats.optimized_mean);
  }
  stats.baseline_std = std::sqrt(baseline_var / stats.baseline_errors.size());
  stats.optimized_std = std::sqrt(optimized_var / stats.optimized_errors.size());
}

/**
 * @brief 保存统计结果到CSV文件
 */
void saveStatsToCSV(const std::string & filename, const AblationStats & stats)
{
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return;
  }

  // 写入表头
  file << "Frame,Baseline_Error,Optimized_Error,Improvement_Rate(%)\n";

  // 写入数据
  for (size_t i = 0; i < stats.baseline_errors.size(); ++i) {
    file << i << ","
         << std::fixed << std::setprecision(4) << stats.baseline_errors[i] << ","
         << std::fixed << std::setprecision(4) << stats.optimized_errors[i] << ","
         << std::fixed << std::setprecision(2) << stats.improvement_rates[i] << "\n";
  }
  file.close();
  std::cout << "[Info] Statistics saved to: " << filename << std::endl;
}

/**
 * @brief 保存统计摘要
 */
void saveSummaryStats(const std::string & filename, const AblationStats & stats)
{
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return;
  }

  file << "=== Ablation Study Statistics ===\n\n";
  file << "Total Samples: " << stats.total_samples << "\n";
  file << "Successful Samples: " << stats.successful_samples << "\n\n";

  file << "Baseline (No Optimization):\n";
  file << "  Mean Error: " << std::fixed << std::setprecision(4) << stats.baseline_mean << " pixels\n";
  file << "  Std Dev: " << std::fixed << std::setprecision(4) << stats.baseline_std << " pixels\n";
  file << "  Min Error: " << std::fixed << std::setprecision(4) 
       << *std::min_element(stats.baseline_errors.begin(), stats.baseline_errors.end()) << " pixels\n";
  file << "  Max Error: " << std::fixed << std::setprecision(4) 
       << *std::max_element(stats.baseline_errors.begin(), stats.baseline_errors.end()) << " pixels\n\n";

  file << "Optimized (With SJTU_cost + optimize_yaw):\n";
  file << "  Mean Error: " << std::fixed << std::setprecision(4) << stats.optimized_mean << " pixels\n";
  file << "  Std Dev: " << std::fixed << std::setprecision(4) << stats.optimized_std << " pixels\n";
  file << "  Min Error: " << std::fixed << std::setprecision(4) 
       << *std::min_element(stats.optimized_errors.begin(), stats.optimized_errors.end()) << " pixels\n";
  file << "  Max Error: " << std::fixed << std::setprecision(4) 
       << *std::max_element(stats.optimized_errors.begin(), stats.optimized_errors.end()) << " pixels\n\n";

  file << "Improvement:\n";
  file << "  Mean Improvement Rate: " << std::fixed << std::setprecision(2) 
       << stats.mean_improvement << " %\n";
  file << "  Error Reduction: " << std::fixed << std::setprecision(4) 
       << (stats.baseline_mean - stats.optimized_mean) << " pixels\n";

  file.close();
  std::cout << "[Info] Summary saved to: " << filename << std::endl;
}

/**
 * @brief 生成可视化图表（使用OpenCV绘图）
 */
void generateVisualization(const std::string & filename, const AblationStats & stats)
{
  if (stats.baseline_errors.empty()) {
    return;
  }

  int num_samples = stats.baseline_errors.size();
  int plot_width = 1400;
  int plot_height = 700;
  int margin = 80;
  int legend_height = 60;

  cv::Mat visualization(plot_height + legend_height, plot_width, CV_8UC3, cv::Scalar(255, 255, 255));

  // 绘制标题
  cv::putText(visualization, "Ablation Study: Reprojection Error Comparison",
              cv::Point(plot_width / 2 - 250, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
              cv::Scalar(0, 0, 0), 2);

  // 绘制Y轴标签
  cv::putText(visualization, "Error (pixels)", cv::Point(10, 40), 
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  // 绘制X轴标签
  cv::putText(visualization, "Frame Index", 
              cv::Point(plot_width - 150, plot_height + margin - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  // 计算数据范围
  double max_error = *std::max_element(stats.baseline_errors.begin(), stats.baseline_errors.end());
  max_error = std::max(max_error, *std::max_element(stats.optimized_errors.begin(), 
                                                     stats.optimized_errors.end()));
  max_error *= 1.1;  // 留出10%的边距

  int plot_area_width = plot_width - 2 * margin;
  int plot_area_height = plot_height - margin;

  // 绘制网格线
  for (int i = 0; i <= 10; ++i) {
    int y = margin + (plot_area_height * i / 10);
    cv::line(visualization, cv::Point(margin, y), cv::Point(plot_width - margin, y),
             cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    double value = max_error * (10 - i) / 10;
    cv::putText(visualization, std::to_string((int)value), cv::Point(10, y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);
  }

  // 绘制数据点和线
  for (int i = 0; i < num_samples - 1; ++i) {
    // 基线数据（蓝色）
    int x1 = margin + (plot_area_width * i / (num_samples - 1));
    int y1 = plot_height - margin - (int)(plot_area_height * stats.baseline_errors[i] / max_error);
    int x2 = margin + (plot_area_width * (i + 1) / (num_samples - 1));
    int y2 = plot_height - margin - (int)(plot_area_height * stats.baseline_errors[i + 1] / max_error);
    cv::line(visualization, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    cv::circle(visualization, cv::Point(x1, y1), 3, cv::Scalar(255, 0, 0), -1);

    // 优化数据（绿色）
    y1 = plot_height - margin - (int)(plot_area_height * stats.optimized_errors[i] / max_error);
    y2 = plot_height - margin - (int)(plot_area_height * stats.optimized_errors[i + 1] / max_error);
    cv::line(visualization, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::circle(visualization, cv::Point(x1, y1), 3, cv::Scalar(0, 255, 0), -1);
  }

  // 绘制最后的点
  {
    int i = num_samples - 1;
    int x = margin + (plot_area_width * i / (num_samples - 1));
    int y = plot_height - margin - (int)(plot_area_height * stats.baseline_errors[i] / max_error);
    cv::circle(visualization, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), -1);

    y = plot_height - margin - (int)(plot_area_height * stats.optimized_errors[i] / max_error);
    cv::circle(visualization, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
  }

  // 绘制坐标轴
  cv::line(visualization, cv::Point(margin, margin), cv::Point(margin, plot_height),
           cv::Scalar(0, 0, 0), 2);
  cv::line(visualization, cv::Point(margin, plot_height), cv::Point(plot_width - margin, plot_height),
           cv::Scalar(0, 0, 0), 2);

  // 绘制图例
  cv::line(visualization, cv::Point(50, plot_height + 20), cv::Point(100, plot_height + 20),
           cv::Scalar(255, 0, 0), 2);
  cv::putText(visualization, "With Optimization", cv::Point(110, plot_height + 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  cv::line(visualization, cv::Point(450, plot_height + 20), cv::Point(500, plot_height + 20),
           cv::Scalar(0, 255, 0), 2);
  cv::putText(visualization, "Baseline (No Optimization)", 
              cv::Point(510, plot_height + 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  cv::imwrite(filename, visualization);
  std::cout << "[Info] Visualization saved to: " << filename << std::endl;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[])
{
  try {
    std::cout << "=== Ablation Study: SJTU_cost & optimize_yaw ===" << std::endl;
    std::cout << "Testing reprojection error with and without optimization" << std::endl;

    // 加载配置
    auto yaml = YAML::LoadFile(config_path);
    auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
    auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> cam_mat(camera_matrix_data.data());
    Eigen::Matrix<double, 1, 5> dist_coeffs(distort_coeffs_data.data());
    cv::eigen2cv(cam_mat, camera_matrix);
    cv::eigen2cv(dist_coeffs, distort_coeffs);

    // 初始化
    Detector detector;
    PnpSolver pnp_solver(config_path);
    AblationStats stats;

    // 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
      return -1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_count = 0;

    std::cout << "Processing " << total_frames << " frames..." << std::endl;

    cv::Mat frame;
    while (cap.read(frame) && frame_count < total_frames) {
      if (frame.empty()) continue;

      // 检测装甲板
      ArmorArray detected_armors = detector.detect(frame);

      for (auto & armor : detected_armors) {
        // 计算基线误差
        double baseline_error = baseline_solvePnP(armor, pnp_solver);
        
        // 计算优化误差
        double optimized_error = optimized_solvePnP(armor, pnp_solver);

        // 只记录有效的结果
        if (baseline_error > 0 && optimized_error > 0) {
          stats.baseline_errors.push_back(baseline_error);
          stats.optimized_errors.push_back(optimized_error);
          
          double improvement_rate = (baseline_error - optimized_error) / baseline_error * 100.0;
          stats.improvement_rates.push_back(improvement_rate);
        }
      }

      frame_count++;
      if (frame_count % 30 == 0) {
        std::cout << "Processed " << frame_count << "/" << total_frames << " frames, "
                  << "Total samples: " << stats.baseline_errors.size() << std::endl;
      }
    }

    cap.release();

    // 计算统计量
    computeStats(stats);

    // 输出摘要
    std::cout << "\n=== Ablation Study Results ===" << std::endl;
    std::cout << "Total Samples: " << stats.total_samples << std::endl;
    std::cout << "Successful Samples: " << stats.successful_samples << std::endl;
    std::cout << "\nBaseline (No Optimization):" << std::endl;
    std::cout << "  Mean Error: " << std::fixed << std::setprecision(4) << stats.baseline_mean << " pixels" << std::endl;
    std::cout << "  Std Dev: " << std::fixed << std::setprecision(4) << stats.baseline_std << " pixels" << std::endl;
    std::cout << "\nOptimized (With SJTU_cost + optimize_yaw):" << std::endl;
    std::cout << "  Mean Error: " << std::fixed << std::setprecision(4) << stats.optimized_mean << " pixels" << std::endl;
    std::cout << "  Std Dev: " << std::fixed << std::setprecision(4) << stats.optimized_std << " pixels" << std::endl;
    std::cout << "\nImprovement:" << std::endl;
    std::cout << "  Mean Improvement Rate: " << std::fixed << std::setprecision(2) 
              << stats.mean_improvement << " %" << std::endl;
    std::cout << "  Error Reduction: " << std::fixed << std::setprecision(4) 
              << (stats.baseline_mean - stats.optimized_mean) << " pixels" << std::endl;

    // 保存结果
    saveStatsToCSV("ablation_study_detailed.csv", stats);
    saveSummaryStats("ablation_study_summary.txt", stats);
    generateVisualization("ablation_study_plot.png", stats);

    std::cout << "\n[Success] Ablation study completed!" << std::endl;
    return 0;
  }
  catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}

