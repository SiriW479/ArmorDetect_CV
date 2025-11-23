#include "../include/armor.hpp"
#include "detector.hpp"
#include "utils/draw.hpp"
#include "pnp_solver.hpp"
#include "tracker.hpp"
#include "utils/draw.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace armor_task;

// 相机内参
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 610, 0, 320, 0, 613, 240, 0, 0, 1);

// 畸变系数
cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

// 测试视频路径
std::string video_path = "/home/wxy/Downloads/circular1.avi";


// // 绘制装甲板检测结果
// void drawArmorDetection(cv::Mat& img, const ArmorArray& armors) {
//     for (const auto& armor : armors) {
//         // 绘制装甲板中心点
//         tools::draw_point(img, armor.center, cv::Scalar(0, 0, 255), 3);
        
//         // 显示3D位置信息（如果已解算）
//         if (armor.p_camera != Eigen::Vector3d::Zero()) {
//             std::string pos_info = "Pos:(" + 
//                                   std::to_string((int)armor.p_camera[0]) + "," +
//                                   std::to_string((int)armor.p_camera[1]) + "," +
//                                   std::to_string((int)armor.p_camera[2]) + ")mm";
//             tools::draw_text(img, pos_info, 
//                            cv::Point(armor.box.x, armor.box.y + armor.box.height + 15),
//                            cv::Scalar(0, 255, 255), 0.4, 1);
            
//             // 投影装甲板角点
//             if (!armor.corners.empty()) {
//                 tools::draw_points(img, armor.corners, cv::Scalar(0, 255, 0), 2);
//             }
            
//             // 绘制 p_camera 的投影点
//             if (armor.p_camera[2] > 0) { // Z > 0，在相机前方
//                 cv::Point2f projected_point;
//                 projected_point.x = camera_matrix.at<double>(0, 0) * armor.p_camera[0] / armor.p_camera[2] + camera_matrix.at<double>(0, 2);
//                 projected_point.y = camera_matrix.at<double>(1, 1) * armor.p_camera[1] / armor.p_camera[2] + camera_matrix.at<double>(1, 2);
                
//                 // 确保投影点在图像范围内
//                 if (projected_point.x >= 0 && projected_point.x < img.cols && 
//                     projected_point.y >= 0 && projected_point.y < img.rows) {
//                     tools::draw_point(img, projected_point, cv::Scalar(255, 0, 0), 5);
//                     tools::draw_text(img, "p_camera", 
//                                    cv::Point(projected_point.x + 10, projected_point.y - 10),
//                                    cv::Scalar(255, 0, 0), 0.4, 1);
//                 }
//             }
//         }
//     }
// }

// // 绘制Target详细信息
// void drawTargetInfo(cv::Mat& img, const std::vector<Target>& targets, const std::string& tracker_state, const PnpSolver& pnp_solver) {
//     // 显示追踪器状态
//     tools::draw_text(img, "Tracker State: " + tracker_state, 
//                      cv::Point(10, 30), cv::Scalar(0, 255, 0), 0.7, 2);
    
//     if (targets.empty()) {
//         tools::draw_text(img, "No Target", cv::Point(10, 60), 
//                         cv::Scalar(0, 0, 255), 0.6, 2);
//         return;
//     }
    
//     const auto& target = targets.front();
//     int y_offset = 60;
//     int line_height = 20;
    
//     // Target基本信息
//     tools::draw_text(img, "=== Target Info ===", cv::Point(10, y_offset), 
//                      cv::Scalar(255, 255, 255), 0.6, 2);
//     y_offset += line_height;
    
//     // 获取所有装甲板的预测位置和角度
//     auto xyza_list = target.armor_xyza_list();
//     std::cout<<"list size: " << xyza_list.size() << std::endl;

//     // 定义相机到世界坐标系的旋转矩阵
//     Eigen::Matrix3d R_wc;
//     R_wc << 0, -1,  0,
//             0,  0, -1,
//             1,  0,  0;

//     // 遍历所有装甲板并重投影
//     for (size_t i = 0; i < xyza_list.size(); ++i) {
//         Eigen::Vector3d p_world = xyza_list[i].head<3>();  // 获取世界坐标
        
//         // 获取重投影点（默认使用小装甲板）
//         auto image_points = pnp_solver.reproject_armor(
//             xyza_list[i].head<3>(), xyza_list[i][3], target.car_num, false);
        
//         // 绘制重投影点
//         tools::draw_points(img, image_points, cv::Scalar(0, 255, 0), 2);

//         // 输出 xyz 位置
//         std::cout << "Armor " << i << " position: x=" << p_world[0] << ", y=" << p_world[1] << ", z=" << p_world[2] << std::endl;
//     }
        
    
             
//     Eigen::VectorXd ekf_state = target.ekf_x();  // 使用 ekf_x() 获取状态向量
    
//     // 打印 EKF 状态向量
//     std::cout << "EKF State x: [";
//     for (int i = 0; i < ekf_state.size(); ++i) {
//         std::cout << ekf_state[i];
//         if (i < ekf_state.size() - 1) std::cout << ", ";
//     }
//     std::cout << "]" << std::endl;
    
//     if (ekf_state.size() >= 5) {
//         Eigen::Vector3d ekf_world(ekf_state[0], ekf_state[2], ekf_state[4]);
//         // 使用上面定义的 R_wc
//         Eigen::Vector3d ekf_camera = R_wc * ekf_world;
//         if (ekf_camera[2] > 0) { 
//             cv::Point2f ekf_proj;
//             ekf_proj.x = camera_matrix.at<double>(0, 0) * ekf_camera[0] / ekf_camera[2] + camera_matrix.at<double>(0, 2);
//             ekf_proj.y = camera_matrix.at<double>(1, 1) * ekf_camera[1] / ekf_camera[2] + camera_matrix.at<double>(1, 2);
//             if (ekf_proj.x >= 0 && ekf_proj.x < img.cols &&
//                 ekf_proj.y >= 0 && ekf_proj.y < img.rows) {
//                 cv::circle(img, ekf_proj, 5, cv::Scalar(0, 0, 255), -1); // 红色圆点
//                 cv::putText(img, "EKF", cv::Point(ekf_proj.x + 10, ekf_proj.y - 10),
//                             cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
//             }
//         }
//     }
    
//     // 是否切换标志
//     if (target.is_switch_) {
//         cv::putText(img, "TARGET SWITCHED!", cv::Point(10, y_offset), 
//                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
//         y_offset += line_height;
//     }
    
//     // 是否跳跃
//     if (target.jumped) {
//         cv::putText(img, "TARGET JUMPED!", cv::Point(10, y_offset), 
//                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
//         y_offset += line_height;
//     }
    
//     // EKF状态信息
//     cv::putText(img, "=== EKF State ===", cv::Point(10, y_offset), 
//                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
//     y_offset += line_height;
    
//     // 发散状态检查
//     bool diverged = target.diverged();
//     std::string diverge_status = diverged ? "DIVERGED!" : "Converged";
//     cv::Scalar diverge_color = diverged ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
//     cv::putText(img, "Status: " + diverge_status, cv::Point(10, y_offset), 
//                cv::FONT_HERSHEY_SIMPLEX, 0.5, diverge_color, 2);
//     y_offset += line_height;
    
    
// }
 
// // 显示性能信息
// void drawPerformanceInfo(cv::Mat& img, double fps, double detect_time, double track_time) {
//     int x_pos = img.cols - 200;
//     int y_offset = 30;
//     int line_height = 20;
    
//     // FPS信息
//     std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 5);
//     tools::draw_text(img, fps_text, cv::Point(x_pos, y_offset), 
//                      cv::Scalar(0, 255, 0), 0.6, 2);
//     y_offset += line_height;
    
//     // 检测时间
//     std::string detect_text = "Detect: " + std::to_string(detect_time).substr(0, 5) + "ms";
//     tools::draw_text(img, detect_text, cv::Point(x_pos, y_offset), 
//                      cv::Scalar(255, 255, 255), 0.5, 1);
//     y_offset += line_height;
    
//     // 追踪时间
//     std::string track_text = "Track: " + std::to_string(track_time).substr(0, 5) + "ms";
//     tools::draw_text(img, track_text, cv::Point(x_pos, y_offset), 
//                      cv::Scalar(255, 255, 255), 0.5, 1);
// }

int main(int argc, char *argv[])
{
    try {
        // 实例化功能组件
        std::cout << "Initializing components..." << std::endl;
        Detector detector;
        PnpSolver pnp_solver("/home/wxy/ArmorDetect_CV/config/demo.yaml");
        Tracker tracker("/home/wxy/ArmorDetect_CV/config/demo.yaml", pnp_solver);
    std::cout << "[Debug] Detector instance: " << &detector << std::endl;
    std::cout << "[Debug] PnpSolver instance: " << &pnp_solver << std::endl;
    std::cout << "[Debug] Tracker instance: " << &tracker << std::endl;

        // 打开视频
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
            return -1;
        }

        // 获取视频信息
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double video_fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

      std::cout << "Video Info: Resolution: " << frame_width << "x" << frame_height
            << ", FPS: " << video_fps
            << ", Total Frames: " << total_frames << std::endl;

        cv::Mat frame;
        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        auto last_fps_time = start_time;
        double fps = 0.0;

        std::cout << "Starting processing..." << std::endl;
        std::cout << "Press 'q' to quit, 'p' to pause/resume, SPACE to step frame" << std::endl;

        bool paused = false;

        while (true) {
            auto frame_start = std::chrono::steady_clock::now();

            if (!paused) {
                if (!cap.read(frame)) {
                    std::cout << "End of video or failed to read frame" << std::endl;
                    break;
                }
                frame_count++;
            }

            if (frame.empty()) continue;

            cv::Mat display_frame = frame.clone();

            if (!paused) {
                // 检测阶段
                auto detect_start = std::chrono::steady_clock::now();
                ArmorArray detected_armors = detector.detect(frame);
                auto detect_end = std::chrono::steady_clock::now();
                double detect_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
                for (size_t i = 0; i < detected_armors.size(); ++i) {
                    const auto& armor = detected_armors[i];
                    std::cout << "[Armor " << i << "] "
                              << "ID: " << armor.detect_id
                              << ", Num: " << armor.car_num
                              << ", Confidence: " << std::fixed << std::setprecision(3) << armor.confidence
                              << ", Box: (" << armor.box.x << "," << armor.box.y << "," << armor.box.width << "," << armor.box.height << ")"
                              << ", Center: (" << armor.center.x << "," << armor.center.y << ")"
                              << std::endl;
                    std::cout.unsetf(std::ios::floatfield);
                    std::cout << std::defaultfloat << std::setprecision(6);
                }

                // 追踪阶段
                auto track_start = std::chrono::steady_clock::now();
                auto targets = tracker.track(detected_armors, frame_start);
                auto track_end = std::chrono::steady_clock::now();
                double track_time = std::chrono::duration<double, std::milli>(track_end - track_start).count();

                // 绘制检测结果
                drawArmorDetection(display_frame, detected_armors);

                // 绘制Target详细信息
                drawTargetInfo(display_frame, targets, tracker.state(), pnp_solver);

                // 计算FPS
                auto current_time = std::chrono::steady_clock::now();
                auto fps_duration = std::chrono::duration<double>(current_time - last_fps_time).count();
                if (fps_duration > 0.5) { // 每0.5秒更新一次FPS
                    fps = 1.0 / std::chrono::duration<double>(current_time - frame_start).count();
                    last_fps_time = current_time;
                }

                // 显示性能信息
                drawPerformanceInfo(display_frame, fps, detect_time, track_time);

                // 显示进度信息
                std::string progress = "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames);
                cv::putText(display_frame, progress, cv::Point(10, display_frame.rows - 20), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // 控制台输出
                if (frame_count % 30 == 0) {
                    std::cout << "Processing frame " << frame_count << "/" << total_frames
                              << " | FPS: " << std::fixed << std::setprecision(2) << fps
                              << " | Detected: " << detected_armors.size()
                              << " | Tracking: " << targets.size()
                              << " | State: " << tracker.state()
                              << std::endl;
                    std::cout.unsetf(std::ios::floatfield);
                    std::cout << std::defaultfloat << std::setprecision(6);
                }
            }

            // 显示结果
            cv::imshow("Auto Aim Test - Target Visualization", display_frame);

            // 处理键盘输入
            char key = cv::waitKey(paused ? 0 : 1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
                break;
            } else if (key == 'p') { // 'p' 暂停/恢复
                paused = !paused;
                std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            } else if (key == ' ' && paused) { // 空格键单步执行
                if (cap.read(frame)) {
                    frame_count++;
                }
            }
        }

        // 计算总体统计信息
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration<double>(end_time - start_time).count();
        double avg_fps = frame_count / total_duration;

    std::cout << "Processing completed! Total frames processed: " << frame_count
          << ", Total time: " << std::fixed << std::setprecision(3) << total_duration << " seconds"
          << ", Average FPS: " << avg_fps << std::endl;
    std::cout.unsetf(std::ios::floatfield);
    std::cout << std::setprecision(6);

        // 清理资源
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}