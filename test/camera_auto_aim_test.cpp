#include "../include/armor.hpp"
#include "detector.hpp"
#include "img_tools.hpp"
#include "pnp_solver.hpp"
#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include "../io/communication.hpp"
#include "utils/draw.hpp"
#include <yaml-cpp/yaml.h>

using namespace armor_task;
using namespace io;

// 测试摄像头索引
int camera_index = 0;

int main(int argc, char *argv[])
{
    try {
        // 实例化功能组件
        std::cout << "Initializing components..." << std::endl;
        Detector detector("/home/nvidia/NJU_RMVision/armor_task/models/yolov8_armor.onnx");
        
        // 从配置文件加载相机参数
        auto [camera_matrix, distort_coeffs] = loadCameraParameters("/home/nvidia/NJU_RMVision/armor_task/config/demo.yaml");
        
        PnpSolver pnp_solver(camera_matrix, distort_coeffs);
        Tracker tracker("/home/nvidia/NJU_RMVision/armor_task/config/demo.yaml", pnp_solver);

        // 初始化通信 - 使用不同的串口进行发送和接收
        USB usb("/dev/ttyACM0", "/dev/ttyACM0"); // 发送端口, 接收端口 - 根据实际设备调整

        // 共享数据结构
        std::queue<Command> command_queue;
        std::mutex queue_mutex;
        Eigen::Quaterniond imu_quaternion = Eigen::Quaterniond::Identity();
        std::mutex imu_mutex;
        std::atomic<bool> running(true);

        // IMU接收线程
        std::thread imu_thread([&]() {
            while (running) {
                try {
                    Eigen::Quaterniond q = usb.receive_quaternion();
                    {
                        std::lock_guard<std::mutex> lock(imu_mutex);
                        imu_quaternion = q;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "IMU receive error: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });

        // 命令发送线程
        std::thread command_thread([&]() {
            while (running) {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (!command_queue.empty()) {
                    Command cmd = command_queue.front();
                    command_queue.pop();
                    lock.unlock();

                    try {
                        usb.send_command(cmd);
                    } catch (const std::exception& e) {
                        std::cerr << "Command send error: " << e.what() << std::endl;
                    }
                } else {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });

        // 打开摄像头
        cv::VideoCapture cap(camera_index);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera with index: " << camera_index << std::endl;
            return -1;
        }

        // 获取摄像头信息
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double camera_fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "Camera Info:" << std::endl;
        std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
        std::cout << "  FPS: " << camera_fps << std::endl;

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
                // 设置IMU四元数到PnP求解器
                Eigen::Quaterniond current_imu;
                {
                    std::lock_guard<std::mutex> lock(imu_mutex);
                    current_imu = imu_quaternion;
                }
                pnp_solver.set_R_gimbal2world(current_imu);

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
                              << ", Confidence: " << armor.confidence
                              << ", Box: (" << armor.box.x << "," << armor.box.y << "," << armor.box.width << "," << armor.box.height << ")"
                              << ", Center: (" << armor.center.x << "," << armor.center.y << ")"
                              << std::endl;
                }

                
                // 追踪阶段
                auto track_start = std::chrono::steady_clock::now();
                auto targets = tracker.track(detected_armors, frame_start);
                auto track_end = std::chrono::steady_clock::now();
                double track_time = std::chrono::duration<double, std::milli>(track_end - track_start).count();

                // 计算并发送命令
                if (!targets.empty()) {
                    const auto& target = targets.front();
                    Eigen::VectorXd ekf_state = target.ekf_x();
                    
                    // 简单的瞄准控制逻辑（需要根据实际需求调整）
                    float yaw = static_cast<float>(ekf_state[0]);   // x位置作为yaw
                    float pitch = static_cast<float>(ekf_state[2]); // y位置作为pitch

                    Command cmd = {yaw, pitch};
                    
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    command_queue.push(cmd);
                    std::cout << "Queued Command - Yaw: " << yaw << ", Pitch: " << pitch << std::endl;
                }

                // 绘制检测结果
                drawArmorDetection(display_frame, detected_armors, camera_matrix);

                // 绘制Target详细信息
                drawTargetInfo(display_frame, targets, tracker.state(), camera_matrix);

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
                std::string progress = "Frame: " + std::to_string(frame_count);
                cv::putText(display_frame, progress, cv::Point(10, display_frame.rows - 20), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // 控制台输出
                if (frame_count % 30 == 0) {
                    std::cout << "\rProcessing frame " << frame_count 
                              << " | FPS: " << std::setprecision(3) << fps 
                              << " | Detected: " << detected_armors.size() 
                              << " | Tracking: " << targets.size() 
                              << " | State: " << tracker.state() << std::flush;
                }
            }

            // 显示结果
            cv::imshow("Auto Aim Test - Target Visualization", display_frame);

            // 处理键盘输入
            char key = cv::waitKey(paused ? 0 : 1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
                running = false;
                break;
            } else if (key == 'p') { // 'p' 暂停/恢复
                paused = !paused;
                std::cout << (paused ? "\nPaused" : "\nResumed") << std::endl;
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

        std::cout << "\n\nProcessing completed!" << std::endl;
        std::cout << "Total frames processed: " << frame_count << std::endl;
        std::cout << "Total time: " << std::setprecision(3) << total_duration << " seconds" << std::endl;
        std::cout << "Average FPS: " << std::setprecision(3) << avg_fps << std::endl;

        // 停止线程
        running = false;
        if (imu_thread.joinable()) {
            imu_thread.join();
        }
        if (command_thread.joinable()) {
            command_thread.join();
        }

        // 清理资源
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}