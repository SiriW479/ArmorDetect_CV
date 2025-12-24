#include "../include/armor.hpp"
#include "detector.hpp"
#include "utils/draw.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace armor_task;

// 测试视频路径
std::string video_path = "/home/wxy/Downloads/circular1.avi";

int main(int argc, char *argv[])
{
    try
    {
        // 实例化 Detector
        std::cout << "Initializing Detector..." << std::endl;
        Detector detector;

        // 打开视频
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
            return -1;
        }

        // 获取视频信息
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double video_fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "Video Info:" << std::endl;
        std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
        std::cout << "  FPS: " << video_fps << std::endl;
        std::cout << "  Total Frames: " << total_frames << std::endl;

        cv::Mat frame;
        int frame_count = 0;

        std::cout << "Starting detection..." << std::endl;
        std::cout << "Press 'q' to quit, 'p' to pause/resume, SPACE to step frame" << std::endl;

        bool paused = false;

        while (true)
        {
            if (!paused)
            {
                if (!cap.read(frame))
                {
                    std::cout << "End of video or failed to read frame" << std::endl;
                    break;
                }
                frame_count++;
            }

            if (frame.empty())
                continue;

            cv::Mat display_frame = frame.clone();

            if (!paused)
            {
                // 检测装甲板
                ArmorArray detected_armors = detector.detect(frame);

                // 控制台输出检测结果
                std::cout << "Frame " << frame_count << ": Detected " << detected_armors.size() << " armors" << std::endl;
                for (size_t i = 0; i < detected_armors.size(); ++i)
                {
                    const auto &armor = detected_armors[i];
                    std::cout << "  [Armor " << i << "] Center: (" << armor.center.x << "," << armor.center.y << ")" << std::endl;
                }

                // 绘制检测结果
                drawArmorDetection(display_frame, detected_armors);

                // 用蓝色框可视化检测到的装甲板
                for (const auto &armor : detected_armors)
                {
                    cv::rectangle(display_frame, armor.box, cv::Scalar(255, 0, 0), 2); // 蓝色框
                }

                // 显示进度信息
                std::string progress = "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames);
                cv::putText(display_frame, progress, cv::Point(10, display_frame.rows - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }

            // 显示结果
            cv::imshow("Armor Detection Test", display_frame);

            // 处理键盘输入
            char key = cv::waitKey(paused ? 0 : 1) & 0xFF;
            if (key == 'q' || key == 27)
            { // 'q' 或 ESC 退出
                break;
            }
            else if (key == 'p')
            { // 'p' 暂停/恢复
                paused = !paused;
                std::cout << (paused ? "\nPaused" : "\nResumed") << std::endl;
            }
            else if (key == ' ' && paused)
            { // 空格键单步执行
                if (cap.read(frame))
                {
                    frame_count++;
                }
            }
        }

        // 清理资源
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
