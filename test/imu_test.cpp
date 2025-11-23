#include "../io/communication.hpp"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <sstream>

using namespace io;

int main(int argc, char *argv[])
{
    try {
        std::cout << "Initializing communication test..." << std::endl;

        // 初始化通信 - 使用不同的串口进行发送和接收，根据实际设备调整
        USB usb("/dev/ttyACM0", "/dev/ttyACM0"); // 发送端口, 接收端口

        std::cout << "Starting continuous IMU data reception..." << std::endl;

        while (true) {
            try {
                Eigen::Quaterniond q = usb.receive_quaternion();
                std::cout << "Received IMU: (" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << ")" << std::endl;
                // 短暂延迟以避免CPU占用过高，可根据需要调整
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } catch (const std::exception& e) {
                std::cerr << "IMU receive error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 错误时稍长延迟
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
