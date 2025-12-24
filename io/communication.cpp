#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <Eigen/Dense>
#include "communication.hpp"

namespace io {
USB::USB(const std::string& send_port, const std::string& receive_port) {
        // Initialize send port
        send_fd = open(send_port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (send_fd < 0) {
            std::cerr << "Error opening send serial port: " << send_port << std::endl;
            return;
        }

        memset(&send_tty, 0, sizeof send_tty);
        if (tcgetattr(send_fd, &send_tty) != 0) {
            std::cerr << "Error getting send serial attributes" << std::endl;
            close(send_fd);
            return;
        }

        cfsetospeed(&send_tty, B115200);
        cfsetispeed(&send_tty, B115200);

        send_tty.c_cflag = (send_tty.c_cflag & ~CSIZE) | CS8;
        send_tty.c_iflag &= ~IGNBRK;
        send_tty.c_lflag = 0;
        send_tty.c_oflag = 0;
        send_tty.c_cc[VMIN] = 0;
        send_tty.c_cc[VTIME] = 5;

        send_tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        send_tty.c_cflag |= (CLOCAL | CREAD);
        send_tty.c_cflag &= ~(PARENB | PARODD);
        send_tty.c_cflag &= ~CSTOPB;
        send_tty.c_cflag &= ~CRTSCTS;

        if (tcsetattr(send_fd, TCSANOW, &send_tty) != 0) {
            std::cerr << "Error setting send serial attributes" << std::endl;
            close(send_fd);
            return;
        }

        // Initialize receive port
        receive_fd = open(receive_port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (receive_fd < 0) {
            std::cerr << "Error opening receive serial port: " << receive_port << std::endl;
            close(send_fd);
            return;
        }

        memset(&receive_tty, 0, sizeof receive_tty);
        if (tcgetattr(receive_fd, &receive_tty) != 0) {
            std::cerr << "Error getting receive serial attributes" << std::endl;
            close(send_fd);
            close(receive_fd);
            return;
        }

        cfsetospeed(&receive_tty, B115200);
        cfsetispeed(&receive_tty, B115200);

        receive_tty.c_cflag = (receive_tty.c_cflag & ~CSIZE) | CS8;
        receive_tty.c_iflag &= ~IGNBRK;
        receive_tty.c_lflag = 0;
        receive_tty.c_oflag = 0;
        receive_tty.c_cc[VMIN] = 0;
        receive_tty.c_cc[VTIME] = 5;

        receive_tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        receive_tty.c_cflag |= (CLOCAL | CREAD);
        receive_tty.c_cflag &= ~(PARENB | PARODD);
        receive_tty.c_cflag &= ~CSTOPB;
        receive_tty.c_cflag &= ~CRTSCTS;

        if (tcsetattr(receive_fd, TCSANOW, &receive_tty) != 0) {
            std::cerr << "Error setting receive serial attributes" << std::endl;
            close(send_fd);
            close(receive_fd);
        }
    }

USB::~USB() {
        if (send_fd >= 0) {
            close(send_fd);
        }
        if (receive_fd >= 0) {
            close(receive_fd);
        }
    }

bool USB::send_command(const Command& cmd) {
        // Pack command into byte array: 2 floats (8 bytes)
        char buffer[8];
        memcpy(buffer, &cmd.yaw, sizeof(float));
        memcpy(buffer + 4, &cmd.pitch, sizeof(float));
        ssize_t n_written = write(send_fd, buffer, sizeof(buffer));
        return n_written == sizeof(buffer);
    }

Eigen::Quaterniond USB::receive_quaternion() {
        // Read 16 bytes for 4 floats (since sender sends floats)
        char buffer[16];
        ssize_t n_read = read(receive_fd, buffer, sizeof(buffer));
        
        float w_f, x_f, y_f, z_f;
        memcpy(&w_f, buffer, sizeof(float));
        memcpy(&x_f, buffer + 4, sizeof(float));
        memcpy(&y_f, buffer + 8, sizeof(float));
        memcpy(&z_f, buffer + 12, sizeof(float));


        // Convert floats to doubles (no rounding, just precision increase)
        return Eigen::Quaterniond(static_cast<double>(w_f), 
                                  static_cast<double>(x_f), 
                                  static_cast<double>(y_f), 
                                  static_cast<double>(z_f));
    }

        
} // namespace io
