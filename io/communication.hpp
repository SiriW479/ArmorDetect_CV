#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP

#include <Eigen/Dense>
#include <string>
#include <termios.h>

namespace io {
struct Command {
    float yaw;
    float pitch;
    
};

struct imu_data {
    float q1;
    float q2;
    float q3;
    float q4;
};

class USB {
private:
    int send_fd;
    int receive_fd;
    struct termios send_tty;
    struct termios receive_tty;

public:
    USB(const std::string& send_port, const std::string& receive_port);
    ~USB();

    bool send_command(const Command& cmd);
    Eigen::Quaterniond receive_quaternion();
};
} // namespace io
#endif // COMMUNICATION_HPP

