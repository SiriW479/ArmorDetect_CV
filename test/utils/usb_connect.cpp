#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <iostream>

struct Command {
    float data[4];
    bool flag;
};

bool send_Command_to_cboard(const Command& cmd) {
    int fd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::cerr << "Error opening /dev/ttyACM0" << std::endl;
        return false;
    }

    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error getting tty attributes" << std::endl;
        close(fd);
        return false;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error setting tty attributes" << std::endl;
        close(fd);
        return false;
    }

    // Pack Command: 4 floats + 1 bool = 17 bytes
    char buffer[17];
    memcpy(buffer, cmd.data, 4 * sizeof(float));
    buffer[16] = cmd.flag ? 1 : 0;

    ssize_t written = write(fd, buffer, sizeof(buffer));
    close(fd);
    return written == sizeof(buffer);
}
