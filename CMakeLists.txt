cmake_minimum_required(VERSION 3.10)
project(ArmorTask)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,/usr/lib/x86_64-linux-gnu")
# 指定 OpenCV 安装路径
#set(OpenCV_DIR "/home/wxy/Downloads/opencv-4.12.0/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)


# 查找 cv_bridge
#find_package(cv_bridge REQUIRED)

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 查找 yaml-cpp
find_package(yaml-cpp REQUIRED)

# 查找 spdlog
find_package(spdlog REQUIRED)

# 查找 ROS 2 依赖
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${PROJECT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/utils
    ${OpenCV_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${rclcpp_INCLUDE_DIRS}  # ROS 2 包含目录
)

# 添加源文件
set(SOURCES
    #tasks/main.cpp
    tasks/detector.cpp
    tasks/pnp_solver.cpp
    #tasks/ros2_manager.cpp  # 这是 ROS 2 相关的文件
    tasks/tracker.cpp
    tasks/ekf.cpp
    tasks/aimer.cpp
    tasks/trajectory_normal.cpp
    tasks/target.cpp
    tools/logger.cpp
    tools/math_tools.cpp  
    test/utils/draw.cpp
    io/communication.cpp
)


# 编译视频测试可执行文件
add_executable(vedio_test test/auto_aim_vedio_test.cpp ${SOURCES})

# 编译检测测试可执行文件
add_executable(detect_test test/detect_test.cpp ${SOURCES})

# ============================================================================
# 消融实验可执行文件
# ============================================================================
add_executable(ablation_study_test test/ablation_study_test.cpp ${SOURCES})

# 编译摄像头测试可执行文件
# add_executable(camera_test test/camera_auto_aim_test.cpp ${SOURCES})

# add_executable(imu_test test/imu_test.cpp ${SOURCES})


# 链接 ROS 2 的库
#target_link_libraries(${PROJECT_NAME} rclcpp::rclcpp)

# 链接测试可执行文件
target_link_libraries(vedio_test ${OpenCV_LIBS} ${EIGEN3_LIBRARIES} yaml-cpp spdlog::spdlog pthread)
target_link_libraries(detect_test ${OpenCV_LIBS} ${EIGEN3_LIBRARIES} yaml-cpp spdlog::spdlog pthread)
target_link_libraries(ablation_study_test ${OpenCV_LIBS} ${EIGEN3_LIBRARIES} yaml-cpp spdlog::spdlog pthread)
# target_link_libraries(camera_test ${OpenCV_LIBS} ${EIGEN3_LIBRARIES} yaml-cpp spdlog::spdlog pthread)
# target_link_libraries(imu_test ${OpenCV_LIBS} ${EIGEN3_LIBRARIES} yaml-cpp spdlog::spdlog pthread)


# 添加编译选项
target_compile_options(vedio_test PRIVATE -Wall -Wextra)
target_compile_options(detect_test PRIVATE -Wall -Wextra)
target_compile_options(ablation_study_test PRIVATE -Wall -Wextra)
# target_compile_options(camera_test PRIVATE -Wall -Wextra)
# target_compile_options(imu_test PRIVATE -Wall -Wextra)

# 设置构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

# 如果是 Debug 模式，添加调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vedio_test PRIVATE -g)
    target_compile_options(detect_test PRIVATE -g)
    target_compile_options(ablation_study_test PRIVATE -g)
    # target_compile_options(camera_test PRIVATE -g)
    # target_compile_options(imu_test PRIVATE -g)
endif()

# 消融实验消息
message(STATUS "Build configuration:")
message(STATUS "  Executable targets: vedio_test, detect_test, ablation_study_test")
message(STATUS "  Ablation study test configured successfully")
