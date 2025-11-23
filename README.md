# Armor Detection and Tracking System 

## 项目简介 

本项目是一个基于 C++ 和 OpenCV 实现的实时装甲板检测、定位与跟踪系统。系统采用现代化的 **"Top-Down"** 目标检测方案，首先通过 **YOLOv8 神经网络模型** 快速定位图像中的装甲板大致位置，然后利用传统图像处理方法在检测框内精确提取灯条角点，最后通过 **PnP算法** 解算装甲板的三维位姿，并使用 **卡尔曼滤波器(Kalman Filter)** 对其运动轨迹进行预测。

## 代码结构说明

```
.
├── CMakeLists.txt         
├── README.md              
├── models/                # 存放模型文件
│   └── yolov8_armor.onnx
├── assets/                # 存放测试视频
│   └── test_video.avi
├── include/               
│   ├── armor.hpp          # 装甲板定义
│   └── structures.hpp     # 通用结构体定义
│   └── other_moudules.hpp # 待定模块
├── src/                   
│   ├── main.cpp           # 主程序入口
│   ├── ros2_manager.cpp   # 信息管理
│   ├── detector.cpp       # 神经网络推理与灯条提取模块
│   ├── pnp_solver.cpp     # PnP位姿解算
│   ├── decision_maker.cpp # 装甲板决策
│   └── tracker.cpp        # 卡尔曼滤波跟踪与预测模块
├── tasks/                 
│   ├── ros2_manager.hpp   # 信息管理
│   ├── detector.hpp       # 神经网络推理与灯条提取模块
│   ├── pnp_solver.hpp     # PnP位姿解算
│   ├── decision_maker.hpp # 装甲板决策
│   └── tracker.hpp        # 卡尔曼滤波跟踪与预测模块
└── tools/                 
    ├── img_tools.hpp      
    └── plotter.hpp      
```

## 结构体设计
- **`Armor`**
struct Armor 
{
    cv::Rect Box;                    // 方形位置，使用cv::Rect来表示装甲板的矩形框
    float confidence;                // ResNet识别的置信度
    cv::Scalar color;                // 装甲板的颜色，使用cv::Scalar表示（BGR格式）
    int detect_id;                   // 自动分配的装甲板ID
    int car_num;                     // 根据ResNet识别结果得到的装甲板数字
    LightBar left_LightBar;          // 左方灯条，类型为LightBar
    LightBar right_LightBar;         // 右方灯条，类型为LightBar
    float priority;                     // 评分系统给出的打击评分
    float yaw;                       // pnp解算出的偏航角
    cv::Point3f p_camera;            // pnp解算出的三维位置信息，包含(x, y, z)
};

- **`LightBar`**
struct LightBar 
{
    cv::Point2f center;              // 灯条的中心点
    cv::Point2f top;                 // 灯条上方的点
    cv::Point2f bottom;              // 灯条下方的点
    cv::Point2f top2bottom;          // 灯条从上到下的方向向量
};

- **`Robot`**
struct Robot 
{
    Armor armor;                     // 机器人所包含的Armor类
    float omega;                     // 机器人旋转的角速度
    float vx;                        // 机器人在水平方向的线速度
    float vz;                        // 机器人在垂直方向的线速度
};

## 模块职责及设计思路

- **`ros2_manager`**: 负责图像信息的订阅及最终装甲板现实坐标系位置的发布。
  - **模块一**：img_subscriber
    - 功能：订阅相机节点发布的图像信息作为系统输入
  - **模块二**：armor_publisher
    - 功能：发布最终预测的装甲板位置信息作为系统输出

- **`detector`**: 负责完成从“输入图像”到“输出带角点的装甲板列表”的全过程。
  - **模块一**：preprocess
    - 功能：进行模型输入前的预处理
  - **模块二**：search_armors
    - 功能：使用 YOLO 进行模型推理，输出装甲板（结构体）数列  
      此时所有 armor 仅有 Box、confidence、color、detect_id 确定  
      car_num、Light_Bar(left right)、priority、yaw、p_camera(x,y,z) 未知
  - **模块三**：postprocess
    - **3.1**：classify
      - 功能：使用 Resnet 进行数字识别，填充 armor 的 car_num 信息
    - **3.2**：extract_lightbars
      - 功能：用传统 CV 提取出 armor.box 里的灯条并将结果填充 armor 的 Light_Bar(left right) 信息

- **`pnp_solver`**: 接收一个带角点的装甲板，为其计算三维位姿。
  - **模块一**：pnp_solver（该模块无需修改）
    - 功能：使用 PnP 进行位置解算，填充 armor 的 yaw、p_camera(x,y,z) 信息

- **`decision_maker`**: 对一个给定的目标进行状态跟踪，并预测其未来位置。
  - **模块一**：score_evaluate
    - 功能：对输入的装甲板列表进行统一打分，并填充 armor 的 priority 信息

- **`tracker`**: 对一个给定的目标进行状态跟踪，并预测其未来位置。
  - **模块一**：ekf_tracker
    - 功能：完成装甲板预测功能（配合状态机使用）

- **`main`**: 程序入口

