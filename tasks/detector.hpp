#pragma once

#include "structures.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace armor_task {
class Detector
{
  public:
    // 构造函数
    Detector();

    // 析构函数
    ~Detector() = default;

    /*
    主要检测函数:
        输入：带装甲板的图片     类型：cv::Mat
        输出：装甲板列表        类型：ArmorArray
    */
    ArmorArray detect(const cv::Mat &frame);

    // 用于设置敌方颜色
    void set_enemy_color(Color color);

    // 模块一：图像预处理
    cv::Mat preprocess_image(const cv::Mat &frame);

  private:
    // 模块二：寻找灯条
    void find_lightbars(const cv::Mat &frame, std::vector<LightBar> &light_bars);

    // 模块三：匹配灯条成装甲板
    void match_armors(const cv::Mat &frame, std::vector<LightBar> &light_bars, ArmorArray &armors);

  private:
    Color enemy_color_; // 敌方颜色

    // ========== 识别参数 ==========
    // 预处理
    int threshold_;

    // 匹配装甲板
    float armor_min_width_ratio_;
    float armor_max_width_ratio_;
};
} // namespace armor_task