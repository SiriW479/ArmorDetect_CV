#pragma once

#include "structures.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace armor_task
{
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

    // ========== 参数调整接口 ==========
    void set_threshold(int t) { threshold_ = t; }
    void set_max_angle_error(float err) { max_angle_error_ = err; }
    void set_lightbar_ratio(float min_r, float max_r) 
    { 
        min_lightbar_ratio_ = min_r; 
        max_lightbar_ratio_ = max_r;
    }
    void set_min_lightbar_length(float len) { min_lightbar_length_ = len; }
    void set_armor_width_ratio(float min_r, float max_r)
    {
        armor_min_width_ratio_ = min_r;
        armor_max_width_ratio_ = max_r;
    }

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

    // 灯条检测参数
    float max_angle_error_;     // 灯条与竖直方向的最大角度差（度）
    float min_lightbar_ratio_;  // 灯条长宽比最小值
    float max_lightbar_ratio_;  // 灯条长宽比最大值
    float min_lightbar_length_; // 灯条最小长度（像素）

    // 装甲板匹配参数
    float armor_min_width_ratio_;
    float armor_max_width_ratio_;
  };
} // namespace armor_task