#include "detector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace armor_task
{

    Detector::Detector() : enemy_color_(red),
                           threshold_(100),
                           armor_min_width_ratio_(1.0f),
                           armor_max_width_ratio_(3.2f),
                            max_angle_error_(25.0f),
                           min_lightbar_ratio_(1.5f),
                           max_lightbar_ratio_(20.0f),
                           min_lightbar_length_(8.0f)
    {
    }

    void Detector::set_enemy_color(Color color)
    {
        enemy_color_ = color;
    }

ArmorArray Detector::detect(const cv::Mat &frame)
{
    ArmorArray armors;
    // 模块一：图像预处理
    cv::Mat preprocessed_img = preprocess_image(frame);
    // 模块二：寻找灯条
    std::vector<LightBar> light_bars;
    find_lightbars(preprocessed_img, light_bars);

    // 调试：显示灯条数量
    if (light_bars.size() > 3)
    {
        std::cout << "[Debug] Found " << light_bars.size() << " light bars." << std::endl;
    }

    // 模块三：匹配灯条成装甲板
    match_armors(frame, light_bars, armors);

    std::cout << "[Debug] Detector found " << armors.size() << " armors in this frame." << std::endl;
    return armors;
}    cv::Mat Detector::preprocess_image(const cv::Mat &frame)
    {
        // 1. 颜色通道分离
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        cv::Mat gray_img;

        // 2. 根据敌方颜色进行通道相减
        if (enemy_color_ == blue)
        {
            cv::subtract(channels[0], channels[2], gray_img); // b - r
        }
        else
        {
            cv::subtract(channels[2], channels[0], gray_img); // r - b
        }

        // 3. 灰度化和二值化
        cv::Mat binary_img;
        cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);

        return binary_img;
    }

    void Detector::find_lightbars(const cv::Mat &frame, std::vector<LightBar> &light_bars)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (const auto &contour : contours)
        {
            if (contour.size() < 5)
            {
                continue;
            }

            cv::RotatedRect rotated_rect = cv::minAreaRect(contour);

            // 几何形状检查
            float aspect_ratio = std::max(rotated_rect.size.width, rotated_rect.size.height) /
                                 std::min(rotated_rect.size.width, rotated_rect.size.height);
            if (aspect_ratio < 1.2 || rotated_rect.size.area() < 10)
            {
                continue;
            }

            LightBar light_bar;
            light_bar.center = rotated_rect.center;
            light_bar.angle = rotated_rect.angle;
            light_bar.width = std::min(rotated_rect.size.width, rotated_rect.size.height);
            light_bar.height = std::max(rotated_rect.size.width, rotated_rect.size.height);
            light_bar.color = enemy_color_;

            cv::Point2f rect_points[4];
            rotated_rect.points(rect_points);
            // 对角点按y坐标排序，找到顶部和底部点
            std::sort(rect_points, rect_points + 4, [](const cv::Point2f &a, const cv::Point2f &b) {
                return a.y < b.y;
            });
            light_bar.top = (rect_points[0] + rect_points[1]) / 2;
            light_bar.bottom = (rect_points[2] + rect_points[3]) / 2;
            light_bar.top2bottom = light_bar.bottom - light_bar.top;

            light_bars.push_back(light_bar);
        }
    }

    void Detector::match_armors(const cv::Mat &frame, std::vector<LightBar> &light_bars, ArmorArray &armors)
    {
        if (light_bars.size() < 2)
        {
            return;
        }

        // 按x坐标排序
        std::sort(light_bars.begin(), light_bars.end(),
                  [](const LightBar &a, const LightBar &b) { return a.center.x < b.center.x; });

        for (size_t i = 0; i < light_bars.size() - 1; ++i)
        {
            for (size_t j = i + 1; j < light_bars.size(); ++j)
            {
                const LightBar &left_bar = light_bars[i];
                const LightBar &right_bar = light_bars[j];

                // 几何约束检查
                float angle_diff = std::abs(left_bar.angle - right_bar.angle);
                float height_ratio = std::min(left_bar.height, right_bar.height) /
                                     std::max(left_bar.height, right_bar.height);
                float y_diff = std::abs(left_bar.center.y - right_bar.center.y);
                float distance = cv::norm(left_bar.center - right_bar.center);
                float armor_width_ratio = distance / ((left_bar.height + right_bar.height) / 2.0f);

                if (angle_diff > 7.0f || height_ratio < 0.7f || y_diff > 20.0f ||
                    armor_width_ratio < armor_min_width_ratio_ || armor_width_ratio > armor_max_width_ratio_)
                {
                    continue;
                }

                cv::Rect rect = cv::boundingRect(std::vector<cv::Point>{
                    left_bar.top, left_bar.bottom, right_bar.top, right_bar.bottom});

                Armor armor(left_bar, right_bar, armors.size(), rect);
                armor.color = enemy_color_;

                // 设置装甲板角点 (左上, 右上, 右下, 左下)
                armor.corners.push_back(left_bar.top);
                armor.corners.push_back(right_bar.top);
                armor.corners.push_back(right_bar.bottom);
                armor.corners.push_back(left_bar.bottom);

                armors.push_back(armor);
            }
        }
    }

} // namespace armor_task
