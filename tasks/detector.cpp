#include "detector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace armor_task {

Detector::Detector() : enemy_color_(blue) {}

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
    // 模块三：匹配灯条成装甲板
    match_armors(frame, light_bars, armors);
    return armors;
}

cv::Mat Detector::preprocess_image(const cv::Mat &frame)
{
    cv::Mat hsv_image;
    cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat thresholded_img;
    if (enemy_color_ == blue) {
        // 蓝色阈值
        cv::inRange(hsv_image, cv::Scalar(100, 150, 150), cv::Scalar(124, 255, 255), thresholded_img);
    } else {
        // 红色阈值
        cv::Mat red_mask1, red_mask2;
        cv::inRange(hsv_image, cv::Scalar(0, 150, 150), cv::Scalar(10, 255, 255), red_mask1);
        cv::inRange(hsv_image, cv::Scalar(170, 150, 150), cv::Scalar(180, 255, 255), red_mask2);
        thresholded_img = red_mask1 | red_mask2;
    }

    // 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresholded_img, thresholded_img, cv::MORPH_DILATE, kernel);

    return thresholded_img;
}

void Detector::find_lightbars(const cv::Mat &frame, std::vector<LightBar> &light_bars)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours) {
        if (contour.size() < 5) {
            continue;
        }

        cv::RotatedRect rotated_rect = cv::fitEllipse(contour);
        float aspect_ratio = std::max(rotated_rect.size.width, rotated_rect.size.height) /
                             std::min(rotated_rect.size.width, rotated_rect.size.height);

        // 根据长宽比和面积筛选灯条
        if (aspect_ratio > 1.5 && rotated_rect.size.area() > 10) {
            LightBar light_bar;
            light_bar.center = rotated_rect.center;
            light_bar.angle = rotated_rect.angle;
            light_bar.width = rotated_rect.size.width;
            light_bar.height = rotated_rect.size.height;
            light_bars.push_back(light_bar);
        }
    }
}

void Detector::match_armors(const cv::Mat &frame, std::vector<LightBar> &light_bars, ArmorArray &armors)
{
    if (light_bars.size() < 2) {
        return;
    }

    std::sort(light_bars.begin(), light_bars.end(),
              [](const LightBar &a, const LightBar &b) { return a.center.x < b.center.x; });

    for (size_t i = 0; i < light_bars.size() - 1; ++i) {
        for (size_t j = i + 1; j < light_bars.size(); ++j) {
            const LightBar &left_bar = light_bars[i];
            const LightBar &right_bar = light_bars[j];

            // 角度差
            float angle_diff = std::abs(left_bar.angle - right_bar.angle);
            // 高度差
            float height_ratio = std::min(left_bar.height, right_bar.height) /
                                 std::max(left_bar.height, right_bar.height);
            // Y坐标差
            float y_diff = std::abs(left_bar.center.y - right_bar.center.y);
            // 距离
            float distance = cv::norm(left_bar.center - right_bar.center);
            // 装甲板宽度与灯条高度比
            float armor_width_ratio = distance / ((left_bar.height + right_bar.height) / 2.0f);

            // 筛选条件
            if (angle_diff < 10.0f && height_ratio > 0.7f && y_diff < 30.0f &&
                armor_width_ratio > 1.0f && armor_width_ratio < 5.0f) {
                
                cv::Rect rect = cv::boundingRect(std::vector<cv::Point>{
                    cv::Point(left_bar.center.x - left_bar.width / 2, left_bar.center.y - left_bar.height / 2),
                    cv::Point(right_bar.center.x + right_bar.width / 2, right_bar.center.y + right_bar.height / 2)
                });

                Armor armor(left_bar, right_bar, armors.size(), rect);
                armor.color = enemy_color_;
                armors.push_back(armor);
            }
        }
    }
}

} // namespace armor_task