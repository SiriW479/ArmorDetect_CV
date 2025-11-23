#include "detector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace armor_task {

Detector::Detector() : 
    enemy_color_(blue),
    threshold_(128),
    armor_min_width_ratio_(1.0f),
    armor_max_width_ratio_(3.2f)
{}

void Detector::set_enemy_color(Color color)
{
    enemy_color_ = color;
}

ArmorArray Detector::detect(const cv::Mat &frame)
{
    // 根据图片估算ROI区域，大约在图像垂直方向的下半部分
    int roi_y = frame.rows / 2;
    // 确保ROI的高度不会超出图像边界
    cv::Rect roi_rect(0, roi_y, frame.cols, frame.rows - roi_y);
    cv::Mat roi_frame = frame(roi_rect);

    ArmorArray armors;
    // 模块一：图像预处理
    cv::Mat preprocessed_img = preprocess_image(roi_frame);
    // 模块二：寻找灯条
    std::vector<LightBar> light_bars;
    find_lightbars(preprocessed_img, light_bars);
    // 模块三：匹配灯条成装甲板
    match_armors(roi_frame, light_bars, armors);

    // 将检测到的装甲板坐标从ROI坐标系转换回原始图像坐标系
    for (auto &armor : armors) {
        armor.box.x += roi_rect.x;
        armor.box.y += roi_rect.y;
        armor.center.x += roi_rect.x;
        armor.center.y += roi_rect.y;
        
        // 转换角点坐标
        for (auto& corner : armor.corners) {
            corner.x += roi_rect.x;
            corner.y += roi_rect.y;
        }

        // 转换灯条中心点坐标
        armor.left_lightbar.center.x += roi_rect.x;
        armor.left_lightbar.center.y += roi_rect.y;
        armor.right_lightbar.center.x += roi_rect.x;
        armor.right_lightbar.center.y += roi_rect.y;

        armor.left_lightbar.top.x += roi_rect.x;
        armor.left_lightbar.top.y += roi_rect.y;
        armor.left_lightbar.bottom.x += roi_rect.x;
        armor.left_lightbar.bottom.y += roi_rect.y;

        armor.right_lightbar.top.x += roi_rect.x;
        armor.right_lightbar.top.y += roi_rect.y;
        armor.right_lightbar.bottom.x += roi_rect.x;
        armor.right_lightbar.bottom.y += roi_rect.y;
    }

    std::cout << "[Debug] Detector found " << armors.size() << " armors in this frame." << std::endl;
    return armors;
}

cv::Mat Detector::preprocess_image(const cv::Mat &frame)
{
    cv::Mat gray_img;
    cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
    
    return binary_img;
}

void Detector::find_lightbars(const cv::Mat &frame, std::vector<LightBar> &light_bars)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (const auto &contour : contours) {
        if (contour.size() < 5) {
            continue;
        }

        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        
        // 几何形状检查 (这里缺少 check_geometry 的具体实现, 我们先用基本逻辑替代)
        float aspect_ratio = std::max(rotated_rect.size.width, rotated_rect.size.height) /
                             std::min(rotated_rect.size.width, rotated_rect.size.height);
        if (aspect_ratio < 1.2 || rotated_rect.size.area() < 10) {
            continue;
        }

        LightBar light_bar;
        light_bar.center = rotated_rect.center;
        light_bar.angle = rotated_rect.angle;
        light_bar.width = rotated_rect.size.width;
        light_bar.height = rotated_rect.size.height;
        light_bar.color = enemy_color_;

        cv::Point2f rect_points[4];
        rotated_rect.points(rect_points);
        auto top_point = *std::min_element(std::begin(rect_points), std::end(rect_points), [](const cv::Point2f &a, const cv::Point2f &b) {
            return a.y < b.y;
        });
        auto bottom_point = *std::max_element(std::begin(rect_points), std::end(rect_points), [](const cv::Point2f &a, const cv::Point2f &b) {
            return a.y < b.y;
        });
        light_bar.top = top_point;
        light_bar.bottom = bottom_point;
        light_bar.top2bottom = light_bar.bottom - light_bar.top;
        light_bars.push_back(light_bar);
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

            if (left_bar.color != right_bar.color) continue;

            // 几何形状检查
            float angle_diff = std::abs(left_bar.angle - right_bar.angle);
            float height_ratio = std::min(left_bar.height, right_bar.height) /
                                 std::max(left_bar.height, right_bar.height);
            float y_diff = std::abs(left_bar.center.y - right_bar.center.y);
            float distance = cv::norm(left_bar.center - right_bar.center);
            float armor_width_ratio = distance / ((left_bar.height + right_bar.height) / 2.0f);

            if (angle_diff > 7.0f || height_ratio < 0.7f || y_diff > 20.0f || 
                armor_width_ratio < armor_min_width_ratio_ || armor_width_ratio > armor_max_width_ratio_) {
                continue;
            }
            
            cv::Rect rect = cv::boundingRect(std::vector<cv::Point>{
                cv::Point(left_bar.center.x - left_bar.width / 2, left_bar.center.y - left_bar.height / 2),
                cv::Point(right_bar.center.x + right_bar.width / 2, right_bar.center.y + right_bar.height / 2)
            });

            Armor armor(left_bar, right_bar, armors.size(), rect);
            armor.color = enemy_color_;

            // 从旋转矩形中提取角点
            cv::Point2f left_pts[4], right_pts[4];
            cv::RotatedRect left_rrect(left_bar.center, cv::Size2f(left_bar.width, left_bar.height), left_bar.angle);
            cv::RotatedRect right_rrect(right_bar.center, cv::Size2f(right_bar.width, right_bar.height), right_bar.angle);
            left_rrect.points(left_pts);
            right_rrect.points(right_pts);

            // 对角点进行排序 (左上, 右上, 右下, 左下)
            std::sort(left_pts, left_pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y < b.y;
            });
            std::sort(right_pts, right_pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y < b.y;
            });

            armor.corners.push_back(left_pts[0].x < left_pts[1].x ? left_pts[0] : left_pts[1]); // 左上
            armor.corners.push_back(right_pts[0].x > right_pts[1].x ? right_pts[0] : right_pts[1]); // 右上
            armor.corners.push_back(right_pts[2].x > right_pts[3].x ? right_pts[2] : right_pts[3]); // 右下
            armor.corners.push_back(left_pts[2].x < left_pts[3].x ? left_pts[2] : left_pts[3]); // 左下

            armors.push_back(armor);
        }
    }
}


} // namespace armor_task