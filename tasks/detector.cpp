#include "detector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace armor_task
{

    Detector::Detector() : enemy_color_(red),
                           threshold_(95),
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

        // 调试：显示灯条数量
        if (light_bars.size() > 3)
        {
            std::cout << "[Debug] Found " << light_bars.size() << " light bars." << std::endl;
        }

        // 模块三：匹配灯条成装甲板
        match_armors(roi_frame, light_bars, armors);

        // 将检测到的装甲板坐标从ROI坐标系转换回原始图像坐标系
        for (auto &armor : armors)
        {
            armor.box.x += roi_rect.x;
            armor.box.y += roi_rect.y;
            armor.center.x += roi_rect.x;
            armor.center.y += roi_rect.y;

            // 转换角点坐标
            for (auto &corner : armor.corners)
            {
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
        cv::findContours(frame.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (const auto &contour : contours)
        {
            // 轮廓点数过少时跳过
            if (contour.size() < 5)
            {
                continue;
            }

            cv::RotatedRect rotated_rect = cv::minAreaRect(contour);

            // 面积过小的轮廓不是灯条
            if (rotated_rect.size.area() < 5)
            {
                continue;
            }

            // 计算灯条的长宽和比例
            float long_side = std::max(rotated_rect.size.width, rotated_rect.size.height);
            float short_side = std::min(rotated_rect.size.width, rotated_rect.size.height);
            float aspect_ratio = long_side / (short_side + 1e-5);

            // 检查灯条比例是否在范围内（灯条应该是竖长的）
            if (aspect_ratio < min_lightbar_ratio_ || aspect_ratio > max_lightbar_ratio_)
            {
                continue;
            }

            // 检查灯条最小长度
            if (long_side < min_lightbar_length_)
            {
                continue;
            }

            // 检查灯条与竖直方向的角度差（灯条应该大致竖直）
            float angle = rotated_rect.angle;
            // 将角度归一化到 -90 到 90 度范围
            if (angle > 90.0f)
                angle -= 180.0f;
            if (angle < -90.0f)
                angle += 180.0f;

            // 灯条应该接近竖直（角度接近 0 或 180），容许 max_angle_error_ 度的偏差
            float angle_error = std::abs(angle);
            if (angle > 0)
                angle_error = std::min(angle_error, 180.0f - angle_error);

            if (angle_error > max_angle_error_)
            {
                continue;
            }

            // 灯条几何检验通过，添加到列表
            LightBar light_bar;
            light_bar.center = rotated_rect.center;
            light_bar.angle = rotated_rect.angle;
            light_bar.width = rotated_rect.size.width;
            light_bar.height = rotated_rect.size.height;
            light_bar.color = enemy_color_;

            // 提取灯条的上下两个角点
            cv::Point2f rect_points[4];
            rotated_rect.points(rect_points);
            auto top_point = *std::min_element(
                std::begin(rect_points), std::end(rect_points),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.y < b.y; });
            auto bottom_point = *std::max_element(
                std::begin(rect_points), std::end(rect_points),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.y < b.y; });

            light_bar.top = top_point;
            light_bar.bottom = bottom_point;
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

        // 从左到右排序灯条
        std::sort(light_bars.begin(), light_bars.end(),
                  [](const LightBar &a, const LightBar &b)
                  { return a.center.x < b.center.x; });

        // 收集所有候选装甲板对及其匹配评分
        std::vector<std::tuple<size_t, size_t, float>> candidate_pairs;

        for (size_t i = 0; i < light_bars.size() - 1; ++i)
        {
            for (size_t j = i + 1; j < light_bars.size(); ++j)
            {
                const LightBar &left_bar = light_bars[i];
                const LightBar &right_bar = light_bars[j];

                // 颜色必须相同
                if (left_bar.color != right_bar.color)
                    continue;

                // 计算几何特征
                float angle_diff = std::abs(left_bar.angle - right_bar.angle);

                // 高度比：两条灯条的高度应该接近
                float height_ratio = std::min(left_bar.height, right_bar.height) /
                                     std::max(left_bar.height, right_bar.height);

                // Y方向距离：两条灯条应该在大致同一高度
                float y_diff = std::abs(left_bar.center.y - right_bar.center.y);

                // 灯条间距离和宽度比
                float distance = cv::norm(left_bar.center - right_bar.center);
                float avg_height = (left_bar.height + right_bar.height) / 2.0f;
                float armor_width_ratio = distance / (avg_height + 1e-5);

                // 放宽约束以适应正对相机和各种斜角视角
                // 角度差：放宽到 30°
                // 高度比：放宽到 0.3 以适应视角变化
                // y差：放宽到 50 像素
                if (angle_diff > 30.0f || height_ratio < 0.3f || y_diff > 50.0f ||
                    armor_width_ratio < armor_min_width_ratio_ || armor_width_ratio > armor_max_width_ratio_)
                {
                    continue;
                }

                // 计算匹配评分（分数越低越好）
                float score = angle_diff * 1.0f + (1.0f - height_ratio) * 20.0f + y_diff * 0.3f;
                candidate_pairs.push_back(std::make_tuple(i, j, score));
            }
        }

        // 按评分从低到高排序（最佳匹配优先）
        std::sort(candidate_pairs.begin(), candidate_pairs.end(),
                  [](const auto &a, const auto &b)
                  { return std::get<2>(a) < std::get<2>(b); });

        // 贪心选择最佳的灯条对，每条灯条最多只能使用一次
        std::vector<bool> used_bars(light_bars.size(), false);

        for (const auto &[i, j, score] : candidate_pairs)
        {
            // 如果灯条已被使用，跳过
            if (used_bars[i] || used_bars[j])
            {
                continue;
            }

            const LightBar &left_bar = light_bars[i];
            const LightBar &right_bar = light_bars[j];

            // 构建装甲板的边界框
            cv::Rect rect = cv::boundingRect(std::vector<cv::Point>{
                cv::Point(left_bar.center.x - left_bar.width / 2, left_bar.center.y - left_bar.height / 2),
                cv::Point(right_bar.center.x + right_bar.width / 2, right_bar.center.y + right_bar.height / 2)});

            Armor armor(left_bar, right_bar, armors.size(), rect);
            armor.color = enemy_color_;

            // 提取灯条的四个角点
            cv::Point2f left_pts[4], right_pts[4];
            cv::RotatedRect left_rrect(left_bar.center, cv::Size2f(left_bar.width, left_bar.height), left_bar.angle);
            cv::RotatedRect right_rrect(right_bar.center, cv::Size2f(right_bar.width, right_bar.height), right_bar.angle);
            left_rrect.points(left_pts);
            right_rrect.points(right_pts);

            // 角点排序 (上->下)
            std::sort(left_pts, left_pts + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                      { return a.y < b.y; });
            std::sort(right_pts, right_pts + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                      { return a.y < b.y; });

            // 根据 x 坐标排列角点 (左上, 右上, 右下, 左下)
            armor.corners.push_back(left_pts[0].x < left_pts[1].x ? left_pts[0] : left_pts[1]);
            armor.corners.push_back(right_pts[0].x > right_pts[1].x ? right_pts[0] : right_pts[1]);
            armor.corners.push_back(right_pts[2].x > right_pts[3].x ? right_pts[2] : right_pts[3]);
            armor.corners.push_back(left_pts[2].x < left_pts[3].x ? left_pts[2] : left_pts[3]);

            armors.push_back(armor);
            used_bars[i] = true;
            used_bars[j] = true;
        }
    }

} // namespace armor_task