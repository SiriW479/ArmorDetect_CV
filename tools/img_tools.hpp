#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
/**
 * @brief 命名空间
 */
#if 0
namespace tools
{
// 绘图函数，依次传入：图像、点
inline void draw_points(cv::Mat &img, const std::vector<cv::Point> &points, const cv::Scalar &color = cv::Scalar(0, 0, 255), int thickness = 2)
{
    std::vector<std::vector<cv::Point>> contours;
    contours.emplace_back(points);
    cv::drawContours(img, contours, -1, color, thickness);
}

inline void draw_points(cv::Mat &img, const std::vector<cv::Point2f> &points, const cv::Scalar &color = cv::Scalar(0, 0, 255), int thickness = 2)
{
    std::vector<cv::Point> int_points(points.begin(), points.end());
    draw_points(img, int_points, color, thickness);
}
// 绘图函数，依次传入：图像、字符串、显示位置
inline void draw_text(cv::Mat &img, const std::string &text, const cv::Point &point, double font_scale = 1.0, const cv::Scalar &color = cv::Scalar(0, 255, 255), int thickness = 2)
{
    cv::putText(img, text, point, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}
inline std::vector<cv::Point2f> project3DPointsTo2D(const std::vector<cv::Point3f> &points_3d, const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs)
{
    std::vector<cv::Point2f> points_2d;
    cv::projectPoints(points_3d, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix, distort_coeffs, points_2d);
    return points_2d;
}
inline void draw_info_box(cv::Mat &image, const std::string &text, int font_scale = 1, int thickness = 2)
{
    // 1. 拆分多行
    std::vector<std::string> lines;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line, '\n'))
    {
        lines.push_back(line);
    }

    // 2. 字体参数
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    int baseline = 0;
    int line_height = 0;
    int max_width = 0;

    // 3. 获取最大宽度和总高度
    for (const auto &l : lines)
    {
        cv::Size text_size = cv::getTextSize(l, font_face, font_scale, thickness, &baseline);
        line_height = std::max(line_height, text_size.height + baseline);
        max_width = std::max(max_width, text_size.width);
    }

    int padding = 10;
    int box_width = max_width + padding * 2;
    int box_height = line_height * lines.size() + padding * 2;

    // 4. 设置背景位置（右下角）
    int x = image.cols - box_width - 20;  // 右边偏移 20px
    int y = image.rows - box_height - 20; // 底部偏移 20px

    // 5. 绘制背景矩形（半透明黑）
    cv::Mat roi = image(cv::Rect(x, y, box_width, box_height));
    cv::Mat overlay;
    roi.copyTo(overlay);
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(box_width, box_height), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);

    // 6. 绘制文字
    for (size_t i = 0; i < lines.size(); ++i)
    {
        cv::putText(image, lines[i], cv::Point(x + padding, y + padding + line_height * (i + 1) - baseline), font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
    }
}

inline void plotArmors(const ArmorArray &armors, cv::Mat &img)
{
    if (armors.empty())
    {
        // 没有检测到装甲板，直接显示原图
        cv::imshow("Armors", img);
        cv::waitKey(1);
        return;
    }

    for (const auto &armor : armors)
    {
        // 1. 绘制检测框
        cv::rectangle(img, armor.box, cv::Scalar(0, 255, 0), 2);

        // 2. 绘制中心点
        cv::circle(img, armor.center, 4, cv::Scalar(0, 0, 255), -1);

        // 3. 绘制灯条位置
        cv::circle(img, armor.left_lightbar.top, 3, cv::Scalar(255, 0, 0), -1);
        cv::circle(img, armor.left_lightbar.bottom, 3, cv::Scalar(255, 0, 0), -1);
        cv::circle(img, armor.right_lightbar.top, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, armor.right_lightbar.bottom, 3, cv::Scalar(0, 0, 255), -1);

        // 4. 装甲板信息字符串
        std::stringstream ss;
        ss << "ID:" << armor.car_num << " Conf:" << std::fixed << std::setprecision(2) << armor.confidence;

        // 颜色信息
        std::string color_str = (armor.color == Color::red ? "RED" : armor.color == Color::blue ? "BLUE" : "PURPLE");
        ss << " " << color_str;

        // 3D 位置信息
        ss << " Pos:[" << std::fixed << std::setprecision(1) << armor.p_camera[0] << "," << armor.p_camera[1] << "," << armor.p_camera[2] << "]";

        // 偏航角
        ss << " Yaw:" << std::fixed << std::setprecision(1) << armor.yaw;

        // 5. 在装甲板框上方绘制文字
        cv::Point text_origin(armor.box.x, std::max(0, armor.box.y - 10));
        cv::putText(img, ss.str(), text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }

    // 绘制完成后再显示
    cv::imshow("Armors", img);
    cv::waitKey(1);
}

inline void plotSingleArmor(const Armor &target, const Armor &armor, cv::Mat &img)
{
    // 判断是否为有效装甲板（这里用 detect_id == -1 代表无效）
    if (armor.detect_id == -1)
    {
        cv::imshow("SingleArmor", img);
        cv::waitKey(1);
        return;
    }

    // 1. 绘制检测框
    cv::rectangle(img, armor.box, cv::Scalar(0, 255, 0), 2);

    // 2. 绘制中心点
    cv::circle(img, target.center, 4, cv::Scalar(0, 0, 255), -1);
    cv::circle(img, armor.center, 4, cv::Scalar(255, 0, 0), -1);

    // 3. 绘制灯条位置
    cv::circle(img, armor.left_lightbar.top, 3, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, armor.left_lightbar.bottom, 3, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, armor.right_lightbar.top, 3, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, armor.right_lightbar.bottom, 3, cv::Scalar(0, 255, 0), -1);

    // 4. 装甲板信息字符串
    std::stringstream ss;
    ss << "ID:" << armor.car_num << " Conf:" << std::fixed << std::setprecision(2) << armor.confidence;

    // 颜色信息
    std::string color_str = (armor.color == Color::red ? "RED" : armor.color == Color::blue ? "BLUE" : "PURPLE");
    ss << " " << color_str;

    // 3D 位置信息
    ss << " Pos:[" << std::fixed << std::setprecision(1) << armor.p_camera[0] << "," << armor.p_camera[1] << "," << armor.p_camera[2] << "]";

    // 偏航角
    ss << " Yaw:" << std::fixed << std::setprecision(1) << armor.yaw;

    // 5. 在装甲板框上方绘制文字
    cv::Point text_origin(armor.box.x, std::max(0, armor.box.y - 10));
    cv::putText(img, ss.str(), text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    // 绘制完成后显示
    cv::imshow("SingleArmor", img);
    cv::waitKey(1);
}

} // namespace tools

#endif // TOOLS__IMG_TOOLS_HPP
