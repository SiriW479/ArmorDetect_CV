#include "../include/armor.hpp"
#include "detector.hpp"
#include "pnp_solver.hpp"
#include "target.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

using namespace armor_task;

// ==========================================
// 1. Cartesian EKF Implementation (Baseline)
// ==========================================
class CartesianKF {
public:
    // State: [x, vx, y, vy, z, vz]
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    double last_timestamp; // seconds
    bool is_initialized = false;

    CartesianKF() {
        x = Eigen::VectorXd::Zero(6);
        P = Eigen::MatrixXd::Identity(6, 6);
        F = Eigen::MatrixXd::Identity(6, 6);
        H = Eigen::MatrixXd::Zero(3, 6);
        H(0, 0) = 1; H(1, 2) = 1; H(2, 4) = 1; // Measure x, y, z

        // Tuning parameters (Simple CV model)
        Q = Eigen::MatrixXd::Identity(6, 6) * 10.0; 
        R = Eigen::MatrixXd::Identity(3, 3) * 0.1;
    }

    void init(const Eigen::Vector3d& pos, double timestamp) {
        x << pos(0), 0, pos(1), 0, pos(2), 0;
        P = Eigen::MatrixXd::Identity(6, 6);
        last_timestamp = timestamp;
        is_initialized = true;
    }

    void predict(double timestamp) {
        if (!is_initialized) return;
        double dt = timestamp - last_timestamp;
        if (dt <= 0) return;
        last_timestamp = timestamp;

        F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt;
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::Vector3d& z_meas) {
        if (!is_initialized) return;
        Eigen::VectorXd z(3);
        z << z_meas(0), z_meas(1), z_meas(2);

        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        x = x + K * y;
        P = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P;
    }

    // Predict future position without updating state
    Eigen::Vector3d predictFuture(double dt) {
        Eigen::MatrixXd F_future = Eigen::MatrixXd::Identity(6, 6);
        F_future(0, 1) = dt; F_future(2, 3) = dt; F_future(4, 5) = dt;
        Eigen::VectorXd x_pred = F_future * x;
        return {x_pred(0), x_pred(2), x_pred(4)};
    }
};

// ==========================================
// 2. Test Logic
// ==========================================

struct PredictionRecord {
    double target_timestamp; 
    std::vector<Eigen::Vector3d> polar_armors; // All 4 armors
    Eigen::Vector3d pred_cartesian;
};

// Config paths
const std::string video_path = "./test_video.avi"; 
const std::string config_path = "./config/demo.yaml";

int main() {
    std::cout << "=== Model Comparison: Polar EKF vs Cartesian EKF ===" << std::endl;

    // 1. Load Config & Init
    auto yaml = YAML::LoadFile(config_path);
    auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
    auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
    
    cv::Mat camera_matrix, distort_coeffs;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> cam_mat(camera_matrix_data.data());
    Eigen::Matrix<double, 1, 5> dist_coeffs_eigen(distort_coeffs_data.data());
    cv::eigen2cv(cam_mat, camera_matrix);
    cv::eigen2cv(dist_coeffs_eigen, distort_coeffs);

    Detector detector;
    PnpSolver pnp_solver(config_path);

    // 2. Open Video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video " << video_path << std::endl;
        return -1;
    }

    // 3. Variables for Tracking
    std::unique_ptr<Target> polar_target = nullptr;
    std::map<int, CartesianKF> cartesian_kfs;
    
    std::deque<PredictionRecord> history;
    double predict_dt = 0.15; // Predict 150ms into the future
    double fps = 30.0; // Assume 30 fps if not available
    if (cap.get(cv::CAP_PROP_FPS) > 0) fps = cap.get(cv::CAP_PROP_FPS);
    
    std::ofstream csv("model_comparison_results.csv");
    csv << "Frame,Time,ArmorID,Error_Polar,Error_Cartesian\n";

    cv::Mat frame;
    int frame_count = 0;
    
    // EKF Params for Polar
    Eigen::VectorXd P0_dig(11);
    P0_dig << 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100; // Simple init

    while (cap.read(frame)) {
        if (frame.empty()) break;
        double current_time = frame_count / fps;
        auto current_tp = std::chrono::steady_clock::time_point(
            std::chrono::microseconds(static_cast<int64_t>(current_time * 1e6))
        );

        // Detect
        auto armors = detector.detect(frame);
        
        Armor* best_armor = nullptr;
        if (!armors.empty()) {
            // Solve PnP for all
            for (auto& armor : armors) {
                pnp_solver._solve_pnp(armor); 
            }
            best_armor = &armors[0];
        }

        if (best_armor) {
            // 1. Update Polar Target
            if (!polar_target) {
                polar_target = std::make_unique<Target>(*best_armor, current_tp, 4, P0_dig); 
            } else {
                polar_target->predict(current_tp);
                polar_target->update(*best_armor);
            }

            // 2. Update Cartesian KF
            int tracker_id = 0; 
            if (cartesian_kfs.find(tracker_id) == cartesian_kfs.end()) {
                cartesian_kfs[tracker_id].init(best_armor->p_world.head(3), current_time);
            } else {
                cartesian_kfs[tracker_id].predict(current_time);
                cartesian_kfs[tracker_id].update(best_armor->p_world.head(3));
            }
        }
        
        // 4. Verify Past Predictions
        while (!history.empty()) {
            auto& rec = history.front();
            if (std::abs(rec.target_timestamp - current_time) < 0.02) { // Match
                if (best_armor) {
                    Eigen::Vector3d truth = best_armor->p_world.head(3);
                    
                    // Cartesian Error
                    double err_cart = (rec.pred_cartesian - truth).norm();
                    
                    // Polar Error: Min distance to any of the 4 predicted armors
                    double err_polar = 1e9;
                    for (const auto& p : rec.polar_armors) {
                        double d = (p - truth).norm();
                        if (d < err_polar) err_polar = d;
                    }
                    
                    // Log
                    csv << frame_count << "," << current_time << "," << best_armor->car_num << "," 
                        << err_polar << "," << err_cart << "\n";
                        
                    if (frame_count % 30 == 0) {
                        std::cout << "Frame " << frame_count << " | Polar Err: " << err_polar 
                                  << " | Cart Err: " << err_cart << std::endl;
                    }
                }
                history.pop_front();
            } else if (rec.target_timestamp < current_time - 0.05) {
                history.pop_front();
            } else {
                break;
            }
        }

        // Store new prediction
        if (best_armor && polar_target) {
             Target temp = *polar_target;
             auto future_tp = std::chrono::steady_clock::time_point(
                std::chrono::microseconds(static_cast<int64_t>((current_time + predict_dt) * 1e6))
             );
             temp.predict(future_tp);
             
             PredictionRecord rec;
             rec.target_timestamp = current_time + predict_dt;
             rec.pred_cartesian = cartesian_kfs[0].predictFuture(predict_dt);
             
             auto list = temp.armor_xyza_list();
             for(const auto& v : list) {
                 rec.polar_armors.push_back(v.head(3));
             }
             history.push_back(rec);
        }

        frame_count++;
    }
    
    std::cout << "Comparison finished. Results saved to model_comparison_results.csv" << std::endl;
    return 0;
}
