#pragma once

#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "third_party/cuvslam/include/cuvslam/cuvslam2.h"

struct CuVSLAMFrame {
    std::size_t frame_id;
    int64_t timestamp_ns;
    cv::Mat rgb;
    cv::Mat depth;
};

class CuVSLAMTracker {
public:
    CuVSLAMTracker(std::filesystem::path config_path, bool run_slam = true, bool silent_log = false);
    ~CuVSLAMTracker();

    bool Initialize();
    void Shutdown();
    void Reset();
    void FinishTracking();
    bool IsTrackingFinished() const;

    void SetWaitFinishSlam();
    bool IsWaitFinishSlam() const;
    
    std::size_t TrackFrame(int64_t timestamp_ns, const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_image_path = "", const std::string& dpt_image_path = "");
    std::size_t TrackFrame(float timestamp, const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_image_path = "", const std::string& dpt_image_path = "");
    std::size_t TrackFrame(double timestamp, const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_image_path = "", const std::string& dpt_image_path = "");

    bool GetFrameIds(std::vector<std::size_t>& frame_ids) const;
    bool GetFrame(std::size_t frame_id, cv::Mat& rgb, cv::Mat& depth) const;
    bool GetFrame(std::size_t frame_id, cv::Mat& rgb, cv::Mat& depth, std::string& rgb_image_path, std::string& dpt_image_path) const;
    bool GetFrame(std::size_t frame_id, int64_t& timestamp_ns, cv::Mat& rgb, cv::Mat& depth) const;
    bool GetFrame(std::size_t frame_id, float& timestamp, cv::Mat& rgb, cv::Mat& depth) const;
    bool GetCurrentPose(std::size_t frame_id, cuvslam::Pose& pose, int64_t& timestamp_ns) const;
    bool GetCurrentPose(std::size_t frame_id, std::vector<double>& tum_pose, float& timestamp) const;
    bool GetFinalSlamPose(std::size_t frame_id, std::vector<double>& tum_pose, float& timestamp) const;
    bool GetPose(std::size_t frame_id, std::vector<double>& tum_pose, float& timestamp) const;
    bool GetLatestPose(cuvslam::Pose& pose, std::size_t& frame_id, int64_t& timestamp_ns) const;
    bool GetLatestPose(std::vector<double>& tum_pose, std::size_t& frame_id, float& timestamp) const;
    bool GetLandmarks(std::size_t frame_id, std::vector<cuvslam::Landmark>& landmarks) const;
    bool GetObservations(std::size_t frame_id, std::vector<cuvslam::Observation>& observations) const;
    bool GetCovariance(std::size_t frame_id, cuvslam::PoseCovariance& covariance) const;
    std::size_t GetFrameCount() const;

    std::map<std::size_t, std::pair<cuvslam::Pose, int64_t>> GetAllPoses() const;
    std::unordered_map<std::size_t, cuvslam::Vector3f> GetFinalLandmarks() const;

    bool GetCameraParameters(int& width, int& height, float& fx, float& fy, float& cx, float& cy, float& fps, float& depth_scale) const;
    
    bool IsLost() const;
    bool IsKeyframe(std::size_t frame_id) const;
    
    bool SaveTrajectory(const std::filesystem::path& filedir) const;
    bool SaveOdomTrajectory(const std::filesystem::path& filepath) const;
    bool SaveSlamTrajectory(const std::filesystem::path& filepath) const;
    bool SaveFinalSlamTrajectory(const std::filesystem::path& filepath) const;

    // orb parameters (optional, can be set in config file)
    float orb_scaleFactor_;
    int orb_nFeatures_;
    int orb_nLevels_;
    int orb_iniThFAST_;
    int orb_minThFAST_;

private:
    bool LoadConfig(std::filesystem::path config_path);
    
    void TrackingThreadLoop();
    
    void ProcessFrame(std::size_t frame_id, int64_t timestamp_ns, const cv::Mat& rgb, const cv::Mat& depth);
    
    cuvslam::Image MatToImage(const cv::Mat& mat, int64_t timestamp_ns, 
                              cuvslam::ImageData::Encoding encoding);
    
    void StoreTrackingResult(std::size_t frame_id, int64_t timestamp_ns,
                            const cuvslam::Pose& slam_pose,
                            const cuvslam::PoseEstimate& pose_estimate,
                            const std::vector<cuvslam::Landmark>* landmarks = nullptr,
                            const std::vector<cuvslam::Observation>* observations = nullptr,
                            bool is_keyframe = false);

    // Configuration
    bool run_slam_;
    bool silent_log_;
    cuvslam::Camera camera_;
    cuvslam::Odometry::RGBDSettings rgbd_settings_;
    cuvslam::Odometry::Config odom_config_;
    cuvslam::Slam::Config slam_config_;
    cuvslam::Rig rig_;

    std::shared_ptr<cuvslam::Odometry> tracker_;
    std::shared_ptr<cuvslam::Slam> slam_;
    
    // lr camera parameters
    int cam_width_, cam_height_;
    float cam_fx_, cam_fy_, cam_cx_, cam_cy_;
    float cam_fps_;
    float cam_depth_scale_;

    // Threading
    std::unique_ptr<std::thread> tracking_thread_;
    std::atomic<bool> tracking_finished_;
    std::atomic<bool> shutdown_requested_;
    std::atomic<bool> is_initialized_;
    std::atomic<bool> tracking_lost_;
    std::atomic<bool> wait_slam_finish_;
    
    // Frame queue
    std::queue<std::shared_ptr<CuVSLAMFrame>> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    static constexpr size_t MAX_QUEUE_SIZE = 100;
    
    // Storage
    mutable std::mutex data_mutex_;
    std::set<std::size_t> frame_ids_;
    std::map<std::size_t, std::pair<cuvslam::Pose, int64_t>> odom_poses_;
    std::map<std::size_t, std::pair<cuvslam::Pose, int64_t>> slam_poses_;
    std::unordered_map<std::size_t, cuvslam::PoseCovariance> covariances_;
    std::unordered_map<std::size_t, std::vector<cuvslam::Landmark>> landmarks_;
    std::unordered_map<std::size_t, std::vector<cuvslam::Observation>> observations_;
    std::unordered_map<std::size_t, bool> keyframes_;
    std::unordered_map<std::size_t, std::shared_ptr<CuVSLAMFrame>> frames_;
    std::unordered_map<std::size_t, std::string> rgb_image_paths_;
    std::unordered_map<std::size_t, std::string> dpt_image_paths_;
    std::map<std::size_t, std::pair<cuvslam::Pose, int64_t>> final_slam_poses_;
    
    // Latest tracking info
    std::size_t latest_frame_id_;
    int64_t latest_timestamp_ns_;
    cuvslam::Pose latest_pose_;
    
    // Frame counter
    std::atomic<std::size_t> frame_counter_;
};