#include "include/cuvslam_tracker.h"

CuVSLAMTracker::CuVSLAMTracker(std::filesystem::path config_path, bool run_slam, bool silent_log) 
    : run_slam_(run_slam)
    , silent_log_(silent_log)
    , tracking_finished_(false)
    , shutdown_requested_(false)
    , is_initialized_(false)
    , tracking_lost_(false)
    , latest_frame_id_(0)
    , latest_timestamp_ns_(0)
    , frame_counter_(0) 
{
    
    if (!this->LoadConfig(config_path)) {
        throw std::runtime_error("Failed to load configuration from " + config_path.string());
    }
    
    std::cout << "[CuVSLAMTracker::CuVSLAMTracker] Configuration loaded successfully" << std::endl;
}

CuVSLAMTracker::~CuVSLAMTracker() {
    this->Shutdown();
}

bool CuVSLAMTracker::LoadConfig(std::filesystem::path config_path) {
    // ORB-SLAM3 format YAML configuration parsing
    cv::FileStorage cfg(config_path.string().c_str(), cv::FileStorage::READ);
    if(!cfg.isOpened()) {
       std::cerr << "[Error] Failed to open config file at: " << config_path << std::endl;
       return false;
    }
    std::cout << "[CuVSLAMTracker::LoadConfig] Reading parameters from " << config_path << std::endl;

    // Basic camera parameters
    this->cam_width_ = cfg["Camera.width"].operator int();
    this->cam_height_ = cfg["Camera.height"].operator int();
    this->cam_fx_ = cfg["Camera1.fx"].operator float();
    this->cam_fy_ = cfg["Camera1.fy"].operator float();
    this->cam_cx_ = cfg["Camera1.cx"].operator float();
    this->cam_cy_ = cfg["Camera1.cy"].operator float();
    this->cam_fps_ = cfg["Camera.fps"].operator float();
    this->cam_depth_scale_ = cfg["RGBD.DepthMapFactor"].operator float();

    std::cout << "[CuVSLAMTracker::LoadConfig] Camera: " << cam_width_ << "x" << cam_height_ 
              << ", fx=" << cam_fx_ << ", fy=" << cam_fy_ 
              << ", cx=" << cam_cx_ << ", cy=" << cam_cy_ << std::endl;
    std::cout << "[CuVSLAMTracker::LoadConfig] Depth scale: " << cam_depth_scale_ << std::endl;

    // Setup camera
    this->camera_.size = {this->cam_width_, this->cam_height_};
    this->camera_.principal = {this->cam_cx_, this->cam_cy_};
    this->camera_.focal = {this->cam_fx_, this->cam_fy_};
    this->camera_.rig_from_camera = cuvslam::Pose();  // Identity pose
    this->camera_.distortion.model = cuvslam::Distortion::Model::Pinhole;
    this->camera_.distortion.parameters.clear();  // No distortion for pinhole

    // Add camera to rig
    this->rig_.cameras.clear();
    this->rig_.cameras.push_back(this->camera_);

    // Setup RGBD settings
    this->rgbd_settings_.depth_scale_factor = this->cam_depth_scale_;
    this->rgbd_settings_.depth_camera_id = 0;
    this->rgbd_settings_.enable_depth_stereo_tracking = false;

    // Setup odometry configuration
    this->odom_config_.async_sba = true;
    this->odom_config_.enable_final_landmarks_export = true;
    this->odom_config_.enable_observations_export = true;
    this->odom_config_.enable_landmarks_export = true;
    this->odom_config_.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
    this->odom_config_.rgbd_settings = this->rgbd_settings_;
    this->odom_config_.use_gpu = true;
    this->odom_config_.use_motion_model = true;
    this->odom_config_.use_denoising = false;

    // Setup SLAM configuration
    this->slam_config_.use_gpu = true;
    this->slam_config_.enable_reading_internals = true;
    this->slam_config_.throttling_time_ms = 1000;

    // Optional ORB parameters
    this->orb_scaleFactor_ = cfg["ORBextractor.scaleFactor"].operator float();
    this->orb_nFeatures_ = cfg["ORBextractor.nFeatures"].operator int();
    this->orb_nLevels_ = cfg["ORBextractor.nLevels"].operator int();
    this->orb_iniThFAST_ = cfg["ORBextractor.iniThFAST"].operator int();
    this->orb_minThFAST_ = cfg["ORBextractor.minThFAST"].operator int();

    cfg.release();
    return true;
}

bool CuVSLAMTracker::Initialize() {
    if (is_initialized_) {
        std::cerr << "[CuVSLAMTracker::Initialize] Already initialized" << std::endl;
        return false;
    }
    
    try {
        std::cout << "[CuVSLAMTracker::Initialize] Warming up GPU..." << std::endl;
        cuvslam::WarmUpGPU();
        
        std::cout << "[CuVSLAMTracker::Initialize] Creating odometry tracker..." << std::endl;
        this->tracker_ = std::make_shared<cuvslam::Odometry>(this->rig_, this->odom_config_);
        if (this->run_slam_) {
            std::cout << "[CuVSLAMTracker::Initialize] Creating SLAM system..." << std::endl;
            auto primary_cameras = this->tracker_->GetPrimaryCameras();
            this->slam_ = std::make_shared<cuvslam::Slam>(this->rig_, primary_cameras, this->slam_config_);
        }

        this->shutdown_requested_ = false;
        this->tracking_lost_ = false;
        this->tracking_thread_ = std::make_unique<std::thread>(&CuVSLAMTracker::TrackingThreadLoop, this);
        
        this->is_initialized_ = true;
        std::cout << "[CuVSLAMTracker::Initialize] Initialization complete" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[CuVSLAMTracker::Initialize] Initialization failed: " << e.what() << std::endl;
        this->is_initialized_ = false;
        return false;
    }
}

void CuVSLAMTracker::FinishTracking() {
    std::cout << "[CuVSLAMTracker::FinishTracking] Tracking finished signal set." << std::endl;
    this->tracking_finished_.store(true);

    std::lock_guard<std::mutex> lock(this->data_mutex_);
    if (this->wait_slam_finish_.load() && this->final_slam_poses_.empty()) {
        std::vector<cuvslam::PoseStamped> final_poses;
        try {
            this->slam_->GetAllSlamPoses(final_poses);
        } catch (const std::exception& e) {
            std::cerr << "[CuVSLAMTracker::FinishTracking] Failed to get final SLAM poses: " << e.what() << std::endl;
            return ;
        }

        std::size_t final_frame_id = 0;
        for (const auto& pose_stamped : final_poses) {
            this->final_slam_poses_[final_frame_id++] = {pose_stamped.pose, pose_stamped.timestamp_ns};
        }
        std::cout << "[CuVSLAMTracker::FinishTracking] Cached final SLAM poses for " << this->final_slam_poses_.size() << " frames." << std::endl;
    }
}

bool CuVSLAMTracker::IsTrackingFinished() const {
    return this->tracking_finished_.load();
}

void CuVSLAMTracker::SetWaitFinishSlam() {
    std::cout<< "[CuVSLAMTracker::SetWaitFinishSlam] Waiting for SLAM to finish..." << std::endl;
    this->wait_slam_finish_.store(true);
}

bool CuVSLAMTracker::IsWaitFinishSlam() const {
    return this->wait_slam_finish_.load();
}

void CuVSLAMTracker::Shutdown() {
    if (!is_initialized_) {
        return;
    }
    
    std::cout << "[CuVSLAMTracker::Shutdown] Shutting down..." << std::endl;

    this->shutdown_requested_.store(true);
    
    // Wait for tracking thread
    if (this->tracking_thread_ && this->tracking_thread_->joinable()) {
        this->queue_cv_.notify_all();
        this->tracking_thread_->join();
    }
    
    std::cout << "[CuVSLAMTracker::Shutdown] Shutdown complete" << std::endl;
}

std::size_t CuVSLAMTracker::TrackFrame(float timestamp, const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_image_path, const std::string& dpt_image_path) {
    int64_t timestamp_ns = static_cast<int64_t>(timestamp * 1e9);
    return this->TrackFrame(timestamp_ns, rgb, depth, rgb_image_path, dpt_image_path);
}

std::size_t CuVSLAMTracker::TrackFrame(double timestamp, const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_image_path, const std::string& dpt_image_path) {
    int64_t timestamp_ns = static_cast<int64_t>(timestamp * 1e9);
    return this->TrackFrame(timestamp_ns, rgb, depth, rgb_image_path, dpt_image_path);
}

std::size_t CuVSLAMTracker::TrackFrame(int64_t timestamp_ns, const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_image_path, const std::string& dpt_image_path) {
    if (!this->is_initialized_) {
        std::cerr << "[CuVSLAMTracker::TrackFrame] Not initialized" << std::endl;
        return 0;
    }
    
    if (rgb.empty() || depth.empty()) {
        std::cerr << "[CuVSLAMTracker::TrackFrame] Empty image provided" << std::endl;
        return 0;
    }
    
    std::size_t frame_id = this->frame_counter_++;
    
    // CuVSLAMFrame frame_info;
    auto frame_info = std::make_shared<CuVSLAMFrame>();
    frame_info->frame_id = frame_id;
    frame_info->timestamp_ns = timestamp_ns;
    frame_info->rgb = rgb.clone();
    frame_info->depth = depth.clone();
    
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex_);
        
        if (this->frame_queue_.size() >= MAX_QUEUE_SIZE) {
            std::cerr << "[CuVSLAMTracker::TrackFrame] Frame queue full, dropping frame " << frame_id << std::endl;
            return 0;
        }
        else if (this->frame_queue_.size() >= MAX_QUEUE_SIZE * 0.9) {
            while (this->frame_queue_.size() >= MAX_QUEUE_SIZE * 0.8) {
                std::cout << "[CuVSLAMTracker::TrackFrame] Queue approaching capacity (" << this->frame_queue_.size() << "/" << MAX_QUEUE_SIZE << "), last frame id: " << frame_id << std::endl;
                this->queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
                if (this->shutdown_requested_) return 0;
            }
            std::cout<< "[CuVSLAMTracker::TrackFrame] Resuming frame submission, current queue size: " << this->frame_queue_.size() << std::endl;
        }
        else if (!this->silent_log_) {
            std::cout << "[CuVSLAMTracker::TrackFrame] Enqueuing frame " << frame_id << std::endl;
        }
        
        this->frames_[frame_id] = frame_info;
        this->frame_ids_.insert(frame_id);
        this->frame_queue_.push(frame_info);

        if(!rgb_image_path.empty()) this->rgb_image_paths_[frame_id] = rgb_image_path;
        if(!dpt_image_path.empty()) this->dpt_image_paths_[frame_id] = dpt_image_path;
    }
    
    this->queue_cv_.notify_one();
    return frame_id;
}

bool CuVSLAMTracker::GetFrameIds(std::vector<std::size_t>& frame_ids) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    frame_ids.assign(this->frame_ids_.begin(), this->frame_ids_.end());
    return true;
}

bool CuVSLAMTracker::GetFrame(std::size_t frame_id, int64_t& timestamp_ns, cv::Mat& rgb, cv::Mat& depth) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);

    auto it = this->frames_.find(frame_id);
    if (it == this->frames_.end()) {
        return false;
    }

    timestamp_ns = it->second->timestamp_ns;
    rgb = it->second->rgb.clone();
    depth = it->second->depth.clone();
    return true;
}

bool CuVSLAMTracker::GetFrame(std::size_t frame_id, float& timestamp, cv::Mat& rgb, cv::Mat& depth) const {
    int64_t timestamp_ns;
    if (!this->GetFrame(frame_id, timestamp_ns, rgb, depth)) {
        return false;
    }
    timestamp = timestamp_ns / 1e9f;
    return true;
}

bool CuVSLAMTracker::GetFrame(std::size_t frame_id, cv::Mat& rgb, cv::Mat& depth) const {
    int64_t timestamp_ns;
    return this->GetFrame(frame_id, timestamp_ns, rgb, depth);
}

bool CuVSLAMTracker::GetFrame(std::size_t frame_id, cv::Mat& rgb, cv::Mat& depth, std::string& rgb_image_path, std::string& dpt_image_path) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);

    auto it = this->frames_.find(frame_id);
    if (it == this->frames_.end()) {
        return false;
    }

    rgb = it->second->rgb.clone();
    depth = it->second->depth.clone();
    auto rgb_path_it = this->rgb_image_paths_.find(frame_id);
    if (rgb_path_it != this->rgb_image_paths_.end()) {
        rgb_image_path = rgb_path_it->second;
    } else {
        rgb_image_path.clear();
    }
    auto dpt_path_it = this->dpt_image_paths_.find(frame_id);
    if (dpt_path_it != this->dpt_image_paths_.end()) {
        dpt_image_path = dpt_path_it->second;
    } else {
        dpt_image_path.clear();
    }
    return true;
}

bool CuVSLAMTracker::GetCurrentPose(std::size_t frame_id, cuvslam::Pose& pose, int64_t& timestamp_ns) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    if (this->run_slam_) {
        auto it = this->slam_poses_.find(frame_id);
        if (it == this->slam_poses_.end()) {
            return false;
        }
        
        pose = it->second.first;
        timestamp_ns = it->second.second;
        return true;
    }
    else {

        auto it = this->odom_poses_.find(frame_id);
        if (it == this->odom_poses_.end()) {
            return false;
        }
        
        pose = it->second.first;
        timestamp_ns = it->second.second;
        return true;
    }
}

bool CuVSLAMTracker::GetCurrentPose(std::size_t frame_id, std::vector<double>& tum_pose, float& timestamp) const {
    int64_t timestamp_ns;
    cuvslam::Pose pose;
    if (!this->GetCurrentPose(frame_id, pose, timestamp_ns)) {
        return false;
    }

    timestamp = timestamp_ns / 1e9f;
    tum_pose = {static_cast<double>(pose.translation[0]), static_cast<double>(pose.translation[1]), static_cast<double>(pose.translation[2]),
                static_cast<double>(pose.rotation[0]), static_cast<double>(pose.rotation[1]), static_cast<double>(pose.rotation[2]), static_cast<double>(pose.rotation[3])};
    return true;
}

bool CuVSLAMTracker::GetFinalSlamPose(std::size_t frame_id, std::vector<double>& tum_pose, float& timestamp) const {
    if (!this->run_slam_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    auto it = this->final_slam_poses_.find(frame_id);
    if (it == this->final_slam_poses_.end()) {
        return false;
    }
        
    const auto& pose = it->second.first;
    const auto& timestamp_ns = it->second.second;

    timestamp = timestamp_ns / 1e9f;
    tum_pose = {static_cast<double>(pose.translation[0]), static_cast<double>(pose.translation[1]), static_cast<double>(pose.translation[2]),
                static_cast<double>(pose.rotation[0]), static_cast<double>(pose.rotation[1]), static_cast<double>(pose.rotation[2]), static_cast<double>(pose.rotation[3])};
    return true;
    
}

bool CuVSLAMTracker::GetPose(std::size_t frame_id, std::vector<double>& tum_pose, float& timestamp) const {
    if (this->wait_slam_finish_.load()) {
        return this->GetFinalSlamPose(frame_id, tum_pose, timestamp);
    }
    else {
        return this->GetCurrentPose(frame_id, tum_pose, timestamp);
    }
}

bool CuVSLAMTracker::GetLatestPose(cuvslam::Pose& pose, std::size_t& frame_id, int64_t& timestamp_ns) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    if (this->latest_frame_id_ == 0) {
        return false;
    }
    
    pose = this->latest_pose_;
    frame_id = this->latest_frame_id_;
    timestamp_ns = this->latest_timestamp_ns_;
    return true;
}

bool CuVSLAMTracker::GetLatestPose(std::vector<double>& tum_pose, std::size_t& frame_id, float& timestamp) const {
    int64_t timestamp_ns;
    cuvslam::Pose pose;
    if (!this->GetLatestPose(pose, frame_id, timestamp_ns)) {
        return false;
    }
    timestamp = timestamp_ns / 1e9f;
    tum_pose = {static_cast<double>(pose.translation[0]), static_cast<double>(pose.translation[1]), static_cast<double>(pose.translation[2]),
                static_cast<double>(pose.rotation[0]), static_cast<double>(pose.rotation[1]), static_cast<double>(pose.rotation[2]), static_cast<double>(pose.rotation[3])};
    return true;
}

std::map<std::size_t, std::pair<cuvslam::Pose, int64_t>> CuVSLAMTracker::GetAllPoses() const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    if(this->run_slam_){
        return this->slam_poses_;
    }
    else {
        return this->odom_poses_;
    }
}

bool CuVSLAMTracker::GetLandmarks(std::size_t frame_id, std::vector<cuvslam::Landmark>& landmarks) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    auto it = this->landmarks_.find(frame_id);
    if (it == this->landmarks_.end()) {
        return false;
    }
    
    landmarks = it->second;
    return true;
}

bool CuVSLAMTracker::GetObservations(std::size_t frame_id, std::vector<cuvslam::Observation>& observations) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    auto it = this->observations_.find(frame_id);
    if (it == this->observations_.end()) {
        return false;
    }
    
    observations = it->second;
    return true;
}

std::unordered_map<std::size_t, cuvslam::Vector3f> CuVSLAMTracker::GetFinalLandmarks() const {
    if (!this->tracker_) {
        return {};
    }
    
    try {
        return this->tracker_->GetFinalLandmarks();
    } catch (const std::exception& e) {
        std::cerr << "[CuVSLAMTracker] Failed to get final landmarks: " << e.what() << std::endl;
        return {};
    }
}

bool CuVSLAMTracker::GetCovariance(std::size_t frame_id, cuvslam::PoseCovariance& covariance) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    auto it = this->covariances_.find(frame_id);
    if (it == this->covariances_.end()) {
        return false;
    }
    
    covariance = it->second;
    return true;
}

std::size_t CuVSLAMTracker::GetFrameCount() const {
    return this->frame_counter_.load()-1;
}

bool CuVSLAMTracker::GetCameraParameters(int& width, int& height, float& fx, float& fy, float& cx, float& cy, float& fps, float& depth_scale) const {
    width = this->cam_width_;
    height = this->cam_height_;
    fx = this->cam_fx_;
    fy = this->cam_fy_;
    cx = this->cam_cx_;
    cy = this->cam_cy_;
    fps = this->cam_fps_;
    depth_scale = this->cam_depth_scale_;

    return true;
}

bool CuVSLAMTracker::IsLost() const {
    return this->tracking_lost_.load();
}

bool CuVSLAMTracker::IsKeyframe(std::size_t frame_id) const {
    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    auto it = this->keyframes_.find(frame_id);
    if (it == this->keyframes_.end()) {
        return false;
    }
    
    return it->second;
}

bool CuVSLAMTracker::SaveTrajectory(const std::filesystem::path& filedir) const {
    std::cout<< "[CuVSLAMTracker::SaveTrajectory] Saving trajectories to directory: " << filedir << std::endl;

    bool odom_saved, slam_saved, final_slam_saved;
    odom_saved = slam_saved = final_slam_saved = false;

    odom_saved = this->SaveOdomTrajectory(filedir / "tracking_odom_trajectory.txt");

    if (this->run_slam_)
        slam_saved = this->SaveSlamTrajectory(filedir / "tracking_slam_trajectory.txt");

    if (this->run_slam_ && this->wait_slam_finish_.load())
        final_slam_saved = this->SaveFinalSlamTrajectory(filedir / "tracking_final_slam_trajectory.txt");
    
    return odom_saved && (!this->run_slam_ || slam_saved) && (!this->run_slam_ || !this->wait_slam_finish_.load() || final_slam_saved);
}

bool CuVSLAMTracker::SaveOdomTrajectory(const std::filesystem::path& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[CuVSLAMTracker::SaveOdomTrajectory] Failed to open odometry trajectory file: " << filepath << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    // Write in TUM format: timestamp tx ty tz qx qy qz qw
    for (const auto& [frame_id, pose_time_pair] : this->odom_poses_) {
        const auto& pose = pose_time_pair.first;
        const auto& timestamp_ns = pose_time_pair.second;
        
        file << std::fixed << std::setprecision(6)
            << (timestamp_ns / 1e9) << " "
            << pose.translation[0] << " "
            << pose.translation[1] << " "
            << pose.translation[2] << " "
            << pose.rotation[0] << " "
            << pose.rotation[1] << " "
            << pose.rotation[2] << " "
            << pose.rotation[3] << "\n";
    }
    
    file.close();
    
    std::cout << "[CuVSLAMTracker] Saved odometry trajectory with " << this->odom_poses_.size() << " poses to " << filepath << std::endl;
    
    return true;
}

bool CuVSLAMTracker::SaveSlamTrajectory(const std::filesystem::path& filepath) const {
    if (!this->run_slam_) {
        std::cerr << "[CuVSLAMTracker::SaveSlamTrajectory] SLAM tracking not enabled, cannot save SLAM trajectory." << std::endl;
        return false;
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[CuVSLAMTracker::SaveSlamTrajectory] Failed to open SLAM trajectory file: " << filepath << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    // Write in TUM format: timestamp tx ty tz qx qy qz qw
    for (const auto& [frame_id, pose_time_pair] : this->slam_poses_) {
        const auto& pose = pose_time_pair.first;
        const auto& timestamp_ns = pose_time_pair.second;
        
        file << std::fixed << std::setprecision(6)
            << (timestamp_ns / 1e9) << " "
            << pose.translation[0] << " "
            << pose.translation[1] << " "
            << pose.translation[2] << " "
            << pose.rotation[0] << " "
            << pose.rotation[1] << " "
            << pose.rotation[2] << " "
            << pose.rotation[3] << "\n";
    }
    
    file.close();
    
    std::cout << "[CuVSLAMTracker] Saved SLAM trajectory with " << this->slam_poses_.size() << " poses to " << filepath << std::endl;
    
    return true;
}

bool CuVSLAMTracker::SaveFinalSlamTrajectory(const std::filesystem::path& filepath) const {
    if (!this->run_slam_) {
        std::cerr << "[CuVSLAMTracker::SaveFinalSlamTrajectory] SLAM tracking not enabled, cannot save final SLAM trajectory." << std::endl;
        return false;
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[CuVSLAMTracker::SaveFinalSlamTrajectory] Failed to open final SLAM trajectory file: " << filepath << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(this->data_mutex_);
    
    // Write in TUM format: timestamp tx ty tz qx qy qz qw
    for (const auto& [frame_id, pose_time_pair] : this->final_slam_poses_) {
        const auto& pose = pose_time_pair.first;
        const auto& timestamp_ns = pose_time_pair.second;
        
        file << std::fixed << std::setprecision(6)
            << (timestamp_ns / 1e9) << " "
            << pose.translation[0] << " "
            << pose.translation[1] << " "
            << pose.translation[2] << " "
            << pose.rotation[0] << " "
            << pose.rotation[1] << " "
            << pose.rotation[2] << " "
            << pose.rotation[3] << "\n";
    }
    
    file.close();
    
    std::cout << "[CuVSLAMTracker] Saved final SLAM trajectory with " << this->final_slam_poses_.size() << " poses to " << filepath << std::endl;
    
    return true;
}

void CuVSLAMTracker::Reset() {
    std::cout << "[CuVSLAMTracker::Reset] Reset requested" << std::endl;
    
    // Clear queues
    {
        std::lock_guard<std::mutex> lock(this->queue_mutex_);
        std::queue<std::shared_ptr<CuVSLAMFrame>> empty;
        std::swap(this->frame_queue_, empty);
    }
    
    // Clear stored data
    {
        std::lock_guard<std::mutex> lock(this->data_mutex_);
        this->odom_poses_.clear();
        this->slam_poses_.clear();
        this->covariances_.clear();
        this->landmarks_.clear();
        this->observations_.clear();
        this->keyframes_.clear();
        this->latest_frame_id_ = 0;
        this->latest_timestamp_ns_ = 0;
    }
    
    this->frame_counter_ = 0;
    this->tracking_lost_ = false;
    
    std::cout << "[CuVSLAMTracker::Reset] Reset complete" << std::endl;
}

// ============================================================================
// Private Methods
// ============================================================================

void CuVSLAMTracker::TrackingThreadLoop() {
    std::cout << "[CuVSLAMTracker::TrackingThreadLoop] Tracking thread started" << std::endl;
    
    while (!this->shutdown_requested_) {
        std::shared_ptr<CuVSLAMFrame> frame_info;
        
        // Wait for frame
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex_);
            this->queue_cv_.wait(lock, [this] {
                return !this->frame_queue_.empty() || this->shutdown_requested_;
            });
            
            if (this->shutdown_requested_) break;
            
            if (this->frame_queue_.empty()) continue;
            
            frame_info = this->frame_queue_.front();
            this->frame_queue_.pop();
        }
        
        // Process frame
        try {
            this->ProcessFrame(frame_info->frame_id, frame_info->timestamp_ns, 
                        frame_info->rgb, frame_info->depth);
        } catch (const std::exception& e) {
            std::cerr << "[CuVSLAMTracker::TrackingThreadLoop] Error processing frame " 
                     << frame_info->frame_id << ": " << e.what() << std::endl;
            this->tracking_lost_ = true;
        }
    }
    
    std::cout << "[CuVSLAMTracker::TrackingThreadLoop] Tracking thread stopped" << std::endl;
}

void CuVSLAMTracker::ProcessFrame(
    std::size_t frame_id, int64_t timestamp_ns,
    const cv::Mat& rgb, const cv::Mat& depth
) {
    if (!this->tracker_) {
        throw std::runtime_error("Tracker not initialized");
    }
    
    cuvslam::Odometry::ImageSet images;
    cuvslam::Odometry::ImageSet depths;
    cuvslam::Odometry::ImageSet masks;  // Empty for now
    
    // RGB image
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    cuvslam::Image rgb_image = this->MatToImage(bgr, timestamp_ns, 
                                          cuvslam::ImageData::Encoding::RGB);
    images.push_back(rgb_image);
    
    // Depth image
    cuvslam::Image depth_image = this->MatToImage(depth, timestamp_ns,
                                            cuvslam::ImageData::Encoding::MONO);
    depths.push_back(depth_image);
    
    // Track with odometry
    cuvslam::PoseEstimate pose_estimate = this->tracker_->Track(images, masks, depths);
    
    if (!pose_estimate.world_from_rig.has_value()) {
        std::cerr << "[CuVSLAMTracker] Tracking lost at frame " << frame_id << std::endl;
        this->tracking_lost_ = true;
        return;
    }
    
    this->tracking_lost_ = false;
    
    // Extract pose and covariance
    const auto& pose_with_cov = pose_estimate.world_from_rig.value();
    cuvslam::Pose pose = pose_with_cov.pose;
    cuvslam::PoseCovariance covariance = pose_with_cov.covariance;
    
    // Get odometry state for landmarks and observations
    cuvslam::Odometry::State odom_state;
    bool has_state = false;
    try {
        this->tracker_->GetState(odom_state);
        has_state = true;
    } catch (const std::exception& e) {
        std::cerr << "[CuVSLAMTracker::ProcessFrame] Failed to get odometry state: " << e.what() << std::endl;
    }
    
    // Store tracking result
    cuvslam::Pose slam_pose;
    if (has_state) {
        if (this->run_slam_) {
            // Run SLAM tracking
            slam_pose = this->slam_->Track(odom_state);
        }
        this->StoreTrackingResult(frame_id, timestamp_ns, slam_pose, pose_estimate,
                          &odom_state.landmarks, &odom_state.observations,
                          odom_state.keyframe);
    } else {
        this->StoreTrackingResult(frame_id, timestamp_ns, slam_pose, pose_estimate,
                          nullptr, nullptr, false);
    }
    
    // Debug output
    // if (frame_id % 30 == 0) {  // Every 30 frames
    if (!this->silent_log_) {
        std::cout << "[CuVSLAMTracker::ProcessFrame] Frame " << frame_id 
                  << " Odometry Pose | Translation: [" << pose.translation[0] << ", "
                  << pose.translation[1] << ", " << pose.translation[2] << "]" 
                  << " | Rotation: [" << pose.rotation[0] << ", "
                  << pose.rotation[1] << ", " << pose.rotation[2] << ", " << pose.rotation[3] << "]"
                  << " Slam Pose | Translation: [" << slam_pose.translation[0] << ", "
                  << slam_pose.translation[1] << ", " << slam_pose.translation[2] << "]" 
                  << " | Rotation: [" << slam_pose.rotation[0] << ", "
                  << slam_pose.rotation[1] << ", " << slam_pose.rotation[2] << ", " << slam_pose.rotation[3] << "]";
        if (has_state) {
            std::cout << " | Landmarks: " << odom_state.landmarks.size()
                     << " | Observations: " << odom_state.observations.size()
                     << " | Keyframe: " << (odom_state.keyframe ? "Yes" : "No");
        }
        std::cout << std::endl;
    }
    // }
}

cuvslam::Image CuVSLAMTracker::MatToImage(
    const cv::Mat& mat, int64_t timestamp_ns,
    cuvslam::ImageData::Encoding encoding
) {
    cuvslam::Image image;
    
    // Ensure contiguous memory
    cv::Mat contiguous_mat = mat;
    if (!mat.isContinuous()) {
        contiguous_mat = mat.clone();
    }
    
    image.pixels = contiguous_mat.data;
    image.width = contiguous_mat.cols;
    image.height = contiguous_mat.rows;
    image.pitch = contiguous_mat.step[0];
    image.encoding = encoding;
    image.is_gpu_mem = false;
    image.timestamp_ns = timestamp_ns;
    image.camera_index = 0;
    
    // Set data type based on Mat type
    switch (contiguous_mat.depth()) {
        case CV_8U:
            image.data_type = cuvslam::ImageData::DataType::UINT8;
            break;
        case CV_16U:
            image.data_type = cuvslam::ImageData::DataType::UINT16;
            break;
        case CV_32F:
            image.data_type = cuvslam::ImageData::DataType::FLOAT32;
            break;
        default:
            throw std::runtime_error("Unsupported Mat data type");
    }
    
    return image;
}

void CuVSLAMTracker::StoreTrackingResult(
    std::size_t frame_id, int64_t timestamp_ns,
    const cuvslam::Pose& slam_pose,
    const cuvslam::PoseEstimate& pose_estimate,
    const std::vector<cuvslam::Landmark>* landmarks,
    const std::vector<cuvslam::Observation>* observations,
    bool is_keyframe
) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (!pose_estimate.world_from_rig.has_value()) {
        return;
    }
    
    const auto& pose_with_cov = pose_estimate.world_from_rig.value();
    
    this->odom_poses_[frame_id] = {pose_with_cov.pose, timestamp_ns};
    this->covariances_[frame_id] = pose_with_cov.covariance;

    if (this->run_slam_) {
        this->slam_poses_[frame_id] = {slam_pose, timestamp_ns};
    }
    
    if (landmarks != nullptr) {
        this->landmarks_[frame_id] = *landmarks;
    }
    
    if (observations != nullptr) {
        this->observations_[frame_id] = *observations;
    }
    
    this->keyframes_[frame_id] = is_keyframe;
    
    this->latest_frame_id_ = frame_id;
    this->latest_timestamp_ns_ = timestamp_ns;
    this->latest_pose_ = pose_with_cov.pose;
}

