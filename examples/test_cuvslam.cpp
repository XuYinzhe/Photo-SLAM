#include "third_party/cuvlsam/include/cuvslam/cuvslam2.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>  // For image loading

namespace fs = std::filesystem;

// Helper function to load image as ImageData
cuvslam::ImageData loadImage(const std::string& path, cuvslam::ImageData::Encoding encoding) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    
    cuvslam::ImageData image_data;
    image_data.width = img.cols;
    image_data.height = img.rows;
    image_data.pitch = img.step[0];
    image_data.encoding = encoding;
    image_data.is_gpu_mem = false;
    
    // Ensure contiguous memory
    if (!img.isContinuous()) {
        img = img.clone();
    }
    
    if (encoding == cuvslam::ImageData::Encoding::RGB) {
        // Convert RGB to BGR for cuvslam
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        image_data.data_type = cuvslam::ImageData::DataType::UINT8;
    } else if (encoding == cuvslam::ImageData::Encoding::MONO) {
        if (img.type() == CV_8UC1) {
            image_data.data_type = cuvslam::ImageData::DataType::UINT8;
        } else if (img.type() == CV_16UC1) {
            image_data.data_type = cuvslam::ImageData::DataType::UINT16;
        } else if (img.type() == CV_32FC1) {
            image_data.data_type = cuvslam::ImageData::DataType::FLOAT32;
        } else {
            throw std::runtime_error("Unsupported depth image type");
        }
    }
    
    image_data.pixels = img.data;
    return image_data;
}

// Helper function to load timestamps from log file
std::unordered_map<std::string, int64_t> loadTimestamps(const std::string& txt_file) {
    std::unordered_map<std::string, int64_t> time_dict;
    std::ifstream file(txt_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open timestamp file: " + txt_file);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string timestamp_str, frame_name;
        if (iss >> timestamp_str >> frame_name) {
            // Remove newline if present
            frame_name.erase(std::remove(frame_name.begin(), frame_name.end(), '\n'), frame_name.end());
            int64_t timestamp_ns = static_cast<int64_t>(std::stod(timestamp_str) * 1e6);
            time_dict[frame_name] = timestamp_ns;
        }
    }
    
    return time_dict;
}

// Helper function to get sorted file list
std::vector<std::string> getSortedFiles(const std::string& directory) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().filename().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv) {
    try {
        // Check cuVSLAM version
        int32_t major, minor;
        auto version = cuvslam::GetVersion(&major, &minor);
        std::cout << "cuVSLAM Version: " << version << " (Major: " << major << ", Minor: " << minor << ")" << std::endl;
        
        // Set verbosity
        cuvslam::SetVerbosity(1);
        
        // Warm up GPU
        std::cout << "Warming up GPU..." << std::endl;
        cuvslam::WarmUpGPU();
        
        // Configuration
        std::string seq = "sofa1";
        std::string data_dir = "/home/shaun/Desktop/Photo-SLAM-dev/Photo-SLAM-monogs250918/align_dataset_" + seq + "/data/cam1/";
        std::string rgb_dir = data_dir + "rgb";
        std::string depth_dir = data_dir + "depth";
        std::string txt_file = "/home/shaun/Desktop/Photo-SLAM-dev/Photo-SLAM-monogs250918/align_dataset_" + seq + "/raw/cam1/log.txt";
        std::string save_txt = "/home/shaun/Desktop/pycuvslam/l515/lr/" + seq + "_cpp.txt";
        
        // Load dataset file lists
        auto rgb_files = getSortedFiles(rgb_dir);
        auto depth_files = getSortedFiles(depth_dir);
        auto time_dict = loadTimestamps(txt_file);
        
        std::cout << "Found " << rgb_files.size() << " RGB frames" << std::endl;
        std::cout << "Found " << depth_files.size() << " depth frames" << std::endl;
        std::cout << "Found " << time_dict.size() << " timestamps" << std::endl;
        
        // Setup camera (L515 parameters)
        cuvslam::Camera camera;
        camera.size = {640, 480};
        camera.focal = {598.523f, 598.536f};
        camera.principal = {316.782f, 252.758f};
        camera.rig_from_camera = cuvslam::Pose();  // Identity pose
        camera.distortion.model = cuvslam::Distortion::Model::Pinhole;
        
        // Setup RGBD settings
        cuvslam::Odometry::RGBDSettings rgbd_settings;
        rgbd_settings.depth_scale_factor = 4000.0f;
        rgbd_settings.depth_camera_id = 0;
        rgbd_settings.enable_depth_stereo_tracking = false;
        
        // Setup odometry configuration
        cuvslam::Odometry::Config odom_config;
        odom_config.async_sba = true;
        odom_config.enable_final_landmarks_export = true;
        odom_config.enable_observations_export = true;
        odom_config.enable_landmarks_export = true;
        odom_config.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
        odom_config.rgbd_settings = rgbd_settings;
        odom_config.use_gpu = true;
        
        // Create rig
        cuvslam::Rig rig;
        rig.cameras.push_back(camera);
        
        // Create tracker
        std::cout << "Creating tracker..." << std::endl;
        cuvslam::Odometry tracker(rig, odom_config);
        
        // Prepare trajectory output
        std::ofstream traj_file(save_txt);
        if (!traj_file.is_open()) {
            std::cerr << "Warning: Could not open trajectory file for writing" << std::endl;
        }
        
        std::vector<cuvslam::Vector3f> trajectory;
        std::vector<cuvslam::PoseStamped> loop_closure_poses;
        
        // Process frames
        int frame_id = 0;
        for (size_t i = 0; i < rgb_files.size() && i < depth_files.size(); ++i) {
            std::string rgb_path = rgb_dir + "/" + rgb_files[i];
            std::string depth_path = depth_dir + "/" + depth_files[i];
            
            // Extract frame name for timestamp lookup
            std::string frame_name = rgb_files[i];
            frame_name = frame_name.substr(0, frame_name.find_last_of('.'));
            frame_name = frame_name.substr(frame_name.find('_') + 1);
            
            int64_t timestamp_ns = time_dict[frame_name];
            
            // Load images (keep cv::Mat alive during tracking)
            cv::Mat rgb_mat = cv::imread(rgb_path, cv::IMREAD_COLOR);
            cv::Mat depth_mat = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
            
            if (rgb_mat.empty() || depth_mat.empty()) {
                std::cerr << "Warning: Failed to read images at frame " << frame_id << std::endl;
                continue;
            }
            
            // Convert to BGR if needed
            cv::cvtColor(rgb_mat, rgb_mat, cv::COLOR_RGB2BGR);
            
            // Prepare image data
            cuvslam::Image rgb_image;
            rgb_image.pixels = rgb_mat.data;
            rgb_image.width = rgb_mat.cols;
            rgb_image.height = rgb_mat.rows;
            rgb_image.pitch = rgb_mat.step[0];
            rgb_image.encoding = cuvslam::ImageData::Encoding::RGB;
            rgb_image.data_type = cuvslam::ImageData::DataType::UINT8;
            rgb_image.is_gpu_mem = false;
            rgb_image.timestamp_ns = timestamp_ns;
            rgb_image.camera_index = 0;
            
            cuvslam::Image depth_image;
            depth_image.pixels = depth_mat.data;
            depth_image.width = depth_mat.cols;
            depth_image.height = depth_mat.rows;
            depth_image.pitch = depth_mat.step[0];
            depth_image.encoding = cuvslam::ImageData::Encoding::MONO;
            depth_image.data_type = cuvslam::ImageData::DataType::UINT16;
            depth_image.is_gpu_mem = false;
            depth_image.timestamp_ns = timestamp_ns;
            depth_image.camera_index = 0;
            
            // Track
            cuvslam::Odometry::ImageSet images = {rgb_image};
            cuvslam::Odometry::ImageSet depths = {depth_image};
            
            auto pose_estimate = tracker.Track(images, {}, depths);
            
            if (!pose_estimate.world_from_rig.has_value()) {
                std::cerr << "Warning: Tracking failed at frame " << frame_id << std::endl;
                frame_id++;
                continue;
            }
            
            auto odom_pose = pose_estimate.world_from_rig->pose;
            trajectory.push_back(odom_pose.translation);
            
            // Save trajectory in TUM format
            if (traj_file.is_open()) {
                traj_file << std::fixed << std::setprecision(6)
                         << (timestamp_ns / 1e6) << " "
                         << odom_pose.translation[0] << " "
                         << odom_pose.translation[1] << " "
                         << odom_pose.translation[2] << " "
                         << odom_pose.rotation[0] << " "
                         << odom_pose.rotation[1] << " "
                         << odom_pose.rotation[2] << " "
                         << odom_pose.rotation[3] << std::endl;
            }
            
            // Get observations and landmarks if needed
            auto observations = tracker.GetLastObservations(0);
            auto landmarks = tracker.GetLastLandmarks();
            auto final_landmarks = tracker.GetFinalLandmarks();
            
            if (frame_id % 10 == 0) {
                std::cout << "Frame " << frame_id 
                         << ": Pose = [" << odom_pose.translation[0] << ", "
                         << odom_pose.translation[1] << ", "
                         << odom_pose.translation[2] << "]"
                         << ", Observations = " << observations.size()
                         << ", Landmarks = " << landmarks.size()
                         << ", Final Landmarks = " << final_landmarks.size()
                         << std::endl;
            }
            
            frame_id++;
        }
        
        std::cout << "Processing complete. Processed " << frame_id << " frames." << std::endl;
        std::cout << "Trajectory saved to: " << save_txt << std::endl;
        
        if (traj_file.is_open()) {
            traj_file.close();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}