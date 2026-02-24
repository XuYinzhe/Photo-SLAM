#include "include/cuvslam_tracker.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>

const std::string cfg_path = "/home/shaun/Desktop/Photo-SLAM-dev/Photo-SLAM-260206/Photo-SLAM/cfg/run/sofa/orb3-params-l515.yaml";

void PrintPose(const std::string& prefix, uint64_t frame_id, const cuvslam::Pose& pose) {
    std::cout << prefix << " Frame " << frame_id << " pose:\n";
    std::cout << "  Translation: [" 
              << pose.translation[0] << ", "
              << pose.translation[1] << ", "
              << pose.translation[2] << "]\n";
    std::cout << "  Rotation (quat): [" 
              << pose.rotation[0] << ", "
              << pose.rotation[1] << ", "
              << pose.rotation[2] << ", "
              << pose.rotation[3] << "]\n";
}

void TestWithStaticImages() {
    std::cout << "\n========================================\n";
    std::cout << "Test 1: Static Images Test\n";
    std::cout << "========================================\n\n";
    
    try {
        // Create tracker
        auto tracker = std::make_shared<CuVSLAMTracker>(cfg_path);
        
        // Initialize
        if (!tracker->Initialize()) {
            std::cerr << "Failed to initialize tracker\n";
            return;
        }
        
        std::cout << "Tracker initialized successfully\n\n";
        
        // Create dummy static images (640x480)
        cv::Mat rgb = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
        
        // Add some features to the image (simple pattern)
        for (int i = 0; i < 10; i++) {
            cv::circle(rgb, cv::Point(100 + i*50, 240), 20, cv::Scalar(255, 255, 255), -1);
            cv::circle(depth, cv::Point(100 + i*50, 240), 20, cv::Scalar(1000), -1);
        }
        
        std::cout << "Processing 10 static frames...\n";
        
        for (int i = 0; i < 10; i++) {
            int64_t timestamp_ns = i * 33333333LL; // ~30 FPS
            
            uint64_t frame_id = tracker->TrackFrame(timestamp_ns, rgb, depth);
            
            if (frame_id == 0) {
                std::cerr << "Failed to track frame " << i << "\n";
                continue;
            }
            
            std::cout << "Tracked frame " << i << " with frame_id=" << frame_id << "\n";
            
            // Small delay to simulate real-time processing
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Wait for processing to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Get latest pose
        cuvslam::Pose pose;
        uint64_t frame_id;
        int64_t timestamp_ns;
        
        if (tracker->GetLatestPose(pose, frame_id, timestamp_ns)) {
            PrintPose("Latest", frame_id, pose);
        } else {
            std::cout << "No pose available\n";
        }
        
        // Get all poses
        auto all_poses = tracker->GetAllPoses();
        std::cout << "\nTotal tracked poses: " << all_poses.size() << "\n";
        
        // Get frame count
        std::cout << "Total frame count: " << tracker->GetFrameCount() << "\n";
        
        // Check tracking status
        std::cout << "Tracking lost: " << (tracker->IsLost() ? "Yes" : "No") << "\n";
        
        // Save trajectory
        if (tracker->SaveTrajectory("trajectory_static.txt")) {
            std::cout << "\nTrajectory saved to trajectory_static.txt\n";
        }
        
        // Shutdown
        tracker->Shutdown();
        std::cout << "\nTest 1 completed successfully\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test 1 failed with exception: " << e.what() << "\n";
    }
}

void TestWithTUMDataset(const std::string& dataset_path) {
    std::cout << "\n========================================\n";
    std::cout << "Test 2: TUM Dataset Test\n";
    std::cout << "========================================\n\n";
    
    try {
        // Create tracker
        auto tracker = std::make_shared<CuVSLAMTracker>(cfg_path);
        
        // Initialize
        if (!tracker->Initialize()) {
            std::cerr << "Failed to initialize tracker\n";
            return;
        }
        
        std::cout << "Tracker initialized successfully\n\n";
        
        // Read association file
        std::string assoc_file = dataset_path + "/associations.txt";
        std::ifstream file(assoc_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open associations file: " << assoc_file << "\n";
            std::cerr << "Please create associations.txt in TUM format:\n";
            std::cerr << "timestamp_rgb rgb_file timestamp_depth depth_file\n";
            return;
        }
        
        std::string line;
        int frame_count = 0;
        int max_frames = 100; // Process first 100 frames
        
        std::cout << "Processing TUM dataset frames...\n";
        
        while (std::getline(file, line) && frame_count < max_frames) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double ts_rgb, ts_depth;
            std::string rgb_file, depth_file;
            
            if (!(iss >> ts_rgb >> rgb_file >> ts_depth >> depth_file)) {
                continue;
            }
            
            // Read images
            std::string rgb_path = dataset_path + "/" + rgb_file;
            std::string depth_path = dataset_path + "/" + depth_file;
            
            cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
            cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
            
            if (rgb.empty() || depth.empty()) {
                std::cerr << "Failed to read images for frame " << frame_count << "\n";
                continue;
            }
            
            // Convert timestamp to nanoseconds
            int64_t timestamp_ns = static_cast<int64_t>(ts_rgb * 1e9);
            
            // Track frame
            uint64_t frame_id = tracker->TrackFrame(timestamp_ns, rgb, depth);
            
            if (frame_id == 0) {
                std::cerr << "Failed to track frame " << frame_count << "\n";
                continue;
            }
            
            frame_count++;
            
            if (frame_count % 10 == 0) {
                std::cout << "Processed " << frame_count << " frames\n";
                
                // Get latest pose
                cuvslam::Pose pose;
                uint64_t fid;
                int64_t ts;
                if (tracker->GetLatestPose(pose, fid, ts)) {
                    std::cout << "  Current position: ["
                             << pose.translation[0] << ", "
                             << pose.translation[1] << ", "
                             << pose.translation[2] << "]\n";
                }
            }
        }
        
        file.close();
        
        // Wait for processing to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        std::cout << "\nProcessing completed. Total frames: " << frame_count << "\n";
        
        // Get statistics
        auto all_poses = tracker->GetAllPoses();
        std::cout << "Total tracked poses: " << all_poses.size() << "\n";
        std::cout << "Tracking lost: " << (tracker->IsLost() ? "Yes" : "No") << "\n";
        
        // Count keyframes
        int keyframe_count = 0;
        for (const auto& [fid, _] : all_poses) {
            if (tracker->IsKeyframe(fid)) {
                keyframe_count++;
            }
        }
        std::cout << "Total keyframes: " << keyframe_count << "\n";
        
        // Get final landmarks
        auto landmarks = tracker->GetFinalLandmarks();
        std::cout << "Total final landmarks: " << landmarks.size() << "\n";
        
        // Save trajectory
        if (tracker->SaveTrajectory("trajectory_tum.txt")) {
            std::cout << "\nTrajectory saved to trajectory_tum.txt\n";
        }
        
        // Shutdown
        tracker->Shutdown();
        std::cout << "\nTest 2 completed successfully\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test 2 failed with exception: " << e.what() << "\n";
    }
}

void TestThreadSafety() {
    std::cout << "\n========================================\n";
    std::cout << "Test 3: Thread Safety Test\n";
    std::cout << "========================================\n\n";
    
    try {
        // Create tracker
        auto tracker = std::make_shared<CuVSLAMTracker>(cfg_path);
        
        // Initialize
        if (!tracker->Initialize()) {
            std::cerr << "Failed to initialize tracker\n";
            return;
        }
        
        std::cout << "Tracker initialized successfully\n\n";
        
        // Create dummy images
        cv::Mat rgb = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
        
        std::vector<uint64_t> frame_ids;
        std::mutex frame_ids_mutex;
        
        // Thread 1: Submit frames
        std::thread submit_thread([&]() {
            std::cout << "Submit thread started\n";
            for (int i = 0; i < 20; i++) {
                int64_t timestamp_ns = i * 33333333LL;
                uint64_t frame_id = tracker->TrackFrame(timestamp_ns, rgb, depth);
                
                if (frame_id > 0) {
                    std::lock_guard<std::mutex> lock(frame_ids_mutex);
                    frame_ids.push_back(frame_id);
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::cout << "Submit thread finished\n";
        });
        
        // Thread 2: Query poses
        std::thread query_thread([&]() {
            std::cout << "Query thread started\n";
            for (int i = 0; i < 50; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                
                cuvslam::Pose pose;
                uint64_t frame_id;
                int64_t timestamp_ns;
                
                if (tracker->GetLatestPose(pose, frame_id, timestamp_ns)) {
                    if (i % 10 == 0) {
                        std::cout << "Query thread: Latest frame_id=" << frame_id << "\n";
                    }
                }
                
                auto all_poses = tracker->GetAllPoses();
                if (i % 10 == 0) {
                    std::cout << "Query thread: Total poses=" << all_poses.size() << "\n";
                }
            }
            std::cout << "Query thread finished\n";
        });
        
        // Wait for threads
        submit_thread.join();
        query_thread.join();
        
        // Final check
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::cout << "\nFinal statistics:\n";
        std::cout << "Submitted frames: " << frame_ids.size() << "\n";
        std::cout << "Tracked poses: " << tracker->GetAllPoses().size() << "\n";
        std::cout << "Frame counter: " << tracker->GetFrameCount() << "\n";
        
        // Shutdown
        tracker->Shutdown();
        std::cout << "\nTest 3 completed successfully\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test 3 failed with exception: " << e.what() << "\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "===========================================\n";
    std::cout << "CuVSLAM Tracker Test Suite\n";
    std::cout << "===========================================\n";
    
    // Test 1: Static images
    TestWithStaticImages();
    
    // Test 2: TUM dataset (if provided)
    TestWithTUMDataset("/home/shaun/Desktop/pycuvslam/examples/tum/dataset/rgbd_dataset_freiburg3_long_office_household");
    // if (argc > 1) {
    //     std::string dataset_path = argv[1];
    //     if (std::filesystem::exists(dataset_path)) {
    //         TestWithTUMDataset(dataset_path);
    //     } else {
    //         std::cout << "\nSkipping TUM dataset test: path does not exist\n";
    //     }
    // } else {
    //     std::cout << "\nSkipping TUM dataset test: no dataset path provided\n";
    //     std::cout << "Usage: " << argv[0] << " [path_to_tum_dataset]\n";
    // }
    
    // Test 3: Thread safety
    TestThreadSafety();
    
    std::cout << "\n===========================================\n";
    std::cout << "All tests completed\n";
    std::cout << "===========================================\n";
    
    return 0;
}