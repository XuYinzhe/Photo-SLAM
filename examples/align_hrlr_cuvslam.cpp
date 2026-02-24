/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>

#include <opencv2/core/core.hpp>

// #include "ORB-SLAM3/include/System.h"
#include "include/cuvslam_tracker.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

void LoadImagePaths(const std::filesystem::path &strDataPath, 
    std::vector<std::string> &vstrImageFilenamesRGB, 
    std::vector<std::string> &vstrImageFilenamesHR,
    std::vector<std::string> &vstrImageFilenamesD, 
    std::vector<double>& vTimestampsRGBD, 
    std::vector<double>& vTimestampsHR,
    std::vector<std::vector<double>>& vvLRGTPose,
    std::vector<std::vector<double>>& vvHRGTPose,
    double& lr_fps);
void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);
void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);
void LoadPathConfig(std::filesystem::path cfg_path, 
    std::filesystem::path& strORBSLAMVocabularyPath, std::filesystem::path& strORBSLAMParametersPath, std::filesystem::path& strMapperPath, 
    std::filesystem::path& strDataPath, std::filesystem::path& strSavingPath, double& lr_fps, bool& bViewer);

int main(int argc, char **argv)
{
    if (argc != 2){
        std::cerr << "[Error] Wrong arguments input." << std::endl;
    }

    std::filesystem::path strORBSLAMVocabularyPath;
    std::filesystem::path strORBSLAMParametersPath;
    std::filesystem::path strMapperPath; 
    std::filesystem::path strDataPath;
    std::filesystem::path strSavingPath;
    double lr_fps;
    bool bViewer;

    LoadPathConfig(std::string(argv[1]), strORBSLAMVocabularyPath, strORBSLAMParametersPath, strMapperPath, strDataPath, strSavingPath, lr_fps, bViewer);

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesHR;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestampsRGBD = {};
    std::vector<double> vTimestampsHR = {};
    std::vector<std::vector<double>> vvLRGTPose = {};
    std::vector<std::vector<double>> vvHRGTPose = {};
    LoadImagePaths(strDataPath, vstrImageFilenamesRGB, vstrImageFilenamesHR, vstrImageFilenamesD, 
        vTimestampsRGBD, vTimestampsHR, vvLRGTPose, vvHRGTPose, lr_fps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "[main] CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "[main] Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create CuVLSAM 
    std::shared_ptr<CuVSLAMTracker> pCuVSLAMTracker = std::make_shared<CuVSLAMTracker>(strORBSLAMParametersPath, true, false);
    
    if(!pCuVSLAMTracker->Initialize()){
        std::cerr << std::endl << "[main] Failed to initialize CuVSLAMTracker." << std::endl;
        return 1;
    }
    else{
        std::cout << std::endl << "[main] CuVSLAMTracker initialized successfully." << std::endl;
    }

    // Create GaussianMapper
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(pCuVSLAMTracker, strMapperPath, strSavingPath, 0, device_type);
    pGausMapper->setOtherData(vstrImageFilenamesHR, vTimestampsHR, vvLRGTPose, vvHRGTPose);
    std::thread mapper_thread(&GaussianMapper::run_cuvslam, pGausMapper);

    std::vector<float> vTimesTrack;
    vTimesTrack.resize(vstrImageFilenamesRGB.size());

    double time_ni = 1.f/10.f; // !!! debug, should be read from config file
    for(int ni=0; ni<vstrImageFilenamesRGB.size(); ni++){
        // std::cout << "[main] Frame " << ni << ": " << vstrImageFilenamesRGB[ni] << ", " << vstrImageFilenamesD[ni] << std::endl;
    
        cv::Mat imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_COLOR);
        cv::Mat imD = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        // if(imD.type() == CV_16UC3) cv::cvtColor(imD, imD, cv::COLOR_BGR2GRAY);

        if (imRGB.empty())
        {
            std::cerr << std::endl << "[main] Failed to load image at: "
                      << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        if (imD.empty())
        {
            std::cerr << std::endl << "[main] Failed to load depth image at: "
                      << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        float timestamp = ni * (float)time_ni; // !!! debug, should be read from association file

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        pCuVSLAMTracker->TrackFrame(vTimestampsRGBD[ni], imRGB, imD, vstrImageFilenamesRGB[ni], vstrImageFilenamesD[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double time_track = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = time_track;

        double T = 0;
        if (ni < vstrImageFilenamesRGB.size() - 1)
            T = vTimestampsRGBD[ni+1] - vTimestampsRGBD[ni];
        else if (ni > 0)
            T = vTimestampsRGBD[ni] - vTimestampsRGBD[ni-1];

        if (T > time_track) usleep((T - time_track) * 1e6);
    
    }

    std::cout << "[main] Tracking loop finished. Processed " << vstrImageFilenamesRGB.size() << " frames." << std::endl;

    pCuVSLAMTracker->FinishTracking();

    std::cout << "[main] Waiting for mapper to finish processing remaining frames..." << std::endl;

    pGausMapper->waitUntilFinished();

    std::cout << "[main] Mapper finished processing all frames." << std::endl;

    pGausMapper->signalStop(); // !!! please modify if viewer is used

    if (mapper_thread.joinable()) {
        mapper_thread.join();
    }

    pCuVSLAMTracker->Shutdown();

    // saving results

    return 0;
}

void LoadImagePaths(const std::filesystem::path &strDataPath, 
    std::vector<std::string> &vstrImageFilenamesRGB, 
    std::vector<std::string> &vstrImageFilenamesHR,
    std::vector<std::string> &vstrImageFilenamesD, 
    std::vector<double>& vTimestampsRGBD, 
    std::vector<double>& vTimestampsHR,
    std::vector<std::vector<double>>& vvLRGTPose,
    std::vector<std::vector<double>>& vvHRGTPose,
    double& lr_fps)
{

    vstrImageFilenamesRGB.clear();
    vstrImageFilenamesHR.clear();
    vstrImageFilenamesD.clear();
    vTimestampsRGBD.clear();
    vTimestampsHR.clear();

    const std::filesystem::path cam1RGBPath = strDataPath / "cam1" / "rgb";
    const std::filesystem::path cam2RGBPath = strDataPath / "cam2" / "rgb";
    const std::filesystem::path cam1DepthPath = strDataPath / "cam1" / "depth";
    const std::filesystem::path posesPath = strDataPath / "poses";
    const std::filesystem::path poseCam1File = posesPath / "poses_cam1.txt";
    const std::filesystem::path poseCam2File = posesPath / "poses_cam2.txt";

    // Helper function to get numeric part of filename for sorting
    auto extract_number = [](const std::string& filename) -> long {
        size_t start = filename.find_last_of('_');
        size_t end = filename.find_last_of('.');
        if (start == std::string::npos || end == std::string::npos) return 0;
        std::string num_str = filename.substr(start + 1, end - start - 1);
        try {
            return std::stol(num_str);
        } catch (...) {
            return 0;
        }
    };

    // Load cam1 RGB images (frame_xxxx.png)
    for (const auto& entry : std::filesystem::directory_iterator(cam1RGBPath)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            std::string filename = entry.path().filename().string();
            if (filename.find("frame_") == 0) {
                vstrImageFilenamesRGB.push_back(entry.path().string());
            }
        }
    }
    // Sort by frame number
    std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end(),
        [&](const std::string& a, const std::string& b) {
            return extract_number(a) < extract_number(b);
        });

    // Load cam2 RGB images (frame_xxxx.png)
    for (const auto& entry : std::filesystem::directory_iterator(cam2RGBPath)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            std::string filename = entry.path().filename().string();
            if (filename.find("frame_") == 0) {
                vstrImageFilenamesHR.push_back(entry.path().string());
            }
        }
    }
    // Sort by frame number
    std::sort(vstrImageFilenamesHR.begin(), vstrImageFilenamesHR.end(),
        [&](const std::string& a, const std::string& b) {
            return extract_number(a) < extract_number(b);
        });

    // Load cam1 depth images (depth_xxxxxxxx.png)
    for (const auto& entry : std::filesystem::directory_iterator(cam1DepthPath)) {
        if (entry.path().extension() == ".png") {
            std::string filename = entry.path().filename().string();
            if (filename.find("depth_") == 0) {
                vstrImageFilenamesD.push_back(entry.path().string());
            }
        }
    }
    // Sort by embedded frame number (first 4 digits of 8-digit code)
    std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end(),
        [&](const std::string& a, const std::string& b) {
            // long num_a = extract_number(a) / 10000;  // Extract first 4 digits
            // long num_b = extract_number(b) / 10000;
            // return num_a < num_b;
            return extract_number(a) < extract_number(b);
        });

    // Load cam1 timestamps
    std::ifstream f_cam1(poseCam1File);
    if (f_cam1.is_open()) {
        std::string line;
        while (std::getline(f_cam1, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double timestamp, tx, ty, tz, qx, qy, qz, qw;

            iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            std::vector<double> vLRGTPose = {tx, ty, tz, qx, qy, qz, qw};

            vTimestampsRGBD.push_back(timestamp);
            vvLRGTPose.push_back(vLRGTPose);
        }
    }
    else if(lr_fps>0){
        for(size_t i=0; i<vstrImageFilenamesRGB.size(); i++)
            vTimestampsRGBD.push_back(double(i) / lr_fps); 

        std::cout << "Could not open: " << poseCam1File.string() << ". Generating timestamps based on provided fps: "<<lr_fps<<std::endl;
    }
    else throw std::runtime_error("Could not open: " + poseCam1File.string() + ". Please provide valid poses file or fps to generate timestamps.");

    // Load cam2 timestamps
    std::ifstream f_cam2(poseCam2File);
    if (f_cam2.is_open()) {
        std::string line;
        while (std::getline(f_cam2, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double timestamp, tx, ty, tz, qx, qy, qz, qw;

            iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            std::vector<double> vHRGTPose = {tx, ty, tz, qx, qy, qz, qw};

            vTimestampsHR.push_back(timestamp);
            vvHRGTPose.push_back(vHRGTPose);
        }
    }
    else {
        std::cout << "Could not open: " << poseCam2File.string() << ". Proceeding without loading timestamps and poses for cam2." << std::endl;
    }

    std::cout<<"LD RGB: "<<vstrImageFilenamesRGB.size()<<std::endl;
    std::cout<<"LD Depth: "<<vstrImageFilenamesD.size()<<std::endl;
    std::cout<<"HR RGB: "<<vstrImageFilenamesHR.size()<<std::endl;
    std::cout<<"LD time: "<<vTimestampsRGBD.size()<<std::endl;
    std::cout<<"HR time: "<<vTimestampsHR.size()<<std::endl;
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }

    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}

void LoadPathConfig(std::filesystem::path cfg_path, 
    std::filesystem::path& strORBSLAMVocabularyPath, std::filesystem::path& strORBSLAMParametersPath, std::filesystem::path& strMapperPath, 
    std::filesystem::path& strDataPath, std::filesystem::path& strSavingPath, double& lr_fps, bool& bViewer){

    cv::FileStorage cfg(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!cfg.isOpened()) {
       std::cerr << "[Error]Failed to open config file at: " << cfg_path << std::endl;
       exit(-1);
    }
    else std::cout << "Reading parameters from " << cfg_path << std::endl;

    strORBSLAMVocabularyPath = cfg["ORB-SLAM3.Vocabulary"];
    strORBSLAMParametersPath = cfg["ORB-SLAM3.Parameters"];
    strMapperPath = cfg["Mapper.Parameters"];
    strDataPath = cfg["Dataset"];
    strSavingPath = cfg["Results.Saving"];
    lr_fps = cfg["Dataset.fps"].operator double();
    bViewer = (cfg["Results.Viewer"].operator int()) != 0;

    strSavingPath = strSavingPath / "";
    std::filesystem::create_directories(strSavingPath);

    bool allExist = std::filesystem::exists(strORBSLAMVocabularyPath) && std::filesystem::exists(strORBSLAMParametersPath) && 
                std::filesystem::exists(strMapperPath) && std::filesystem::exists(strDataPath);

    if (!allExist) {
        std::cerr << "[Error]Parameters error in config file at: " << cfg_path << std::endl;
        exit(-1);
    }
}

