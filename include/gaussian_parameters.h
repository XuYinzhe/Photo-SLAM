/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <string>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "ORB-SLAM3/include/System.h"

class GaussianModelParams
{
public:
    GaussianModelParams(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false);

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;
};

class GaussianPipelineParams
{
public:
    GaussianPipelineParams(
        bool convert_SHs = false,
        bool compute_cov3D = false);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
};

class GaussianOptimizationParams
{
public:
    GaussianOptimizationParams(
        int iterations = 30'000,
        float position_lr_init = 0.00016f,
        float position_lr_final = 0.0000016f,
        float position_lr_delay_mult = 0.01f,
        int position_lr_max_steps = 30'000,
        float feature_lr = 0.0025f,
        float opacity_lr = 0.05f,
        float scaling_lr = 0.005f,
        float rotation_lr = 0.001f,
        float percent_dense = 0.01f,
        float lambda_dssim = 0.2f,
        int densification_interval = 100,
        int opacity_reset_interval = 3000,
        int densify_from_iter = 500,
        int densify_until_iter = 15'000,
        float densify_grad_threshold = 0.0002f);

public:
    int iterations_;
    float position_lr_init_;
    float position_lr_final_;
    float position_lr_delay_mult_;
    int position_lr_max_steps_;
    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;
    float percent_dense_;
    float lambda_dssim_;
    int densification_interval_;
    int opacity_reset_interval_;
    int densify_from_iter_;
    int densify_until_iter_;
    float densify_grad_threshold_;
};

class KeyframeOptimizationParams
{
public:
    KeyframeOptimizationParams(
        bool align_pose = true,
        bool render_aligned = true,
        bool align_global_pose = true,
        bool align_local_pose = true,
        float theta_lr = 0.005,
        float rho_lr = 0.002,
        int hr_width = 960,
        int hr_height = 480,
        float hr_fps = 30.f,
        float hr_fx = 960.f,
        float hr_fy = 960.f,
        float hr_cx = 480.f,
        float hr_cy = 320.f,
        float hr_k1 = 0.f,
        float hr_k2 = 0.f,
        float hr_p1 = 0.f,
        float hr_p2 = 0.f,
        float hr_k3 = 0.f
    );

public:
    bool debug_ = false;
    std::string debug_dir_ = "_debug";

    bool align_pose_;
    bool render_aligned_;
    bool align_global_pose_;
    bool align_local_pose_;
    bool align_exposure_;

    float exposure_a_lr_, exposure_b_lr_;

    float theta_lr_;
    float rho_lr_;

    float hr_color_theta_lr_;
    float hr_color_rho_lr_;

    int hr_width_, hr_height_;
    float hr_fps_;

    float hr_fx_, hr_fy_, hr_cx_, hr_cy_;
    float hr_k1_, hr_k2_, hr_p1_, hr_p2_, hr_k3_;

    int hr_pixel_samples_;

    int lr_width_, lr_height_;
    float lr_fx_, lr_fy_, lr_cx_, lr_cy_;
    float lr_fps_;

    float hr_fovx_;
    float hr_fovy_;

    cv::Mat lr_undistort_mask_;
    torch::Tensor lr_undistort_mask_tensor_;
    int lr_depth_variance_window_size_ = 3;
    float lr_depth_variance_threshold_ratio_ = 1.f;
    float lr_local_init_opacity_ = 0.9f;

    cv::Mat hr_undistort_mask_;
    cv::Mat hr_undistort_map1_, hr_undistort_map2_;
    bool has_undistort_ = false;

    std::vector<std::size_t> gaus_pyramid_hr_width_;
    std::vector<std::size_t> gaus_pyramid_hr_height_;
    std::vector<torch::Tensor> gaus_pyramid_hr_undistort_mask_;

    std::string orb_vocab_path_;
    ORB_SLAM3::ORBVocabulary* orb_vocabulary_ = nullptr;
    ORB_SLAM3::ORBextractor* orb_extractor_lr_ = nullptr;
    ORB_SLAM3::ORBextractor* orb_extractor_hr_ = nullptr;
    ORB_SLAM3::GeometricCamera* orb_camera_lr_ = nullptr;
    ORB_SLAM3::GeometricCamera* orb_camera_hr_ = nullptr;
    cv::Mat orb_camera_dist_ = cv::Mat::zeros(4,1,CV_32F);
    float orb_thdepth_ = 16.f;
    float orb_bf_ = 1.f;

    std::filesystem::path result_dir_;
};