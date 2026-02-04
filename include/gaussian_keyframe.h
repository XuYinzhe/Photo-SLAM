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

#pragma once

#include <memory>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "ORB-SLAM3/include/System.h"

#include "types.h"
#include "camera.h"
#include "point2d.h"
#include "general_utils.h"
#include "graphics_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters.h"

class GaussianKeyframe
{
public:
    GaussianKeyframe() {}

    GaussianKeyframe(std::size_t fid, int creation_iter = 0)
        : fid_(fid), creation_iter_(creation_iter) {}

    GaussianKeyframe(std::size_t fid, int creation_iter, KeyframeOptimizationParams* kf_params)
        : fid_(fid), creation_iter_(creation_iter), kf_params_(kf_params) {
        
        // trainable parameters initialization
        this->global_delta_pose_ = torch::eye(4, torch::dtype(torch::kFloat32).requires_grad(false).device(torch::kCUDA));
        this->local_delta_pose_ = torch::eye(4, torch::dtype(torch::kFloat32).requires_grad(false).device(torch::kCUDA));
        this->local_delta_pose_init_ = torch::eye(4, torch::dtype(torch::kFloat32).requires_grad(false).device(torch::kCUDA));
        this->base_pose_ = torch::eye(4, torch::dtype(torch::kFloat32).requires_grad(false).device(torch::kCUDA));

        this->theta_ = torch::zeros({3}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));
        this->rho_ = torch::zeros({3}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));

        this->exposure_a_ = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));
        this->exposure_b_ = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));

        // pose optimizer
        std::vector<torch::optim::OptimizerParamGroup> pose_param_groups;
        auto options_rot = torch::optim::AdamOptions(kf_params_->theta_lr_).betas({0.9, 0.999});
        auto options_trans = torch::optim::AdamOptions(kf_params_->rho_lr_).betas({0.9, 0.999});
        
        pose_param_groups.push_back( torch::optim::OptimizerParamGroup(
            {this->theta_}, std::make_unique<torch::optim::AdamOptions>(options_rot)
        ));
        
        pose_param_groups.push_back( torch::optim::OptimizerParamGroup(
            {this->rho_}, std::make_unique<torch::optim::AdamOptions>(options_trans)
        ));

        this->pose_optimizer_ = std::make_unique<torch::optim::Adam>(pose_param_groups);

        // exposure optimizer
        std::vector<torch::optim::OptimizerParamGroup> exposure_param_groups;
        auto options_exposure_a = torch::optim::AdamOptions(kf_params_->exposure_a_lr_).betas({0.9, 0.999});
        auto options_exposure_b = torch::optim::AdamOptions(kf_params_->exposure_b_lr_).betas({0.9, 0.999});

        exposure_param_groups.push_back( torch::optim::OptimizerParamGroup(
            {this->exposure_a_}, std::make_unique<torch::optim::AdamOptions>(options_exposure_a)
        ));
        exposure_param_groups.push_back( torch::optim::OptimizerParamGroup(
            {this->exposure_b_}, std::make_unique<torch::optim::AdamOptions>(options_exposure_b)
        ));

        this->exposure_optimizer_ = std::make_unique<torch::optim::Adam>(exposure_param_groups);
    }

    void stepOptimizer(bool step_pose = false, bool step_exposure = false);
    void zeroOptimizerGrad(bool zero_pose = false, bool zero_exposure = false);
    void resetOptimizer(bool with_lr = false, float theta_lr = -1.f, float rho_lr = -1.f);
    void updateOptimizer(float theta_lr, float rho_lr);
    void updateOptimizer(float decay_ratio);
    void resetFullExposure();

    void setPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz);
    
    void setPose(
        const Eigen::Quaterniond& q,
        const Eigen::Vector3d& t);

    void setGTPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz);

    Sophus::SE3d getPose(bool with_global = false, bool with_local = false);
    Sophus::SE3f getPosef(bool with_global = false, bool with_local = false);

    Sophus::SE3f getGTPosef();

    torch::Tensor getGTLRImg(bool use_cuda = false);
    torch::Tensor getGTLRDpt(bool use_cuda = false);
    torch::Tensor getGTLRDptMsk(bool use_cuda = false);

    void setGTLRDpt(cv::Mat& depth_img);
    void setGTLRImg(cv::Mat& color_img);
    bool setGTHRImg(float align_time, std::vector<std::string>& hr_img_filenames);
    torch::Tensor getGTHRImg(float resize_ratio = 1.0f, bool use_cuda = false);
    torch::Tensor getGTHRImg(torch::Tensor& selection_indices, float resize_ratio = 1.0f, bool use_cuda = false);

    void setCameraParams(const Camera& camera);

    void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);
    void setPoint3DIdxForPoint2D(
        const point2D_idx_t point2D_idx,
        const point3D_id_t point3D_id);

    void computeTransformTensors();

    Eigen::Matrix4f getWorld2View2(
        const Eigen::Vector3f& trans = {0.0f, 0.0f, 0.0f},
        float scale = 1.0f);

    torch::Tensor getProjectionMatrix(
        float znear,
        float zfar,
        float fovX,
        float fovY,
        torch::DeviceType device_type = torch::kCUDA);

    int getCurrentGausPyramidLevel();

    // void undistortHR(cv::Mat& img, torch::DeviceType device_type);

    void updateRenderMatrix(bool is_hr = false);

    bool updateBasePose(
        bool has_thresh = false, 
        float theta_thresh = -1.f,
        float rho_thresh = -1.f);
    bool updateBasePose(
        torch::Tensor& delta_pose,
        bool has_thresh = false, 
        float theta_thresh = -1.f,
        float rho_thresh = -1.f);
    bool updateGlobalDeltaPose(
        bool has_thresh = false, 
        float theta_thresh = -1.f,
        float rho_thresh = -1.f);
    bool updateLocalDeltaPose(
        bool has_thresh = false, 
        float theta_thresh = -1.f,
        float rho_thresh = -1.f);
    bool updateLocalDeltaPose(
        torch::Tensor& delta_pose,
        bool has_thresh = false, 
        float theta_thresh = -1.f,
        float rho_thresh = -1.f);
    bool validLieAlgebraUpdate(
        bool has_thresh = false, 
        float theta_thresh = -1.f,
        float rho_thresh = -1.f,
        bool normalize = true);

    torch::Tensor getBasePose();
    torch::Tensor getGlobalDeltaPose();
    torch::Tensor getLocalDeltaPose();
    torch::Tensor getFullDeltaPose();

    void setBasePose(torch::Tensor pose);
    void setGlobalDeltaPose(torch::Tensor pose);
    void setLocalDeltaPose(torch::Tensor pose);
    void setLocalDeltaPoseInit(torch::Tensor pose);

    void updateDenseInitJointGaussians(torch::Tensor& xyz, torch::Tensor indices);
    void setDenseInitJointOptimization(bool flag);
    void updateDenseInitJointState(torch::Tensor delta_pose);
    int matchInitHROrbGMS();
    int matchHROrbGMS(
        torch::Tensor render_hr_tensor, float hr_resize_ratio,
        std::vector<cv::KeyPoint>& orb_keypoints_render_hr,
        std::vector<cv::KeyPoint>& orb_keypoints_gt_hr,
        std::vector<int>& vnMatches_render2gt,
        std::vector<int>& vnMatches_gt2render
    );
    int matchLROrbGMS(
        torch::Tensor render_lr_tensor, 
        std::vector<cv::KeyPoint>& orb_keypoints_render_lr,
        std::vector<cv::KeyPoint>& orb_keypoints_gt_lr,
        std::vector<int>& vnMatches_render2gt,
        std::vector<int>& vnMatches_gt2render
    );

public:
    std::size_t fid_;
    int creation_iter_;
    int remaining_times_of_use_ = 0;

    bool set_camera_ = false;

    camera_id_t camera_id_;
    int camera_model_id_ = 0;

    std::string img_filename_;
    cv::Mat img_undist_, img_auxiliary_undist_, depth_undist_valid_mask_;
    cv::Mat hr_undist_mat_;
    torch::Tensor original_image_; ///< image
    torch::Tensor original_depth_;
    torch::Tensor original_depth_mask_;
    int image_width_;              ///< image
    int image_height_;             ///< image

    int dense_depth_num_;

    torch::Tensor hr_image_;
    torch::Tensor hr_image_undist_;
    std::map<std::tuple<int, int, int>, torch::Tensor> hr_selection_maps_;

    int num_gaus_pyramid_sub_levels_;
    std::vector<int> gaus_pyramid_times_of_use_;
    std::vector<std::size_t> gaus_pyramid_width_;            ///< gaus_pyramid image
    std::vector<std::size_t> gaus_pyramid_height_;           ///< gaus_pyramid image
    std::vector<torch::Tensor> gaus_pyramid_original_image_; ///< gaus_pyramid image
    // Tensor gt_alpha_mask_;

    std::vector<float> intr_; ///< intrinsics

    float FoVx_; ///< intrinsics
    float FoVy_; ///< intrinsics

    bool set_gt_pose_ = false;
    bool set_pose_ = false;
    bool set_projection_matrix_ = false;

    Eigen::Quaterniond R_quaternion_;  ///< extrinsics
    Eigen::Vector3d t_;                ///< extrinsics
    Sophus::SE3d Tcw_;                 ///< extrinsics
    Sophus::SE3d GT_Tcw_;

    torch::Tensor R_tensor_; ///< extrinsics
    torch::Tensor t_tensor_; ///< extrinsics

    float zfar_ = 100.0f;
    float znear_ = 0.01f;

    Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};
    float scale_ = 1.0f;

    torch::Tensor world_view_transform_;    ///< transform tensors
    torch::Tensor projection_matrix_;       ///< transform tensors
    torch::Tensor full_proj_transform_;     ///< transform tensors
    torch::Tensor camera_center_;           ///< transform tensors

    torch::Tensor base_pose_;
    torch::Tensor base_lr_proj_;
    torch::Tensor base_hr_proj_;

    // torch::Tensor theta_ = torch::zeros({3}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));
    // torch::Tensor rho_ = torch::zeros({3}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));

    KeyframeOptimizationParams* kf_params_;

    std::unique_ptr<torch::optim::Adam> pose_optimizer_;
    std::unique_ptr<torch::optim::Adam> exposure_optimizer_;
    torch::Tensor theta_; // rotation update
    torch::Tensor rho_; // translation update
    torch::Tensor global_delta_pose_;
    torch::Tensor local_delta_pose_;
    torch::Tensor local_delta_pose_init_;
    torch::Tensor exposure_a_, exposure_b_;
    bool global_finish_ = false;

    // ORB_SLAM3::Frame orb_frame_lr_;
    // ORB_SLAM3::Frame orb_frame_hr_;
    cv::Mat orb_descriptors_lr_;
    std::vector<cv::KeyPoint> orb_keypoints_lr_;
    std::vector<int> orb_matches_lr2hr_;
    std::vector<int> orb_matches_hr2lr_;

    std::map<std::tuple<int, int, float>, std::tuple<cv::Mat, std::vector<cv::KeyPoint>>> orb_multisizes_hr_;
    // float orb_hr_ratio_;
    // int num_orb_matches_ = 0;

    float lr_timestamp_;
    float hr_timestamp_;
    int hr_fid_;
    bool has_hr_fid_ = false;

    std::vector<Point2D> points2D_;
    std::vector<float> kps_pixel_;
    std::vector<float> kps_point_local_;

    bool need_dense_init_optimization_ = false;
    torch::Tensor dense_init_xyz_;
    torch::Tensor dense_init_rgb_;
    torch::Tensor dense_init_opcity_;
    torch::Tensor dense_init_scale_;
    torch::Tensor dense_init_rotation_;
    torch::Tensor dense_init_feat_dc_;
    torch::Tensor dense_init_feat_rest_;
    torch::Tensor dense_init_transformations_;

    bool done_inactive_geo_densify_ = false;
};
