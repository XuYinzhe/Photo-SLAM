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

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "types.h"
#include "point3d.h"
#include "operate_points.h"
#include "general_utils.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters.h"
#include "gaussian_keyframe.h"

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->xyz_gradient_accum_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
    this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type));              \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel
{
public:
    GaussianModel(const int sh_degree);
    GaussianModel(const GaussianModelParams& model_params);
    GaussianModel(const GaussianModelParams& model_params, bool debug_mode);

    torch::Tensor getScalingActivation();
    torch::Tensor getRotationActivation();
    torch::Tensor getXYZ();
    torch::Tensor getFeatures();
    torch::Tensor getOpacityActivation();

    torch::Tensor getScalingActivation(std::shared_ptr<GaussianKeyframe> pkf);
    torch::Tensor getRotationActivation(std::shared_ptr<GaussianKeyframe> pkf);
    torch::Tensor getXYZ(std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);
    torch::Tensor getFeatures(std::shared_ptr<GaussianKeyframe> pkf);
    torch::Tensor getOpacityActivation(std::shared_ptr<GaussianKeyframe> pkf);

    torch::Tensor getFeaturesDC(std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);
    torch::Tensor getFeaturesRest(std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);
    torch::Tensor getOpacity(std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);
    torch::Tensor getScaling(std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);
    torch::Tensor getRotation(std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);

    torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

    void oneUpShDegree();
    void setShDegree(const int sh);

    void createFromPcd(
        std::map<point3D_id_t, Point3D> pcd,
        const float spatial_lr_scale);
    void createFromPcd(
        const torch::Tensor& xyz,
        const torch::Tensor& rgb,
        const torch::Tensor& idx,
        const float spatial_lr_scale);

    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration);

    void increaseKeyframeInitPcd(std::shared_ptr<GaussianKeyframe> pkf, 
        torch::Tensor& valid_depth, 
        torch::Tensor& full_rgb,
        KeyframeOptimizationParams& kf_params);
    void increaseKeyframeFinalPcd(std::shared_ptr<GaussianKeyframe> pkf);
    void rebindGaussianParametersToKeyframe(std::shared_ptr<GaussianKeyframe> pkf, bool use_cpu = false);
    int updateBatchGradients(const std::vector<std::size_t>& batch_frame_ids, 
        float masked_scale = 1.0f, 
        float unmasked_scale = 1.0f, 
        bool fixed_unmasked_xyz = true,
        bool fixed_unmasked_geo = true,
        bool fixed_unmasked_clr = true);
    void updateKeyframeJointPcd(std::shared_ptr<GaussianKeyframe> pkf, 
        torch::Tensor& delta_pose,
        torch::Tensor& joint_mask);

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void scaledTransformationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_scaling);

    void scaledTransformVisiblePointsOfKeyframe(
        torch::Tensor& point_not_transformed_flags,
        torch::Tensor& diff_pose,
        torch::Tensor& kf_world_view_transform,
        torch::Tensor& kf_full_proj_transform,
        const int kf_creation_iter,
        const int stable_num_iter_existence,
        int& num_transformed,
        const float scale = 1.0f);

    void trainingSetup(const GaussianOptimizationParams& training_args);
    void resetOptimizer(const GaussianOptimizationParams& training_args, float scale_rate = 1.0f);
    float updateLearningRate(int step);
    void setPositionLearningRate(float position_lr);
    void setFeatureLearningRate(float feature_lr);
    void setOpacityLearningRate(float opacity_lr);
    void setScalingLearningRate(float scaling_lr);
    void setRotationLearningRate(float rot_lr);

    void resetOpacity();
    torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

    void prunePoints(torch::Tensor& mask);

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_exist_since_iter);

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_frame_ids,
        torch::Tensor& new_exist_since_iter);

    void densifyAndSplit(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent,
        int N = 2);

    void densifyAndClone(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent);

    void densifyAndPrune(
        float max_grad,
        float min_opacity,
        float extent,
        int max_screen_size);

    void addDensificationStats(
        torch::Tensor& viewspace_point_tensor,
        torch::Tensor& update_filter);

// void increasePointsIterationsOfExistence(const int i = 1);

    void loadPly(std::filesystem::path ply_path);
    void savePly(std::filesystem::path result_path);
    void savePly(std::filesystem::path result_path, std::shared_ptr<GaussianKeyframe> pkf, bool kf_only = false);
    void saveSparsePointsPly(std::filesystem::path result_path);

    float percentDense();
    void setPercentDense(const float percent_dense);

protected:
    float exponLrFunc(int step);

public:
    torch::DeviceType device_type_;

    int active_sh_degree_;
    int max_sh_degree_;

    torch::Tensor xyz_;
    torch::Tensor features_dc_;
    torch::Tensor features_rest_;
    torch::Tensor scaling_;
    torch::Tensor rotation_;
    torch::Tensor opacity_;
    torch::Tensor max_radii2D_;
    torch::Tensor xyz_gradient_accum_;
    torch::Tensor denom_;
    torch::Tensor exist_since_iter_;
    torch::Tensor frame_ids_;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    float percent_dense_;
    float spatial_lr_scale_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

    // std::unordered_map<std::size_t, std::pair<torch::Tensor, torch::Tensor>> dense_init_fids_valid_pixels_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_xyz_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_rgb_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_opacity_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_rotation_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_scaling_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_feat_dc_;
    // std::unordered_map<std::size_t, torch::Tensor> dense_init_fids_feat_rest_;
    std::vector<std::size_t> dense_init_fids_;

protected:
    bool debug_mode_;

    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    std::mutex mutex_settings_;
};
