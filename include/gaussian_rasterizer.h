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

#include <tuple>

#include <torch/torch.h>

#include "rasterize_points.h"
#include "gaussian_model.h"

struct GaussianRasterizationSettings
{
    GaussianRasterizationSettings(
        int image_height,
        int image_width,
        float tanfovx,
        float tanfovy,
        std::vector<float> intr,
        torch::Tensor& bg,
        float scale_modifier,
        torch::Tensor& viewmatrix,
        torch::Tensor& full_projmatrix,
        torch::Tensor& projmatrix,
        int sh_degree,
        torch::Tensor& campos,
        float rolling_shutter_time,
        float exposure_time,
        int n_blur_samples,
        bool enable_optim_pose,
	    bool enable_optim_velocity,
        bool prefiltered,
        bool debug)
        : image_height_(image_height), image_width_(image_width), 
          tanfovx_(tanfovx), tanfovy_(tanfovy),
          fx_(intr[0]), fy_(intr[1]), cx_(intr[2]), cy_(intr[3]), 
          bg_(bg), scale_modifier_(scale_modifier), 
          viewmatrix_(viewmatrix), full_projmatrix_(full_projmatrix), projmatrix_(projmatrix),
          sh_degree_(sh_degree), campos_(campos), 
          rolling_shutter_time_(rolling_shutter_time),
          exposure_time_(exposure_time),
          n_blur_samples_(n_blur_samples),
          enable_optim_pose_(enable_optim_pose), enable_optim_velocity_(enable_optim_velocity),
          prefiltered_(prefiltered), debug_(debug)
    {}

    int image_height_;
    int image_width_;
    float tanfovx_;
    float tanfovy_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    torch::Tensor bg_;
    float scale_modifier_;
    torch::Tensor viewmatrix_;
    torch::Tensor full_projmatrix_;
    torch::Tensor projmatrix_;
    int sh_degree_;
    torch::Tensor campos_;

    float rolling_shutter_time_;
    float exposure_time_;
    int n_blur_samples_;

    bool enable_optim_pose_;
	bool enable_optim_velocity_;
    bool prefiltered_;
    bool debug_;
};

class GaussianRasterizerFunction : public torch::autograd::Function<GaussianRasterizerFunction>
{
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3Ds_precomp,
        torch::Tensor theta,    // cam_rot_delta_
        torch::Tensor rho,      // cam_trans_delta_
        torch::Tensor nu,       // cam_lin_vel_delta_
        torch::Tensor omega,    // cam_ang_vel_delta_
        GaussianRasterizationSettings raster_settings);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_out_color);
};

inline torch::autograd::tensor_list rasterizeGaussians(
    torch::Tensor& means3D,
    torch::Tensor& means2D,
    torch::Tensor& sh,
    torch::Tensor& colors_precomp,
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& cov3Ds_precomp,
    torch::Tensor& theta,    // cam_rot_delta_
    torch::Tensor& rho,      // cam_trans_delta_
    torch::Tensor& nu,       // cam_lin_vel_delta_
    torch::Tensor& omega,    // cam_ang_vel_delta_
    GaussianRasterizationSettings& raster_settings)
{
    return GaussianRasterizerFunction::apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,    // cam_rot_delta_
        rho,      // cam_trans_delta_
        nu,       // cam_lin_vel_delta_
        omega,    // cam_ang_vel_delta_
        raster_settings
    );
}

class GaussianRasterizer : public torch::nn::Module
{
public:
    GaussianRasterizer(GaussianRasterizationSettings& raster_settings)
        : raster_settings_(raster_settings)
    {}

    torch::Tensor markVisibleGaussians(torch::Tensor& positions);

    std::tuple<
        torch::Tensor,  // color
        torch::Tensor,  // radii
        torch::Tensor,  // depth
        torch::Tensor,  // opaticy
        torch::Tensor>  // n_touched
    forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        bool has_shs,
        bool has_colors_precomp,
        bool has_scales,
        bool has_rotations,
        bool has_cov3D_precomp,
        torch::Tensor shs,
        torch::Tensor colors_precomp,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3D_precomp,
        torch::Tensor theta,    // cam_rot_delta_
        torch::Tensor rho,      // cam_trans_delta_
        torch::Tensor nu,       // cam_lin_vel_delta_
        torch::Tensor omega     // cam_ang_vel_delta_
        );

public:
    GaussianRasterizationSettings raster_settings_;
};
