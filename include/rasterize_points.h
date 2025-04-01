/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/torch.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, 	// rendered
	torch::Tensor, 	// out_color
	torch::Tensor, 	// radii
	torch::Tensor, 	// out_depth
	torch::Tensor, 	// out_opaticy
	torch::Tensor, 	// n_touched
	torch::Tensor, 	// geomBuffer
	torch::Tensor, 	// binningBuffer
	torch::Tensor>	// imgBuffer
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& fullprojmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& deblur_nu,		// cam_lin_vel_
	const torch::Tensor& deblur_omega,	// cam_ang_vel_
	const float deblur_rs_time,			// rolling_shutter_time_
	const float deblur_mb_time,			// exposure_time_
	const unsigned deblur_samples,		// n_blur_samples_
	const float tan_fovx, 
	const float tan_fovy,
	const float cx, const float cy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool enable_optim_pose,
	const bool enable_optim_velocity,
	const bool prefiltered,
	const bool debug);

std::tuple<
	torch::Tensor, 	// dL_dmeans2D
	torch::Tensor, 	// dL_dcolors
	torch::Tensor,	// dL_dopacity
	torch::Tensor, 	// dL_dmeans3D
	torch::Tensor, 	// dL_dcov3D
	torch::Tensor, 	// dL_dsh
	torch::Tensor, 	// dL_dscales
	torch::Tensor, 	// dL_drotations
	torch::Tensor, 	// dL_dtau
	torch::Tensor, 	// dL_dnu
	torch::Tensor>	// dL_domega
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& fullprojmatrix,
    const torch::Tensor& projmatrix,
	const torch::Tensor& deblur_nu,		// cam_lin_vel_
	const torch::Tensor& deblur_omega,	// cam_ang_vel_
	const float deblur_rs_time,			// rolling_shutter_time_
	const float deblur_mb_time,			// exposure_time_
	const unsigned deblur_samples,		// n_blur_samples_
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depths,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool enable_optim_pose,
	const bool enable_optim_velocity,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& fullprojmatrix);