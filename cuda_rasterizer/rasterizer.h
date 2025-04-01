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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* fullprojmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* fullprojmatrix,
			const float* cam_pos,
			const float* deblur_nu,				// cam_lin_vel_
			const float* deblur_omega,			// cam_ang_vel_
			const float deblur_rs_time,			// rolling_shutter_time_
			const float deblur_mb_time,			// exposure_time_
			const unsigned deblur_samples,		// n_blur_samples_
			const float tan_fovx, float tan_fovy,
			const float cx, const float cy,
			const bool prefiltered,
			const bool enable_optim_pose,
			const bool enable_optim_velocity,
			float* out_color,
			float* out_depth,
			float* out_opacity,
			int* radii = nullptr,
			int* n_touched = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* fullprojmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			const float* deblur_nu,				// cam_lin_vel_
			const float* deblur_omega,			// cam_ang_vel_
			const float deblur_rs_time,			// rolling_shutter_time_
			const float deblur_mb_time,			// exposure_time_
			const unsigned deblur_samples,		// n_blur_samples_
			const bool enable_optim_pose,
			const bool enable_optim_velocity,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,		// dL_dout_color
			const float* dL_dpix_depth,	// dL_dout_depth
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepths,
			float* dL_dpixvels,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtau,
			float* dL_dnu,
			float* dL_domega,
			bool debug);
	};
};

#endif