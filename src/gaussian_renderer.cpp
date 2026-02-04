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

#include "include/gaussian_renderer.h"

/**
 * @brief 
 * 
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which are all `torch::Tensor`
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    bool has_hr,
    bool has_fix_mean3d,
    bool has_fix_gs_geo,
    bool has_fix_gs_clr,
    float scaling_modifier,
    bool use_override_color)
{
    /* Render the scene. 

       Background tensor (bg_color) must be on GPU!
     */

    // Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    auto screenspace_points = torch::zeros_like(pc->getXYZ(viewpoint_camera),
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; // pass
    }

    // Set up rasterization configuration
    float tanfovx, tanfovy;
    if(viewpoint_camera->kf_params_->align_pose_ && viewpoint_camera->kf_params_->render_aligned_ && has_hr){
        tanfovx = std::tan(viewpoint_camera->kf_params_->hr_fovx_ * 0.5f);
        tanfovy = std::tan(viewpoint_camera->kf_params_->hr_fovy_ * 0.5f);
    }
    else
    {
        tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
        tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);
    }

    viewpoint_camera->updateRenderMatrix(has_hr);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        viewpoint_camera->projection_matrix_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false, false
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ(viewpoint_camera);
    if(has_fix_mean3d) means3D = means3D.detach(); 
    auto means2D = screenspace_points;
    if(has_fix_mean3d) means2D = means2D.detach();
    auto opacities = pc->getOpacityActivation(viewpoint_camera);
    if(has_fix_gs_geo) opacities = opacities.detach();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be computed from
       scaling / rotation by the rasterizer. 
     */
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->getCovarianceActivation();
        has_cov3D_precomp = true;
    }
    else { // main
        scales = pc->getScalingActivation(viewpoint_camera);
        if(has_fix_gs_geo) scales = scales.detach();
        rotations = pc->getRotationActivation(viewpoint_camera);
        if(has_fix_gs_geo) rotations = rotations.detach();
        has_scales = true;
        has_rotations = true;
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
       from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
     */
    bool has_shs = false,
         has_color_precomp = false;
    torch::Tensor shs,
                  colors_precomp;
    if (use_override_color) {
        colors_precomp = override_color;
        has_color_precomp = true;
    }
    else {
        if (pipe.convert_SHs_) {
            int max_sh_degree = pc->max_sh_degree_ + 1;
            torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view({-1, 3, max_sh_degree * max_sh_degree});
            torch::Tensor dir_pp = (pc->getXYZ() - viewpoint_camera->camera_center_.repeat({pc->getFeatures().size(0), 1}));
            auto dir_pp_normalized = dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
            auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_pp_normalized);
            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
            has_color_precomp = true;
        }
        else { // main
            shs = pc->getFeatures(viewpoint_camera);
            if(has_fix_gs_clr) shs = shs.detach();
            has_shs = true;
        }
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen). 
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacities,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp,
        viewpoint_camera->theta_,
        viewpoint_camera->rho_
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);
    auto depth = std::get<2>(rasterizer_result);
    auto opacity = std::get<3>(rasterizer_result);
    auto n_touched = std::get<4>(rasterizer_result);

    /* Those Gaussians that were frustum culled or had a radius of 0 were not visible.
       They will be excluded from value updates used in the splitting criteria.
     */
    return std::make_tuple(
        rendered_image,     /*render*/
        screenspace_points, /*viewspace_points*/
        radii > 0,          /*visibility_filter*/
        radii,               /*radii*/
        depth,
        opacity,
        n_touched
    );
}