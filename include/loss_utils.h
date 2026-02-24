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

#include <vector>

#include <torch/torch.h>

#include "tensor_utils.h"

namespace loss_utils
{

inline torch::Tensor l1_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::abs(network_output - gt).mean();
}

inline torch::Tensor l2_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::pow(network_output - gt, 2).mean();
}

inline torch::Tensor psnr(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).mean();
    return 10.0f * torch::log10(1.0f / mse);
}

/** def psnr(img1, img2):
 *     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
 *     return 20 * torch.log10(1.0 / torch.sqrt(mse))
 */
inline torch::Tensor psnr_gaussian_splatting(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).view({img1.size(0) , -1}).mean(1, /*keepdim=*/true);
    return 20.0f * torch::log10(1.0f / torch::sqrt(mse)).mean();
}

inline torch::Tensor gaussian(
    int window_size,
    float sigma,
    torch::DeviceType device_type = torch::kCUDA)
{
    std::vector<float> gauss_values(window_size);
    for (int x = 0; x < window_size; ++x) {
        int temp = x - window_size / 2;
        gauss_values[x] = std::exp(-temp * temp / (2.0f * sigma * sigma));
    }
    torch::Tensor gauss = torch::tensor(
        gauss_values,
        torch::TensorOptions().device(device_type));
    return gauss / gauss.sum();
}

inline torch::autograd::Variable create_window(
    int window_size,
    int64_t channel,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto _1D_window = gaussian(window_size, 1.5f, device_type).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).to(torch::kFloat).unsqueeze(0).unsqueeze(0);
    auto window = torch::autograd::Variable(_2D_window.expand({channel, 1, window_size, window_size}).contiguous());
    return window;
}

inline torch::Tensor _ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::autograd::Variable &window,
    int window_size,
    int64_t channel,
    bool size_average = true)
{
    int window_size_half = window_size / 2;
    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));
    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_sq;
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu2_sq;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_mu2;

    auto C1 = 0.01 * 0.01;
    auto C2 = 0.03 * 0.03;

    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average)
        return ssim_map.mean();
    else
        return ssim_map.mean(1).mean(1).mean(1);
}

inline torch::Tensor ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true)
{
    auto channel = img1.size(-3);
    auto window = create_window(window_size, channel, device_type);

    // window = window.to(img1.device());
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

inline torch::Tensor get_loss_rgb(torch::Tensor sampling_map,
    torch::Tensor rendered_rgb, torch::Tensor gt_rgb,
    float lambda_ssim = 0.5f,
    torch::Tensor exposure_a = torch::Tensor(), 
    torch::Tensor exposure_b = torch::Tensor(),
    torch::Tensor mask = torch::Tensor(),
    torch::DeviceType device_type = torch::kCUDA
){
    if(rendered_rgb.device() != device_type) rendered_rgb = rendered_rgb.to(device_type);
    if(gt_rgb.device() != device_type) gt_rgb = gt_rgb.to(device_type);

    if(exposure_a.defined() && exposure_b.defined()){
        if (exposure_a.device() != device_type) exposure_a = exposure_a.to(device_type);
        if (exposure_b.device() != device_type) exposure_b = exposure_b.to(device_type);
        rendered_rgb = rendered_rgb * torch::exp(exposure_a) + exposure_b;
        rendered_rgb = torch::clamp(rendered_rgb, 0.0f, 1.0f);
    }

    torch::Tensor l1, ssim_map;
    
    if (mask.defined()) {
        if (mask.device() != device_type) mask = mask.to(device_type);

        auto masked_rendered_rgb = rendered_rgb * mask;
        auto masked_gt_rgb = gt_rgb * mask;
        // l1 = l1_loss(masked_rendered_rgb, masked_gt_rgb);
        ssim_map = ssim(masked_rendered_rgb, masked_gt_rgb, device_type);

        auto rendered_rgb_flat = rendered_rgb.reshape({-1, 3});
        auto gt_rgb_flat = gt_rgb.reshape({-1, 3});
        auto mask_flat = mask.reshape({-1});

        auto mask_sampled = mask_flat.index_select(0, sampling_map);
        auto valid_indices = torch::nonzero(mask_sampled > 0).reshape({-1});
        auto masked_sampling_map = sampling_map.index_select(0, valid_indices);

        auto sampled_rendered = rendered_rgb_flat.index_select(0, masked_sampling_map);
        auto sampled_gt = gt_rgb_flat.index_select(0, masked_sampling_map);

        l1 = l1_loss(sampled_rendered, sampled_gt);

        // masked_rendered_rgb = masked_rendered_rgb.reshape({-1, 3}).index({sampling_map});
        // masked_gt_rgb = masked_gt_rgb.reshape({-1, 3}).index({sampling_map});
        // l1 = l1_loss(masked_rendered_rgb, masked_gt_rgb);

    } else {
        // l1 = l1_loss(rendered_rgb, gt_rgb);
        ssim_map = ssim(rendered_rgb, gt_rgb, device_type);

        auto rendered_rgb_linear = rendered_rgb.reshape({-1, 3}).index({sampling_map});
        auto gt_rgb_linear = gt_rgb.reshape({-1, 3}).index({sampling_map});
        l1 = l1_loss(rendered_rgb_linear, gt_rgb_linear);
    }

    auto loss_rgb = lambda_ssim * (1.0f - ssim_map) + (1.0f - lambda_ssim) * l1;
    return loss_rgb;
}

inline torch::Tensor get_loss_rgb(
    torch::Tensor rendered_rgb, torch::Tensor gt_rgb,
    float lambda_ssim = 0.5f,
    torch::Tensor exposure_a = torch::Tensor(), 
    torch::Tensor exposure_b = torch::Tensor(),
    torch::Tensor mask = torch::Tensor(),
    torch::DeviceType device_type = torch::kCUDA
){
    if(rendered_rgb.device() != device_type) rendered_rgb = rendered_rgb.to(device_type);
    if(gt_rgb.device() != device_type) gt_rgb = gt_rgb.to(device_type);

    if(exposure_a.defined() && exposure_b.defined()){
        if (exposure_a.device() != device_type) exposure_a = exposure_a.to(device_type);
        if (exposure_b.device() != device_type) exposure_b = exposure_b.to(device_type);
        rendered_rgb = rendered_rgb * torch::exp(exposure_a) + exposure_b;
        rendered_rgb = torch::clamp(rendered_rgb, 0.0f, 1.0f);
    }

    torch::Tensor l1, ssim_map;
    if (mask.defined()) {
        if (mask.device() != device_type) mask = mask.to(device_type);
        auto masked_abs = torch::abs(rendered_rgb - gt_rgb);// * mask;
        l1 = masked_abs.masked_select(mask > 0).mean();
        // l1 = (l1 * mask).mean();
        auto masked_rendered_rgb = rendered_rgb * mask;
        auto masked_gt_rgb = gt_rgb * mask;
        // l1 = l1_loss(masked_rendered_rgb, masked_gt_rgb);
        ssim_map = ssim(masked_rendered_rgb, masked_gt_rgb, device_type);
    } else {
        l1 = l1_loss(rendered_rgb, gt_rgb);
        ssim_map = ssim(rendered_rgb, gt_rgb, device_type);
    }

    auto loss_rgb = lambda_ssim * (1.0f - ssim_map) + (1.0f - lambda_ssim) * l1;
    return loss_rgb;
}

inline torch::Tensor get_loss_depth(
    torch::Tensor rendered_depth, torch::Tensor gt_depth,
    torch::Tensor mask = torch::Tensor(),
    torch::DeviceType device_type = torch::kCUDA
){
    if(rendered_depth.device() != device_type) rendered_depth = rendered_depth.to(device_type);
    if(gt_depth.device() != device_type) gt_depth = gt_depth.to(device_type);

    torch::Tensor loss_d;
    if (mask.defined()) {
        if (mask.device() != device_type) mask = mask.to(device_type);
        loss_d = torch::abs(rendered_depth - gt_depth);
        loss_d = loss_d.masked_select(mask > 0).mean();
    } else {
        loss_d = l1_loss(rendered_depth, gt_depth);
    }
    return loss_d;
}

inline torch::Tensor get_loss_rgbd(
    torch::Tensor rendered_rgb, torch::Tensor gt_rgb,
    torch::Tensor rendered_depth, torch::Tensor gt_depth,
    float lambda_ssim = 0.5f,
    float lambda_depth = 0.1f,
    torch::Tensor exposure_a = torch::Tensor(), 
    torch::Tensor exposure_b = torch::Tensor(),
    torch::Tensor mask = torch::Tensor(),
    torch::DeviceType device_type = torch::kCUDA
){
    auto loss_rgb = get_loss_rgb(rendered_rgb, gt_rgb, lambda_ssim, exposure_a, exposure_b, mask, device_type);

    /*
    torch::Tensor loss_d;
    if (mask.defined()) {
        loss_d = torch::abs(rendered_depth - gt_depth);
        // loss_d = (loss_d * mask).mean();
        loss_d = loss_d.masked_select(mask > 0).mean();
        // auto masked_rendered_depth = rendered_depth * mask;
        // auto masked_gt_depth = gt_depth * mask;
        // loss_d = l1_loss(masked_rendered_depth, masked_gt_depth);
    } else {
        loss_d = l1_loss(rendered_depth, gt_depth);
    }
    */
   
   auto loss_d = get_loss_depth(rendered_depth, gt_depth, mask, device_type);

    auto loss_rgbd = loss_rgb + lambda_depth * loss_d;
    return loss_rgbd;

}

inline torch::Tensor get_loss_pairpose(
    torch::Tensor theta1, torch::Tensor rho1,
    torch::Tensor theta2, torch::Tensor rho2,
    float lambda_theta = 0.5f, float scale = 1.0f,
    float delta = 1e-4f
){

    auto pose1 = tensor_utils::se3_exp(theta1, rho1);
    auto pose2 = tensor_utils::se3_exp(theta2, rho2);

    auto relative_pose = pose1.mm(pose2.inverse());

    auto se3_lie = tensor_utils::se3_log(relative_pose);
    auto relative_theta = std::get<0>(se3_lie) * scale;
    auto relative_rho = std::get<1>(se3_lie) * scale;

    auto loss_rotation = torch::sqrt(relative_theta*relative_theta + delta*delta).mean();
    auto loss_translation = torch::sqrt(relative_rho*relative_rho + delta*delta).mean();

    auto loss_pose = lambda_theta * loss_rotation + (1.0f - lambda_theta) * loss_translation;

    return loss_pose;
}

inline torch::Tensor get_loss_pairpose(
    torch::Tensor T1, torch::Tensor T2,
    float lambda_theta = 0.5f,
    float delta = 1e-3f
){
    auto relative_pose = T1.mm(T2.inverse());

    auto se3_lie = tensor_utils::se3_log(relative_pose);
    auto relative_theta = std::get<0>(se3_lie);
    auto relative_rho = std::get<1>(se3_lie);

    auto loss_rotation = torch::sqrt(relative_theta*relative_theta + delta*delta).mean();
    auto loss_translation = torch::sqrt(relative_rho*relative_rho + delta*delta).mean();

    auto loss_pose = lambda_theta * loss_rotation + (1.0f - lambda_theta) * loss_translation;

    return loss_pose;
}

inline torch::Tensor get_loss_posereg(torch::Tensor T, float lambda_theta = 0.6f){
    auto se3_lie = tensor_utils::se3_log(T);
    auto theta = std::get<0>(se3_lie);
    auto rho = std::get<1>(se3_lie);

    auto loss = lambda_theta * (theta*theta).sum() + (1.f - lambda_theta) * (rho*rho).sum();

    return loss;
}

inline torch::Tensor get_loss_hr2lr(
    torch::Tensor hr_linear_selection_map, torch::Tensor hr_rendered_depth, torch::Tensor hr_rendered_opacity_mask,
    torch::Tensor lr_gt_depth, torch::Tensor lr_gt_depth_mask,
    float hr_fx, float hr_fy, float hr_cx, float hr_cy,
    float lr_fx, float lr_fy, float lr_cx, float lr_cy,
    torch::Tensor T_lr2hr)
{
    torch::Tensor y_coords = torch::div(hr_linear_selection_map, hr_rendered_depth.size(1), "floor");
    torch::Tensor x_coords = torch::remainder(hr_linear_selection_map, hr_rendered_depth.size(1));

    auto valid_mask = hr_rendered_opacity_mask.index({y_coords, x_coords}) > 0.f;
    y_coords = y_coords.index({valid_mask});
    x_coords = x_coords.index({valid_mask});

    auto z_hr = hr_rendered_depth.index({y_coords, x_coords});
    auto x_hr = (x_coords.to(torch::kFloat) - hr_cx) * z_hr / hr_fx;
    auto y_hr = (y_coords.to(torch::kFloat) - hr_cy) * z_hr / hr_fy;
    auto Pc_hr = torch::stack({x_hr, y_hr, z_hr}, 1); // [N,3]

    auto Pc_lr = T_lr2hr.inverse().mm(
        torch::cat({Pc_hr, torch::ones({Pc_hr.size(0), 1}, Pc_hr.options())}, 1).transpose(0, 1)
    ).transpose(0, 1).index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); // [N,3]

    auto X_lr = Pc_lr.index({torch::indexing::Slice(), 0});
    auto Y_lr = Pc_lr.index({torch::indexing::Slice(), 1});
    auto Z_lr = Pc_lr.index({torch::indexing::Slice(), 2});

    auto u_lr = lr_fx * (X_lr / Z_lr) + lr_cx;
    auto v_lr = lr_fy * (Y_lr / Z_lr) + lr_cy;

    auto valid_reproj =
        (Z_lr > 0) &
        (u_lr >= 0) & (u_lr < lr_gt_depth.size(1)) &
        (v_lr >= 0) & (v_lr < lr_gt_depth.size(0));

    if(valid_reproj.sum().item<int>() == 0){
        std::cout << "[Warning][loss_utils::get_loss_hr2lr] No valid HRto LR reprojection points found!" << std::endl;
        return torch::zeros({}, hr_rendered_depth.options());
    }

    u_lr = u_lr.index({valid_reproj});
    v_lr = v_lr.index({valid_reproj});
    // Z_lr = Z_lr.index({valid_reproj});
    auto z_hr_valid = Z_lr.index({valid_reproj});

    auto u_lr_int = u_lr.to(torch::kLong);
    auto v_lr_int = v_lr.to(torch::kLong);

    auto z_lr_sampled = lr_gt_depth.index({v_lr_int, u_lr_int});
    auto mask_lr_sampled = lr_gt_depth_mask.index({v_lr_int, u_lr_int}) > 0.f;

    auto overlap_mask = mask_lr_sampled & (z_lr_sampled > 0.f);
    if (overlap_mask.sum().item<int>() == 0) {
        std::cout << "[Warning][loss_utils::get_loss_hr2lr] No overlapping valid region found!" << std::endl;
        return torch::zeros({}, hr_rendered_depth.options());
    }

    auto z_hr_final = z_hr_valid.index({overlap_mask});
    auto z_lr_final = z_lr_sampled.index({overlap_mask});

    // auto lr_xy_vis = torch::zeros({lr_gt_depth.size(0), lr_gt_depth.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    // lr_xy_vis.index_put_({v_lr_int, u_lr_int}, 1.f);
    // auto image_cv2 = tensor_utils::torchTensor2CvMat_Float32(lr_xy_vis);
    // image_cv2.convertTo(image_cv2, CV_8UC1, 255.0f);
    // cv::imwrite("/home/shaun/Desktop/Photo-SLAM-dev/Photo-SLAM-monogs250918/Photo-SLAM/results/a_rgbd/chair/1105/0_debug/debug_hr2lr_vis.png", image_cv2);

    auto loss = l1_loss(z_hr_final, z_lr_final);
    return loss;
}

inline torch::Tensor get_loss_opacity_delta(
    torch::Tensor referred_opacity,
    torch::Tensor rendered_opacity,
    float w_shrink = 8.0f,
    float w_expand = 1.0f
) {
    auto diff = rendered_opacity - referred_opacity; // negative: black regions getting larger
    auto weight = torch::where(diff < 0, torch::full_like(diff, w_shrink), torch::full_like(diff, w_expand));

    return torch::mean(weight * diff.abs());
    // return torch::mean(weight * diff.pow(2));
}



} // namespace loss_utils