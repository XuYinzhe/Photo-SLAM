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

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/cudaimgproc.hpp>
#include <torch/torch.h>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "third_party/simple-knn/spatial.h"
#include "general_utils.h"
#include "sh_utils.h"

namespace tensor_utils
{

inline void deleter(void* arg) {}

/**
 * @brief 
 * 
 * @param mat  {rows, cols, channels}
 * @param device_type 
 * @return torch::Tensor {channels, rows, cols}
 */
inline torch::Tensor cvMat2TorchTensor_Float32(
    cv::Mat& mat,
    torch::DeviceType device_type)
{
    torch::Tensor mat_tensor, tensor;

    switch (mat.channels())
    {
    case 1:
    {
        mat_tensor = torch::from_blob(mat.data, /*sizes=*/{mat.rows, mat.cols});
        tensor = mat_tensor.clone().to(device_type);
    }
    break;

    case 3:
    {
        mat_tensor = torch::from_blob(mat.data, /*sizes=*/{mat.rows, mat.cols, mat.channels()});
        tensor = mat_tensor.clone().to(device_type);
        tensor = tensor.permute({2, 0, 1});
    }
    break;
    
    default:
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    break;
    }

    return tensor.contiguous();
}

inline cv::Mat torchTensor2CvMat_Float32(torch::Tensor& tensor)
{
    cv::Mat mat;
    torch::Tensor mat_tensor = tensor.cpu().squeeze().clone();

    switch (mat_tensor.ndimension())
    {
    case 2:
    {
        mat = cv::Mat(/*rows=*/mat_tensor.size(0),
                      /*cols=*/mat_tensor.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/mat_tensor.data_ptr<float>());
    }
    break;

    case 3:
    {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
        mat_tensor = mat_tensor.to(torch::kCPU);
        mat = cv::Mat(/*rows=*/mat_tensor.size(0),
                      /*cols=*/mat_tensor.size(1),
                      /*type=*/CV_32FC3,
                      /*data=*/mat_tensor.data_ptr<float>());
    }
    break;
    
    default:
        std::cerr << "The tensor has unsupported number of dimensions!" << std::endl;
    break;
    }

    return mat.clone();
}

inline torch::Tensor cvGpuMat2TorchTensor_Float32(cv::cuda::GpuMat& mat)
{
    torch::Tensor mat_tensor, tensor;
    int64_t step = mat.step / sizeof(float);

    switch (mat.channels())
    {
    case 1:
    {
        std::vector<int64_t> strides = {step, 1};
        mat_tensor = torch::from_blob(
            mat.data,
            /*sizes=*/{mat.rows, mat.cols},
            strides,
            deleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone();
    }
    break;

    case 3:
    {
        std::vector<int64_t> strides = {step, static_cast<int64_t>(mat.channels()), 1};
        mat_tensor = torch::from_blob(
            mat.data,
            /*sizes=*/{mat.rows, mat.cols, mat.channels()},
            strides,
            deleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone().permute({2, 0, 1});
    }
    break;
    
    default:
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    break;
    }

    return tensor.contiguous();
}

inline cv::cuda::GpuMat torchTensor2CvGpuMat_Float32(torch::Tensor& tensor)
{
    cv::cuda::GpuMat mat;
    torch::Tensor mat_tensor = tensor.clone();

    switch (mat_tensor.ndimension())
    {
    case 2:
    {
        mat = cv::cuda::GpuMat(/*rows=*/mat_tensor.size(0),
                               /*cols=*/mat_tensor.size(1),
                               /*type=*/CV_32FC1,
                               /*data=*/mat_tensor.data_ptr<float>());
    }
    break;

    case 3:
    {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
        mat = cv::cuda::GpuMat(/*rows=*/mat_tensor.size(0),
                               /*cols=*/mat_tensor.size(1),
                               /*type=*/CV_32FC3,
                               /*data=*/mat_tensor.data_ptr<float>());
    }
    break;

    default:
        std::cerr << "The tensor has unsupported number of channels!" << std::endl;
    break;
    }

    return mat.clone();
}

inline torch::Tensor EigenMatrix2TorchTensor(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto eigen_matrix_T = eigen_matrix;
    eigen_matrix_T.transposeInPlace();
    torch::Tensor tensor = torch::from_blob(
        /*data=*/eigen_matrix_T.data(),
        /*sizes=*/{eigen_matrix.rows(), eigen_matrix.cols()},
        /*options=*/torch::TensorOptions().dtype(torch::kFloat)
    ).clone();

    tensor = tensor.to(device_type);
    return tensor;
}

inline Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> TorchTensor2EigenMatrix(torch::Tensor& tensor)
{
    torch::Tensor tensor_cpu = tensor.cpu().contiguous();
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_matrix_map(
        tensor_cpu.data_ptr<float>(),
        tensor_cpu.size(0),
        tensor_cpu.size(1)
    );
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix = eigen_matrix_map;
    return eigen_matrix;
}

inline Sophus::SE3f TensorTransformation2SE3f(const torch::Tensor& T) {
    torch::Tensor temp = T.cpu().contiguous();
    const float* data_ptr = temp.data_ptr<float>();
    
    if (temp.numel() != 16) {
        std::cerr << "[error] Tensor must have 16 elements for 4x4 matrix in `TensorTransformation2SE3f`" << std::endl;
        return Sophus::SE3f(); 
    }

    // Eigen::Map<const Eigen::Matrix4f> Te(data_ptr);  // Use default column-major mapping
    Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> Te(data_ptr);
    Eigen::Matrix3f R = Te.block<3,3>(0,0);
    Eigen::Vector3f t = Te.block<3,1>(0,3);
    
    Eigen::Quaternionf q(R);

    if (q.squaredNorm() < 1e-10f) {
        q = Eigen::Quaternionf::Identity();
        std::cout << "[warning] Invalid quaternion in `TensorTransformation2SE3f`. Reset to identity." << std::endl;
        return Sophus::SE3f();
    } else {
        q.normalize();
    }
    
    return Sophus::SE3f(q, t);
}

inline torch::Tensor SE3f2TensorTransformation(const Sophus::SE3f& T) {
    Eigen::Matrix4f Te = Eigen::Matrix4f::Identity();
    Te.block<3,3>(0,0) = T.so3().matrix();
    Te.block<3,1>(0,3) = T.translation();

    auto out = torch::empty({4,4}, torch::dtype(torch::kFloat32));
    std::memcpy(out.data_ptr<float>(), Te.data(), 16 * sizeof(float));
    out = out.permute({1, 0});
    return out;
}

inline torch::Tensor se3_exp(torch::Tensor omega, torch::Tensor v) {
    // Rotation SO3
    torch::Tensor theta = torch::norm(omega);

    torch::Tensor omega_hat = torch::zeros({3, 3}, omega.options());
    omega_hat[0][1] = -omega[2]; omega_hat[0][2] = omega[1];
    omega_hat[1][0] = omega[2];  omega_hat[1][2] = -omega[0];
    omega_hat[2][0] = -omega[1]; omega_hat[2][1] = omega[0];
    torch::Tensor omega_hat2 = torch::matmul(omega_hat, omega_hat);

    torch::Tensor R = torch::eye(3, omega.options());
    auto theta_val = theta.item<float>();
    if (theta_val > 1e-5) {
        torch::Tensor A = torch::sin(theta) / theta;
        torch::Tensor B = (1 - torch::cos(theta)) / (theta * theta);
        R = R + A * omega_hat + B * omega_hat2;
    } else 
        R = R + omega_hat + 0.5 * omega_hat2;

    // Translation
    torch::Tensor V = torch::eye(3, omega.options());
    if (theta_val > 1e-5) {
        torch::Tensor C = (1 - torch::cos(theta)) / (theta * theta);
        torch::Tensor D = (theta - torch::sin(theta)) / (theta * theta * theta);
        V = V + C * omega_hat + D * omega_hat2;
    } else 
        V = V + 0.5 * omega_hat + (1.0/6.0) * omega_hat2;

    torch::Tensor t = torch::matmul(V, v.unsqueeze(-1)).squeeze(-1);

    // SE3
    torch::Tensor T = torch::eye(4, omega.options());
    T.slice(0, 0, 3).slice(1, 0, 3) = R;
    T.slice(0, 0, 3).slice(1, 3, 4) = t.unsqueeze(-1);
    return T;
}

inline std::tuple<torch::Tensor, torch::Tensor> se3_log(torch::Tensor T) {
    torch::Tensor R = T.slice(0, 0, 3).slice(1, 0, 3);
    torch::Tensor t = T.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

    // unscaled rotation vector from R
    torch::Tensor omega_unscaled = torch::stack({
        R.index({2, 1}) - R.index({1, 2}),  // R[2,1] - R[1,2]
        R.index({0, 2}) - R.index({2, 0}),  // R[0,2] - R[2,0]
        R.index({1, 0}) - R.index({0, 1})   // R[1,0] - R[0,1]
    });

    // rotation angle (theta)
    torch::Tensor trace_R = R.index({0, 0}) + R.index({1, 1}) + R.index({2, 2});
    torch::Tensor tr_clamped = torch::clamp((trace_R - 1.0) / 2.0, -1.0, 1.0);
    torch::Tensor theta = torch::acos(tr_clamped);

    torch::Tensor omega, v;
    float theta_val = theta.item<float>();
    
    if (theta_val < 1e-5) {
        // Rotation
        omega = 0.5 * omega_unscaled;
        
        // Translation
        auto cross1 = torch::cross(omega, t);  // ω × t
        auto cross2 = torch::cross(omega, cross1);  // ω × (ω × t)
        v = t - 0.5 * cross1 + (1.0 / 12.0) * cross2;
    } else {
        omega = (theta / (2.0 * torch::sin(theta))) * omega_unscaled;
        
        torch::Tensor sin_theta = torch::sin(theta);
        torch::Tensor cos_theta = torch::cos(theta);
        torch::Tensor one_minus_cos_theta = 1.0 - cos_theta;
        torch::Tensor A = (theta * sin_theta) / (2.0 * one_minus_cos_theta);
        torch::Tensor factor = (1.0 - A) / (theta * theta);

        // skew-symmetric matrix for omega
        torch::Tensor omega_skew = torch::zeros({3, 3}, T.options());
        omega_skew[0][1] = -omega[2]; omega_skew[0][2] = omega[1];
        omega_skew[1][0] = omega[2];  omega_skew[1][2] = -omega[0];
        omega_skew[2][0] = -omega[1]; omega_skew[2][1] = omega[0];

        // V^{-1}
        torch::Tensor omega_skew2 = torch::mm(omega_skew, omega_skew);
        torch::Tensor V_inv = torch::eye(3, T.options()) - 0.5 * omega_skew + factor * omega_skew2;
        v = torch::mm(V_inv, t.unsqueeze(1)).squeeze(1);
    }

    return std::make_tuple(omega, v);
}

inline torch::Tensor rotation_to_quat(const torch::Tensor& R) {
    auto trace = R.trace();
    torch::Tensor q = torch::zeros({4}, R.options());
    if (trace.item<float>() > 0.0f) {
        float s = std::sqrt(trace.item<float>() + 1.0f) * 2.0f;
        q[3] = 0.25f * s;
        q[0] = (R[2][1].item<float>() - R[1][2].item<float>()) / s;
        q[1] = (R[0][2].item<float>() - R[2][0].item<float>()) / s;
        q[2] = (R[1][0].item<float>() - R[0][1].item<float>()) / s;
    } else {
        // Handle diagonal cases if trace is small
        if ((R[0][0].item<float>() > R[1][1].item<float>()) && 
            (R[0][0].item<float>() > R[2][2].item<float>())) {
            float s = std::sqrt(1.0f + R[0][0].item<float>() - R[1][1].item<float>() - R[2][2].item<float>()) * 2.0f;
            q[3] = (R[2][1].item<float>() - R[1][2].item<float>()) / s;
            q[0] = 0.25f * s;
            q[1] = (R[0][1].item<float>() + R[1][0].item<float>()) / s;
            q[2] = (R[0][2].item<float>() + R[2][0].item<float>()) / s;
        } else if (R[1][1].item<float>() > R[2][2].item<float>()) {
            float s = std::sqrt(1.0f + R[1][1].item<float>() - R[0][0].item<float>() - R[2][2].item<float>()) * 2.0f;
            q[3] = (R[0][2].item<float>() - R[2][0].item<float>()) / s;
            q[0] = (R[0][1].item<float>() + R[1][0].item<float>()) / s;
            q[1] = 0.25f * s;
            q[2] = (R[1][2].item<float>() + R[2][1].item<float>()) / s;
        } else {
            float s = std::sqrt(1.0f + R[2][2].item<float>() - R[0][0].item<float>() - R[1][1].item<float>()) * 2.0f;
            q[3] = (R[1][0].item<float>() - R[0][1].item<float>()) / s;
            q[0] = (R[0][2].item<float>() + R[2][0].item<float>()) / s;
            q[1] = (R[1][2].item<float>() + R[2][1].item<float>()) / s;
            q[2] = 0.25f * s;
        }
    }
    return q / q.norm();
}

inline torch::Tensor quat_to_rotation(const torch::Tensor& q) {
    auto qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    torch::Tensor R = torch::empty({3,3}, q.options());
    R[0][0] = 1 - 2*(qy*qy + qz*qz);
    R[0][1] = 2*(qx*qy - qz*qw);
    R[0][2] = 2*(qx*qz + qy*qw);
    R[1][0] = 2*(qx*qy + qz*qw);
    R[1][1] = 1 - 2*(qx*qx + qz*qz);
    R[1][2] = 2*(qy*qz - qx*qw);
    R[2][0] = 2*(qx*qz - qy*qw);
    R[2][1] = 2*(qy*qz + qx*qw);
    R[2][2] = 1 - 2*(qx*qx + qy*qy);
    return R;
}

inline void initGaussianOpacity(torch::Tensor& xyz, torch::Tensor& opacity, float init_opacity = 0.3f, bool requires_grad = false){
    int num_points = xyz.size(0);
    opacity = general_utils::inverse_sigmoid(
        init_opacity * torch::ones(
            {num_points, 1}, torch::TensorOptions().dtype(torch::kFloat).device(xyz.device()))
    );
    opacity = opacity.contiguous();

    if (requires_grad) opacity.requires_grad_();
}

inline void initGaussianScaling(torch::Tensor& xyz, torch::Tensor& scaling, float median_clamp_ratio = 3.f, bool requires_grad = false){
    int num_points = xyz.size(0);

    torch::Tensor xyz_copy = xyz.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(xyz_copy), 0.0000001);

    if(median_clamp_ratio > 0.f){
        torch::Tensor clmap_dist = torch::sqrt(dist2);
        float median_scale = clmap_dist.median().item<float>() * median_clamp_ratio;
        clmap_dist = torch::clamp_max(clmap_dist, median_scale);
        scaling = torch::log(clmap_dist);
    }
    else scaling = torch::log(torch::sqrt(dist2));

    auto scales_ndimension = scaling.ndimension();
    scaling = scaling.unsqueeze(scales_ndimension).repeat({1, 3}).contiguous();

    if (requires_grad) scaling.requires_grad_();
}

inline void initGaussianRotation(torch::Tensor& xyz, torch::Tensor& rotation, bool requires_grad = false){
    int num_points = xyz.size(0);
    rotation = torch::zeros(
        {num_points, 4}, torch::TensorOptions().dtype(torch::kFloat).device(xyz.device()));
    rotation.index({torch::indexing::Slice(), 0}) = 1.0f;
    rotation = rotation.contiguous();

    if (requires_grad) rotation.requires_grad_();
}

inline void initGaussianFeatures(
    torch::Tensor& colors,
    torch::Tensor& features_dc, torch::Tensor& features_rest,
    int max_sh_degree = 3, bool requires_grad = false
){
    int num_points = colors.size(0);
    torch::Tensor fused_colors = sh_utils::RGB2SH(colors);
    int feature_dim = (max_sh_degree + 1) * (max_sh_degree + 1);

    torch::Tensor features = torch::zeros(
        {num_points, 3, feature_dim},
        torch::TensorOptions().dtype(torch::kFloat).device(colors.device()));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = fused_colors;

    features_dc = features.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(0, 1)})
                             .transpose(1, 2)
                             .contiguous();
    features_rest = features.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(),
                                    torch::indexing::Slice(1, features.size(2))})
                               .transpose(1, 2)
                               .contiguous();

    if (requires_grad) {
        features_dc.requires_grad_();
        features_rest.requires_grad_();
    }


}

inline torch::Tensor predict_pose_cubic_hermite(
    const std::vector<torch::Tensor>& poses,
    const std::vector<float>& times,
    float t_pred
){
    auto [omega0, rho0] = se3_log(poses[0]);
    auto [omega1, rho1] = se3_log(poses[1]);
    auto [omega2, rho2] = se3_log(poses[2]);

    // auto se3_0 = tensor_utils::se3_log(poses[0]);
    // auto se3_1 = tensor_utils::se3_log(poses[1]);
    // auto se3_2 = tensor_utils::se3_log(poses[2]);

    // auto omega0 = std::get<0>(se3_0);
    // auto rho0 = std::get<1>(se3_0);
    // auto omega1 = std::get<0>(se3_1);
    // auto rho1 = std::get<1>(se3_1);
    // auto omega2 = std::get<0>(se3_2);
    // auto rho2 = std::get<1>(se3_2);

    float t0 = times[0];
    float t1 = times[1];
    float t2 = times[2];

    auto xi0 = torch::cat({omega0, rho0}, 0);
    auto xi1 = torch::cat({omega1, rho1}, 0);
    auto xi2 = torch::cat({omega2, rho2}, 0);

    auto dxi2 = (xi2 - xi1) / (t2 - t1);

    auto xi_pred = xi2.clone();

    if (t2 < t_pred) {
        float dt_future = t_pred - t2;
        float dt = t2 - t1;
        float u = dt_future / dt;

        auto h00 = 2*pow(u,3) - 3*pow(u,2) + 1;
        auto h10 = pow(u,3) - 2*pow(u,2) + u;
        auto h01 = -2*pow(u,3) + 3*pow(u,2);
        auto h11 = pow(u,3) - pow(u,2);

        xi_pred = h00 * xi2 + h10 * dt * dxi2 + h01 * (xi2 + dt * dxi2) + h11 * dt * dxi2;
    } else {
        float u = (t_pred - t1) / (t2 - t1);
        xi_pred = xi1 + u * (xi2 - xi1);
    }

    auto delta_xi = xi_pred - xi2;
    auto omega = delta_xi.slice(0, 0, 3);
    auto v = delta_xi.slice(0, 3, 6);

    auto T_delta = se3_exp(omega, v);
    // auto T_pred = T_delta.mm(poses[2]);

    return T_delta;
}

inline torch::Tensor average_poses(std::vector<torch::Tensor> poses, std::vector<float> weights = {}) {
    std::vector<torch::Tensor> omegas, rhos;
    for (const auto& pose : poses) {
        auto [omega, rho] = se3_log(pose);
        omegas.push_back(omega);
        rhos.push_back(rho);
    }
    
    torch::Tensor omega_avg, rho_avg;
    if (weights.empty()) {
        omega_avg = torch::stack(omegas).mean(0);
        rho_avg = torch::stack(rhos).mean(0);
    } else {
        torch::Tensor weight_tensor = torch::tensor(weights, torch::dtype(torch::kFloat32).device(poses[0].device())).unsqueeze(1);
        omega_avg = torch::stack(omegas).mul(weight_tensor).sum(0) / weight_tensor.sum();
        rho_avg = torch::stack(rhos).mul(weight_tensor).sum(0) / weight_tensor.sum();
    }

    return se3_exp(omega_avg, rho_avg);
}

} // namespace tensor_utils