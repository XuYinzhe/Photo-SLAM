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

#include <torch/torch.h>

namespace general_utils
{

inline torch::Tensor inverse_sigmoid(const torch::Tensor &x)
{
    return torch::log(x / (1 - x));
}

inline torch::Tensor build_rotation(torch::Tensor &r)
{
    auto r0 = r.index({torch::indexing::Slice(), 0});
    auto r1 = r.index({torch::indexing::Slice(), 1});
    auto r2 = r.index({torch::indexing::Slice(), 2});
    auto r3 = r.index({torch::indexing::Slice(), 3});
    auto norm = torch::sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);

    auto q = r / norm.unsqueeze(/*dim=*/1);
    r = q.index({torch::indexing::Slice(), 0});
    auto x = q.index({torch::indexing::Slice(), 1});
    auto y = q.index({torch::indexing::Slice(), 2});
    auto z = q.index({torch::indexing::Slice(), 3});

    auto R = torch::zeros({q.size(0), 3, 3}, torch::TensorOptions().device(torch::kCUDA));
    R.select(1, 0).select(1, 0).copy_(1 - 2 * (y * y + z * z));
    R.select(1, 0).select(1, 1).copy_(2 * (x * y - r * z));
    R.select(1, 0).select(1, 2).copy_(2 * (x * z + r * y));
    R.select(1, 1).select(1, 0).copy_(2 * (x * y + r * z));
    R.select(1, 1).select(1, 1).copy_(1 - 2 * (x * x + z * z));
    R.select(1, 1).select(1, 2).copy_(2 * (y * z - r * x));
    R.select(1, 2).select(1, 0).copy_(2 * (x * z - r * y));
    R.select(1, 2).select(1, 1).copy_(2 * (y * z + r * x));
    R.select(1, 2).select(1, 2).copy_(1 - 2 * (x * x + y * y));
    return R;
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

}
