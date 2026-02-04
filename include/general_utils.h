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

#include <tuple>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

// #include "third_party/GMS-Feature-Matcher/include/gms_matcher.h"

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

inline void projectEigen_depth2pcd(
    Eigen::MatrixXf& P_w,
    const float cx, const float cy,
    const float fx, const float fy,
    const Eigen::ArrayXf& x_grids, 
    const Eigen::ArrayXf& y_grids,
    const Eigen::ArrayXf& z_map,
    const Sophus::SE3f& pose
){
    // Tcw is world to camera
    auto R = pose.inverse().rotationMatrix();
    auto t = pose.inverse().translation();
    // Tcw is camera to world
    // auto R = pose.rotationMatrix();
    // auto t = pose.translation();

    auto x_map = (x_grids - cx) * z_map * (1.f/fx);
    auto y_map = (y_grids - cy) * z_map * (1.f/fy);

    Eigen::MatrixXf P_c(3, z_map.size());
    P_c.row(0) = x_map.eval().transpose();
    P_c.row(1) = y_map.eval().transpose();
    P_c.row(2) = z_map.eval().transpose();

    P_w = (R * P_c).colwise() + t;
}

inline void projectEigen_pcd2depth(
    cv::Mat& depths,
    const float cx, const float cy,
    const float fx, const float fy,
    const Eigen::MatrixXf& P_w,
    const Sophus::SE3f& pose
){
    // Tcw is world to camera
    auto R = pose.rotationMatrix();
    auto t = pose.translation();
    // Tcw is camera to world
    // auto R = pose.inverse().rotationMatrix();
    // auto t = pose.inverse().translation();

    Eigen::MatrixXf P_c(3, P_w.cols());
    P_c = (R * P_w).colwise() + t;

    Eigen::ArrayXf z_map = P_c.row(2).transpose().array();
    Eigen::ArrayXf x_grids = P_c.row(0).transpose().array() * fx / z_map + cx;
    Eigen::ArrayXf y_grids = P_c.row(1).transpose().array() * fy / z_map + cy;

    cv::Mat xcv(1, x_grids.size(), CV_32F, (void*)x_grids.data());
    cv::Mat ycv(1, y_grids.size(), CV_32F, (void*)y_grids.data());
    cv::Mat zcv(1, z_map.size(),   CV_32F, (void*)z_map.data());

    xcv = xcv.reshape(1, depths.rows);
    ycv = ycv.reshape(1, depths.rows);
    zcv = zcv.reshape(1, depths.rows);

    cv::remap(zcv, depths, xcv, ycv, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
}

inline cv::Mat readRGB2cvMat(const std::string& path, bool undistort=false, const cv::Mat& map1=cv::Mat(), const cv::Mat& map2=cv::Mat()){
    cv::Mat rgb = cv::imread(path, cv::IMREAD_UNCHANGED);
    cv::cvtColor(rgb, rgb, CV_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.f/255.f);

    if(undistort && !map1.empty() && !map2.empty()){
        cv::Mat rgb_undist;
        cv::remap(rgb, rgb_undist, map1, map2, cv::InterpolationFlags::INTER_LINEAR);
        return rgb_undist;
    }
    
    return rgb;
}

inline bool findMinWithIndex(const std::vector<float>& vec, float& min_val, size_t& index) {
    if (vec.empty()) {
        index = static_cast<size_t>(-1); // Sentinel for invalid index
        min_val = 0.0f; // Placeholder value
        return false; // Indicate failure
    }
    auto it = std::min_element(vec.begin(), vec.end());
    min_val = *it;
    index = it - vec.begin(); // Compute index via iterator arithmetic
    return true; // Indicate success
}

template<typename T>
inline T nth_largest_in_mat(const cv::Mat& m, size_t n) {
    CV_Assert(m.channels() == 1);
    size_t total = static_cast<size_t>(m.total());
    if (n == 0 || n > total) throw std::out_of_range("n out of range (1-based).");

    std::vector<T> vals;
    vals.reserve(total);
    if (m.isContinuous()) {
        const T* ptr = m.ptr<T>(0);
        vals.assign(ptr, ptr + total);
    } else {
        for (int r = 0; r < m.rows; ++r) {
            const T* row = m.ptr<T>(r);
            vals.insert(vals.end(), row, row + m.cols);
        }
    }

    size_t k = total - n;
    std::nth_element(vals.begin(), vals.begin() + k, vals.end()); // ascending
    return vals[k];
}

inline cv::Mat getDepthVarianceMask(const cv::Mat& depth_in, int window_size, float threshold_ratio = 1.f) {
    CV_Assert(depth_in.channels() == 1);
    CV_Assert(window_size > 0);

    auto depth = depth_in.clone();

    int w = window_size;
    if (w % 2 == 0) w++;

    cv::Mat depth_float;
    if (depth.depth() == CV_32F) depth_float = depth;
    else depth.convertTo(depth_float, CV_32F);

    cv::Mat mean, mean_square, variance;
    cv::boxFilter(depth_float, mean, CV_32F, cv::Size(w, w), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::sqrBoxFilter(depth_float, mean_square, CV_32F, cv::Size(w, w), cv::Point(-1, -1), true, cv::BORDER_REFLECT);

    cv::multiply(mean, mean, variance);
    variance = mean_square - variance;

    // double min_val, max_val;
    // cv::minMaxLoc(variance, &min_val, &max_val);
    // std::cout << "Depth variance range: [" << min_val << ", " << max_val << "]" << " mean: " << cv::mean(variance)[0] << std::endl;

    float variance_threshold = (cv::mean(variance))[0] * threshold_ratio;

    cv::Mat stable_mask = (variance < variance_threshold);
    stable_mask.convertTo(stable_mask, CV_32F, 1.f/255.f);

    return stable_mask;
}

inline torch::Tensor fast_pixel_selector(torch::Tensor input, int num_points = 10000, torch::DeviceType device_type = torch::kCPU) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(input.dim() == 2 || (input.dim() == 3 && input.size(0) == 3),
                "[error][general_utils::fast_pixel_selector] Input must be HxW or 3xHxW");
    
    torch::Tensor gray = input;
    if (input.dim() == 3)
        gray = input.mean(0);
    if (gray.device().type() != device_type)
        gray = gray.to(device_type);

    torch::Tensor kx = torch::tensor({{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}, torch::kFloat32).to(device_type);
    torch::Tensor ky = torch::tensor({{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}, torch::kFloat32).to(device_type);
    
    torch::Tensor dx = torch::conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                    kx.unsqueeze(0).unsqueeze(0), 
                                    {}, 1, 1).squeeze();
    torch::Tensor dy = torch::conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                    ky.unsqueeze(0).unsqueeze(0), 
                                    {}, 1, 1).squeeze();

    torch::Tensor grad_mag = dx.square() + dy.square();
    grad_mag.index_put_({torch::indexing::Slice(0,1), torch::indexing::Slice()}, 0.f);
    grad_mag.index_put_({torch::indexing::Slice(grad_mag.size(0)-1, grad_mag.size(0)), torch::indexing::Slice()}, 0.f);
    grad_mag.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0,1)}, 0.f);
    grad_mag.index_put_({torch::indexing::Slice(), torch::indexing::Slice(grad_mag.size(1)-1, grad_mag.size(1))}, 0.f);

    auto flattened = grad_mag.view(-1);
    auto k = static_cast<int>(0.1 * flattened.size(0));
    std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(flattened, k, 0);
    float threshold = std::get<0>(topk)[k-1].item<float>();

    torch::Tensor candidate_mask = (grad_mag > threshold);
    int num_candidates = candidate_mask.sum().item<int>();

    if (num_candidates == 0)
        return torch::zeros_like(grad_mag);
    num_points = std::min(num_points, num_candidates);

    auto indices = torch::where(candidate_mask);
    torch::Tensor candidate_y = indices[0];
    torch::Tensor candidate_x = indices[1];

    auto device = grad_mag.device();
    auto long_dtype = torch::kLong;
    auto options = torch::TensorOptions().dtype(long_dtype).device(device);
    
    torch::Tensor perm = torch::randperm(num_candidates, options);
    torch::Tensor selected_indices = perm.slice(0, 0, num_points);

    torch::Tensor selected_y = candidate_y.index({selected_indices});
    torch::Tensor selected_x = candidate_x.index({selected_indices});

    torch::Tensor selection_map = torch::zeros_like(grad_mag);
    selection_map.index_put_(
        {selected_y, selected_x},
        1.f
    );

    return selection_map;
}

inline int gms_matcher_selector(
    const std::vector<bool>& vbInliers, 
    const std::vector<cv::DMatch>& matches_all,
    const std::vector<cv::KeyPoint>& kps1,
    const std::vector<cv::KeyPoint>& kps2,
    std::vector<int>& vnMatches12,
    std::vector<int>& vnMatches21
){
    int nGoodMatches = 0;
    for (std::size_t i = 0; i < vbInliers.size(); ++i){
        if(!vbInliers[i]) continue;

        int idx1 = matches_all[i].queryIdx;
        int idx2 = matches_all[i].trainIdx;

        if(kps1[idx1].octave > 2 || kps2[idx2].octave > 2) continue;

        if (vnMatches12[idx1] == -1 && vnMatches21[idx2] == -1){
            vnMatches12[idx1] = idx2;
            vnMatches21[idx2] = idx1;
            nGoodMatches++;
        }
    }
    return nGoodMatches;
}

inline cv::Mat vecs2transformation(const cv::Mat& rvec, const cv::Mat& tvec){
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat T = cv::Mat::eye(4, 4, CV_32F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));

    return T;
}

inline torch::Tensor erode_mask(torch::Tensor mask, int kernel_size = 3) {
    // mask is expected to be (H, W) or (1, 1, H, W)
    if (mask.dim() == 2)
        mask = mask.unsqueeze(0).unsqueeze(0); // (N=1, C=1, H, W)
    else if (mask.dim() == 3)
        mask = mask.unsqueeze(0);

    auto neg_mask = -mask;

    auto eroded = -torch::nn::functional::max_pool2d(
        neg_mask,
        torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
            .stride(1)
            .padding(kernel_size / 2)
    );

    return eroded.squeeze(); // back to (H, W)
}

inline torch::Tensor dilate_mask(torch::Tensor mask, int kernel_size = 3) {
    // mask is expected to be (H, W) or (1, 1, H, W)
    if (mask.dim() == 2)
        mask = mask.unsqueeze(0).unsqueeze(0); // (N=1, C=1, H, W)
    else if (mask.dim() == 3)
        mask = mask.unsqueeze(0);

    auto dilated = torch::nn::functional::max_pool2d(
        mask,
        torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
            .stride(1)
            .padding(kernel_size / 2)
    );

    return dilated.squeeze(); // back to (H, W)
}

inline torch::Tensor merge_large_components(const torch::Tensor& labeled_mask, int min_pixels = 10000){
    const auto flat = labeled_mask.flatten();

    auto num_labels = labeled_mask.max().item<int64_t>() + 1;
    auto counts = torch::bincount(flat, {}, num_labels);

    auto keep = (counts > min_pixels).to(torch::kUInt8);
    if (keep.size(0) > 0) keep[0] = 0; // background -> 0

    auto merged = keep.index_select(0, flat.to(torch::kLong))
                        .view(labeled_mask.sizes())
                        .to(torch::kFloat32);
    return merged;
}

} // namespace general_utils
