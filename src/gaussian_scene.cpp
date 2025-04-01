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

#include "include/gaussian_scene.h"

GaussianScene::GaussianScene(
    GaussianModelParams& args,
    int load_iteration,
    bool shuffle,
    std::vector<float> resolution_scales)
{
    if (load_iteration)
    {
        this->loaded_iter_ = load_iteration;
        std::cout << "Loading trained model at iteration " << load_iteration << std::endl;
    }
}

void GaussianScene::addCamera(Camera& camera)
{
    this->cameras_.emplace(camera.camera_id_, camera);
}

Camera& GaussianScene::getCamera(camera_id_t cameraId)
{
    return this->cameras_[cameraId];
}

void GaussianScene::addKeyframe(std::shared_ptr<GaussianKeyframe> new_kf, bool* shuffled)
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    new_kf->registerOptimParams();
    this->keyframes_.emplace(new_kf->fid_, new_kf);
    *shuffled = false;

    if(!this->optimizer_initialized_)return;

    if(new_kf->enable_optim_pose){
        std::vector<torch::Tensor> temp_param1 = {new_kf->cam_rot_delta_};
        std::vector<torch::Tensor> temp_param2 = {new_kf->cam_trans_delta_};
        this->optimizer_->add_param_group(temp_param1);
        optimizer_->param_groups().back().options().set_lr(init_cam_rotation_lr_);
        this->optimizer_->add_param_group(temp_param2);
        optimizer_->param_groups().back().options().set_lr(init_cam_translation_lr_);
    }
    if(new_kf->enable_optim_velocity){
        std::vector<torch::Tensor> temp_param1 = {new_kf->cam_lin_vel_};
        std::vector<torch::Tensor> temp_param2 = {new_kf->cam_ang_vel_};
        this->optimizer_->add_param_group(temp_param1);
        optimizer_->param_groups().back().options().set_lr(init_cam_linear_velocity_lr_);
        this->optimizer_->add_param_group(temp_param2);
        optimizer_->param_groups().back().options().set_lr(init_cam_angular_velocity_lr_);
    }
    if(new_kf->enable_optim_exposure){
        std::vector<torch::Tensor> temp_param = {new_kf->exposure_a_, new_kf->exposure_b_};
        this->optimizer_->add_param_group(temp_param);
        optimizer_->param_groups().back().options().set_lr(init_cam_exposure_lr_);
    }
}

std::shared_ptr<GaussianKeyframe>
GaussianScene::getKeyframe(std::size_t fid)
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    if (this->keyframes_.find(fid) != this->keyframes_.end())
        return this->keyframes_[fid];
    else
        return nullptr;
}

std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
GaussianScene::keyframes()
{
    return this->keyframes_;
}

std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
GaussianScene::getAllKeyframes()
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    return this->keyframes_;
}

void GaussianScene::cachePoint3D(point3D_id_t point3D_id, Point3D& point3d)
{
    this->cached_point_cloud_[point3D_id] = point3d;
}

Point3D& GaussianScene::getPoint3D(point3D_id_t point3DId)
{
    if (this->cached_point_cloud_.find(point3DId) == this->cached_point_cloud_.end())
        std::cout << "GaussianScene::getPoint3D(" << point3DId << ") invalid point Id, creating new point." << std::endl;

    return this->cached_point_cloud_[point3DId];
}

void GaussianScene::clearCachedPoint3D()
{
    this->cached_point_cloud_.clear();
}

void GaussianScene::applyScaledTransformation(
    const float s,
    const Sophus::SE3f T)
{
    // Apply the scaled transformation on gaussian keyframes
    for (auto& kfit : keyframes_) {
        std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
        Sophus::SE3f Twc = pkf->getPosef().inverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = T * Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
        pkf->computeTransformTensors();
    }
}

/**
 * @brief 
 * 
 * @return std::tuple<Eigen::Vector3f, float> first=translate, second=radius
 */
std::tuple<Eigen::Vector3f, float>
GaussianScene::getNerfppNorm()
{
    std::vector<Eigen::Matrix<float, 3, 1>> cam_centers;
    auto kfs = this->getAllKeyframes();
    std::size_t n_cams = kfs.size();
    cam_centers.reserve(n_cams);
    for (auto& kfit : kfs) {
        auto pkf = kfit.second;
        auto W2C = pkf->getWorld2View2();
        auto C2W = W2C.inverse();
        auto cam_center = C2W.block<3, 1>(0, 3);
        cam_centers.emplace_back(cam_center);
    }

    // get_center_and_diag(cam_centers)
    Eigen::Vector3f avg_cam_center;
    avg_cam_center.setZero();
    for (const auto& cam_center : cam_centers) {
        avg_cam_center.x() += cam_center.x();
        avg_cam_center.y() += cam_center.y();
        avg_cam_center.z() += cam_center.z();
    }
    avg_cam_center.x() /= n_cams;
    avg_cam_center.y() /= n_cams;
    avg_cam_center.z() /= n_cams;

    float max_dist = 0.0f; // diagonal
    for (std::size_t cam_idx = 0; cam_idx < n_cams; ++cam_idx) {
        float dist = (cam_centers[cam_idx] - avg_cam_center).norm();
        if (dist > max_dist)
            max_dist = dist;
    }

    float radius = max_dist * 1.1;

    Eigen::Vector3f translate = -avg_cam_center;

    return std::make_tuple(translate, radius);
}

void GaussianScene::setupKeyframeOptimization(const GaussianOptimizationParams& opt_params) {
    // optimizer_.reset(nullptr);
    this->init_cam_rotation_lr_ = opt_params.cam_rotation_lr_;
    this->init_cam_translation_lr_ = opt_params.cam_translation_lr_;
    this->init_cam_linear_velocity_lr_ = opt_params.cam_linear_velocity_lr_;
    this->init_cam_angular_velocity_lr_ = opt_params.cam_angular_velocity_lr_;
    this->init_cam_exposure_lr_ = opt_params.cam_exposure_lr_;

    std::vector<torch::Tensor> pose_rot_params, pose_trans_params, vel_lin_params, vel_ang_params, exposure_params;

    for (auto& kfit : this->keyframes_) {
        auto& keyframe = kfit.second;
        keyframe->registerOptimParams();

        if (keyframe->enable_optim_pose) {
            pose_rot_params.push_back(keyframe->cam_rot_delta_);
            pose_trans_params.push_back(keyframe->cam_trans_delta_);
        }
        if (keyframe->enable_optim_velocity) {
            vel_lin_params.push_back(keyframe->cam_lin_vel_);
            vel_ang_params.push_back(keyframe->cam_ang_vel_);
        }
        if (keyframe->enable_optim_exposure) {
            exposure_params.push_back(keyframe->exposure_a_);
            exposure_params.push_back(keyframe->exposure_b_);
        }
    }

    this->optimizer_.reset(new torch::optim::Adam(pose_rot_params, torch::optim::AdamOptions(opt_params.cam_rotation_lr_)));
    this->optimizer_->add_param_group(pose_trans_params);
    this->optimizer_->param_groups().back().options().set_lr(opt_params.cam_translation_lr_);
    this->optimizer_->add_param_group(vel_lin_params);
    this->optimizer_->param_groups().back().options().set_lr(opt_params.cam_linear_velocity_lr_);
    this->optimizer_->add_param_group(vel_ang_params);
    this->optimizer_->param_groups().back().options().set_lr(opt_params.cam_angular_velocity_lr_);
    this->optimizer_->add_param_group(exposure_params);
    this->optimizer_->param_groups().back().options().set_lr(opt_params.cam_exposure_lr_);
    this->optimizer_initialized_ = true;
}