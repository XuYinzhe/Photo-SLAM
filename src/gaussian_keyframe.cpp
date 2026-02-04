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

#include "include/gaussian_keyframe.h"

#include "third_party/GMS-Feature-Matcher/include/gms_matcher.h"

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());

void GaussianKeyframe::setPose(
    const double qw,
    const double qx,
    const double qy,
    const double qz,
    const double tx,
    const double ty,
    const double tz)
{
    this->R_quaternion_.w() = qw;
    this->R_quaternion_.x() = qx;
    this->R_quaternion_.y() = qy;
    this->R_quaternion_.z() = qz;
    this->R_quaternion_.normalize();
    this->t_.x() = tx;
    this->t_.y() = ty;
    this->t_.z() = tz;

    this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);
    this->base_pose_ = tensor_utils::SE3f2TensorTransformation(this->Tcw_.cast<float>()).to(torch::kCUDA);

    this->set_pose_ = true;
}

void GaussianKeyframe::setPose(
    const Eigen::Quaterniond& q,
    const Eigen::Vector3d& t)
{
    this->R_quaternion_ = q;
    this->R_quaternion_.normalize();
    this->t_ = t;

    this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);
    this->base_pose_ = tensor_utils::SE3f2TensorTransformation(this->Tcw_.cast<float>()).to(torch::kCUDA);

    this->set_pose_ = true;
}

Sophus::SE3d GaussianKeyframe::getPose(bool with_global, bool with_local)
{
    torch::NoGradGuard no_grad;

    torch::Tensor pose;
    Sophus::SE3d se3;
    if(with_global && with_local)
        pose = this->local_delta_pose_.mm(
            this->local_delta_pose_init_.mm(this->global_delta_pose_.mm(this->base_pose_))
        );
    else if(with_global && !with_local)
        pose = this->global_delta_pose_.mm(this->base_pose_);
    else if(!with_global && with_local)
        pose = this->local_delta_pose_.mm(this->local_delta_pose_init_.mm(this->base_pose_));
    else
        // return this->Tcw_;
        pose = this->base_pose_;

    pose = pose.cpu();
    se3 = tensor_utils::TensorTransformation2SE3f(pose).cast<double>();
    return se3;
}

Sophus::SE3f GaussianKeyframe::getPosef(bool with_global, bool with_local)
{
    torch::NoGradGuard no_grad;

    torch::Tensor pose;
    Sophus::SE3f se3;
    if(with_global && with_local)
        pose = this->local_delta_pose_.mm(
            this->local_delta_pose_init_.mm(this->global_delta_pose_.mm(this->base_pose_))
        );
    else if(with_global && !with_local)
        pose = this->global_delta_pose_.mm(this->base_pose_);
    else if(!with_global && with_local)
        pose = this->local_delta_pose_.mm(this->local_delta_pose_init_.mm(this->base_pose_));
    else
        // return this->Tcw_.cast<float>();
        pose = this->base_pose_;
    pose = pose.cpu();
    se3 = tensor_utils::TensorTransformation2SE3f(pose);
    return se3;
}

void GaussianKeyframe::setGTPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz)
{
    // Eigen::Quaterniond gt_q(qw, qx, qy, qz);
    // Eigen::Vector3d gt_t(tx, ty, tz);
    // gt_q.normalize();

    // Apply 180° rotation around X-axis to convert Blender → OpenCV
    // Eigen::Matrix3d R_corr =
    //     Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
    // Eigen::Quaterniond q_corr = Eigen::Quaterniond(R_corr);

    // std::cout<<"[debug] M_PI\n "<<Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix()<<std::endl;
    // std::cout<<"[debug] -M_PI\n "<<Eigen::AngleAxisd(-M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix()<<std::endl;

    // // blender original
    // Sophus::SE3d T1 = Sophus::SE3d(gt_q, gt_t);
    // std::cout<<"[debug] GT T1\n "<<T1.matrix()<<std::endl;

    // // blender to orbslam inverse auto
    // Sophus::SE3d T2 = T1.inverse();
    // std::cout<<"[debug] GT T2\n "<<T2.matrix()<<std::endl;

    // // blender to orbslam inverse manual
    // Eigen::Matrix3d Rwc = T1.so3().matrix().transpose();
    // Eigen::Vector3d twc = -Rwc * T1.translation();
    // Sophus::SE3d T3 = Sophus::SE3d(Rwc, twc);
    // std::cout<<"[debug] GT T3\n "<<T3.matrix()<<std::endl;

    // // blender 180 rotation q
    // auto qq = q_corr*gt_q;
    // Sophus::SE3d T4 = Sophus::SE3d(qq, gt_t);
    // std::cout<<"[debug] GT T4\n "<<T4.matrix()<<std::endl;

    // // blender 180 rotation t
    // auto tt =  q_corr*gt_t;
    // Sophus::SE3d T5 = Sophus::SE3d(gt_q, tt);
    // std::cout<<"[debug] GT T5\n "<<T5.matrix()<<std::endl;

    // real
    // Eigen::Matrix3d axis_corr;
    // axis_corr << 1, 0, 0,
    //              0, -1, 0,
    //              0, 0, -1;
    // Eigen::Quaterniond axis_q_corr(axis_corr);
    // auto gt_q_corr = q_corr * gt_q;
    // gt_q_corr.normalize();
    // Eigen::Matrix3d gt_R_corr = gt_q_corr.toRotationMatrix();
    // gt_R_corr = axis_corr * gt_R_corr;
    // auto gt_t_corr = axis_corr * gt_t;
    // this->GT_Tcw_ = Sophus::SE3d(gt_R_corr, gt_t);
    // this->GT_Tcw_ = this->GT_Tcw_.inverse();

    // auto gt_q_corr = axis_q_corr * gt_q;
    // gt_q_corr.normalize();
    // gt_q_corr = q_corr * gt_q_corr;
    // this->GT_Tcw_ = Sophus::SE3d(gt_q_corr, gt_t);
    // this->GT_Tcw_ = this->GT_Tcw_.inverse();

    Eigen::Quaterniond gt_q(qw, qx, qy, qz);
    Eigen::Vector3d gt_t(tx, ty, tz);
    gt_q.normalize();

    // std::cout<<"[debug] GT_Tcw_ \n"<<GT_Tcw_.unit_quaternion()<<std::endl;
    this->GT_Tcw_ = Sophus::SE3d(gt_q, gt_t).inverse();

    // Eigen::Matrix3d Rwc = this->GT_Tcw_.so3().matrix().transpose();
    // Eigen::Vector3d twc = -Rwc * this->GT_Tcw_.translation();
    // this->GT_Tcw_ = Sophus::SE3d(Rwc, twc);
    
    this->set_gt_pose_ = true;
}

Sophus::SE3f GaussianKeyframe::getGTPosef(){
    return this->GT_Tcw_.cast<float>();
}

torch::Tensor GaussianKeyframe::getGTLRImg(bool use_cuda){
    if(use_cuda) return this->original_image_.to(torch::kCUDA);
    return this->original_image_;
}

torch::Tensor GaussianKeyframe::getGTLRDpt(bool use_cuda){
    if(!this->original_depth_.defined())
        this->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(this->img_auxiliary_undist_, torch::kCPU);
    if(use_cuda) return this->original_depth_.to(torch::kCUDA);
    return this->original_depth_;
}

torch::Tensor GaussianKeyframe::getGTLRDptMsk(bool use_cuda){
    if(!this->original_depth_mask_.defined())
        this->original_depth_mask_ = tensor_utils::cvMat2TorchTensor_Float32(this->depth_undist_valid_mask_, torch::kCPU);
    if(use_cuda) return this->original_depth_mask_.to(torch::kCUDA);
    return this->original_depth_mask_;
}

torch::Tensor GaussianKeyframe::getGTHRImg(float resize_ratio, bool use_cuda){
    if(!this->has_hr_fid_)
        throw std::runtime_error("[error][GaussianKeyframe::getGTHRImg] `has_hr_fid_` is false! Cannot get HR image.");

    if (resize_ratio > 1.0f-1e-5) 
        return use_cuda ? this->hr_image_undist_.to(torch::kCUDA) : this->hr_image_undist_;
    
    float r = std::max(resize_ratio, 0.1f);
    int new_width = int(floor(kf_params_->hr_width_*r));
    int new_height = int(floor(kf_params_->hr_height_*r));

    auto img = use_cuda ? this->hr_image_undist_.to(torch::kCUDA) : this->hr_image_undist_;

    if(resize_ratio == 1.f) return img;

    img = img.unsqueeze(0);
    auto img_resized = torch::nn::functional::interpolate(
        img,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({new_height, new_width}))
            .mode(torch::kNearest)
    ).squeeze(0);

    return img_resized;
}

torch::Tensor GaussianKeyframe::getGTHRImg(torch::Tensor& selection_indices, float resize_ratio, bool use_cuda){
    auto img = this->getGTHRImg(resize_ratio, use_cuda);

    float r = std::max(resize_ratio, 0.1f);
    int pixel_samples = this->kf_params_->hr_pixel_samples_;
    if (resize_ratio < 1.0f-1e-5) pixel_samples = int(pixel_samples * r);

    auto map_key = std::make_tuple(int(img.size(1)), int(img.size(2)), pixel_samples);
    auto has_cache = this->hr_selection_maps_.find(map_key);

    if(has_cache == this->hr_selection_maps_.end()){
        auto selection_map = general_utils::fast_pixel_selector(img, pixel_samples);
        auto indices = torch::where(selection_map);
        torch::Tensor y_coords = indices[0];
        torch::Tensor x_coords = indices[1];
        selection_indices = y_coords * img.size(2) + x_coords;

        this->hr_selection_maps_[map_key] = selection_indices.to(torch::kCPU).clone();

        if(use_cuda)
            selection_indices = selection_indices.to(torch::kCUDA);
        else
            selection_indices = selection_indices.to(torch::kCPU);

        // std::cout<<"[debug] cache hr selection map for keyframe "<<this->fid_<<" at size "<<img.size(1)<<"x"<<img.size(2)
        //     <<" with "<<pixel_samples<<" samples, total selected pixels: "<<selection_indices.size(0)<<std::endl;
    }
    else {
        selection_indices = has_cache->second.to(img.device());

        // std::cout<<"[debug] use cached hr selection map for keyframe "<<this->fid_<<" at size "<<img.size(1)<<"x"<<img.size(2)
        //     <<" with "<<pixel_samples<<" samples, total selected pixels: "<<selection_indices.size(0)<<std::endl;
    }

    return img;
}

void GaussianKeyframe::setGTLRDpt(cv::Mat& depth_img){
    this->img_auxiliary_undist_ = depth_img.clone();

    cv::Mat depth_nonzero_mask = (this->img_auxiliary_undist_ > 1e-5f);
    depth_nonzero_mask.convertTo(depth_nonzero_mask, CV_32F, 1.f/255.f);

    cv::Mat depth_variance_mask = general_utils::getDepthVarianceMask(
        this->img_auxiliary_undist_,
        kf_params_->lr_depth_variance_window_size_,
        kf_params_->lr_depth_variance_threshold_ratio_
    );

    this->depth_undist_valid_mask_ = depth_nonzero_mask.mul(depth_variance_mask);

    if(!this->kf_params_->lr_undistort_mask_.empty())
        this->depth_undist_valid_mask_ = this->depth_undist_valid_mask_.mul(this->kf_params_->lr_undistort_mask_);
}

void GaussianKeyframe::setGTLRImg(cv::Mat& color_img){
    this->img_undist_ = color_img.clone();

    cv::Mat mImGray; 
    cv::cvtColor(this->img_undist_, mImGray, cv::COLOR_BGR2GRAY);
    mImGray.convertTo(mImGray, CV_8U, 255.f);

    // this->orb_frame_lr_ = ORB_SLAM3::Frame(
    auto orb_frame_lr = ORB_SLAM3::Frame(
        mImGray,
        this->img_undist_,
        this->lr_timestamp_,
        this->kf_params_->orb_extractor_lr_,
        this->kf_params_->orb_vocabulary_,
        this->kf_params_->orb_camera_lr_,
        this->kf_params_->orb_camera_dist_,
        this->kf_params_->orb_bf_,
        this->kf_params_->orb_thdepth_
    );
    this->orb_keypoints_lr_ = orb_frame_lr.mvKeysUn;
    this->orb_descriptors_lr_ = orb_frame_lr.mDescriptors.clone();
    // std::cout<<"[debug::!!] orb_keypoints_lr_ "<<this->orb_keypoints_lr_.size()<<std::endl;
}

bool GaussianKeyframe::setGTHRImg(float align_time, std::vector<std::string>& hr_img_filenames){
    float est_hr_time = this->lr_timestamp_ + align_time;

    if(est_hr_time < 0.f){
        std::cout<<"[warning] estimated hr time < 0: "<<est_hr_time<<" for kf "<<this->fid_<<std::endl; 
        return false;
    }

    this->hr_timestamp_ = est_hr_time;
    this->hr_fid_ = round(this->hr_timestamp_ * this->kf_params_->hr_fps_);
    this->hr_undist_mat_ = general_utils::readRGB2cvMat(hr_img_filenames[this->hr_fid_], 
            true, kf_params_->hr_undistort_map1_, kf_params_->hr_undistort_map2_);
    this->hr_image_undist_ = tensor_utils::cvMat2TorchTensor_Float32(this->hr_undist_mat_, torch::kCPU);
    this->has_hr_fid_ = true;
    std::cout<<"[debug] hr size: "<<this->hr_undist_mat_.size()<<std::endl;

    cv::Mat mImGray;
    cv::Size new_hr_size(
        int(kf_params_->hr_width_ * (float(kf_params_->lr_height_)/float(kf_params_->hr_height_))),
        kf_params_->lr_height_
    );
    cv::Mat hr_resized;
    cv::Mat hr_undist_mat = this->hr_undist_mat_.clone();
    cv::resize(hr_undist_mat, hr_resized, new_hr_size);
    cv::cvtColor(hr_resized, mImGray, cv::COLOR_BGR2GRAY); 
    mImGray.convertTo(mImGray, CV_8U, 255.f);
    std::cout<<"[debug] hr original size: "<<hr_undist_mat.size()<<", resized size: "<<hr_resized.size()<<" original hr size: "<<this->hr_undist_mat_.size()<<std::endl;

    // this->orb_frame_hr_ = ORB_SLAM3::Frame(
    auto orb_frame_hr = ORB_SLAM3::Frame(
        mImGray,
        hr_resized,
        this->hr_timestamp_,
        this->kf_params_->orb_extractor_hr_,
        this->kf_params_->orb_vocabulary_,
        this->kf_params_->orb_camera_hr_,
        this->kf_params_->orb_camera_dist_,
        this->kf_params_->orb_bf_,
        this->kf_params_->orb_thdepth_
    );
    // this->orb_keypoints_hr_ = this->orb_frame_hr_.mvKeysUn;
    auto orb_hr_key = std::make_tuple(
        new_hr_size.width,
        new_hr_size.height,
        float(new_hr_size.height) / float(this->kf_params_->hr_height_)
    );
    this->orb_multisizes_hr_[orb_hr_key] = std::make_tuple(
        orb_frame_hr.mDescriptors.clone(),
        orb_frame_hr.mvKeysUn
    );

    if(kf_params_->debug_){
        std::cout<<"[GaussianKeyframe::setGTHRImg] fid_"<<this->fid_<<" lr timestamp_: "<<this->lr_timestamp_<<
            ", estimated hr timestamp_: "<<this->hr_timestamp_<<", matched hr_fid_: "<<this->hr_fid_<<std::endl;
        std::cout<<"[GaussianKeyframe::setGTHRImg] fid_"<<this->fid_<<" matched hr_fid_"<<this->hr_fid_<<
            "\nlr filename: "<<this->img_filename_<<"\nhr filename: "<<hr_img_filenames[this->hr_fid_]<<std::endl;
    }

    return true;
}

void GaussianKeyframe::setCameraParams(const Camera& camera)
{
    this->camera_id_ = camera.camera_id_;
    this->camera_model_id_ = camera.model_id_;
    this->image_height_ = camera.height_;
    this->image_width_ = camera.width_;

    this->num_gaus_pyramid_sub_levels_ = camera.num_gaus_pyramid_sub_levels_;
    this->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    this->gaus_pyramid_width_ = camera.gaus_pyramid_width_;

    this->intr_.resize(camera.params_.size());
    for (std::size_t i = 0; i < camera.params_.size(); ++i)
        this->intr_[i] = static_cast<float>(camera.params_[i]);

    switch (this->camera_model_id_)
    {
    case 1: // Pinhole
    {
        float focal_length_x = static_cast<float>(camera.params_[0]);
        float focal_length_y = static_cast<float>(camera.params_[1]);
        this->FoVx_ = graphics_utils::focal2fov(focal_length_x, camera.width_);
        this->FoVy_ = graphics_utils::focal2fov(focal_length_y, camera.height_);
        this->set_camera_ = true;
    }
    break;

    default:
    {
        throw std::runtime_error("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!");
    }
    break;
    }
}

void GaussianKeyframe::setPoints2D(const std::vector<Eigen::Vector2d>& points2D)
{
    this->points2D_.clear();
    auto num_points2D = points2D.size();
    this->points2D_.resize(num_points2D);
    for (point2D_idx_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
        points2D_[point2D_idx].xy_ = points2D[point2D_idx];
    }
}

void GaussianKeyframe::setPoint3DIdxForPoint2D(
    const point2D_idx_t point2D_idx,
    const point3D_id_t point3D_id)
{
    points2D_.at(point2D_idx).point3D_id_ = point3D_id;
}

void GaussianKeyframe::computeTransformTensors()
{
    if (this->set_pose_ && this->set_camera_) {
        // this->world_view_transform_ = tensor_utils::EigenMatrix2TorchTensor(
        //     this->getWorld2View2(this->trans_, this->scale_),
        //     torch::kCUDA
        // ).transpose(0, 1);
        this->base_pose_ = tensor_utils::EigenMatrix2TorchTensor(
            this->getWorld2View2(this->trans_, this->scale_), torch::kCUDA);
        this->world_view_transform_ = this->base_pose_.transpose(0, 1);
        // std::cout<<"[debug] fid_"<<this->fid_<<std::endl;
        // std::cout<<"[debug] world_view_transform_"<<this->world_view_transform_<<std::endl;

        if (!this->set_projection_matrix_) {
            // this->projection_matrix_ = this->getProjectionMatrix(
            //     this->znear_,
            //     this->zfar_,
            //     this->FoVx_,
            //     this->FoVy_,
            //     torch::kCUDA
            // ).transpose(0, 1);
            if(kf_params_->align_pose_){
                this->base_hr_proj_ = this->getProjectionMatrix(
                    this->znear_, this->zfar_,
                    kf_params_->hr_fovx_, kf_params_->hr_fovy_,
                    torch::kCUDA);
            }
            this->base_lr_proj_ = this->getProjectionMatrix(
                this->znear_, this->zfar_,
                this->FoVx_, this->FoVy_,
                torch::kCUDA);

            this->projection_matrix_ = this->base_lr_proj_.transpose(0, 1);
            this->set_projection_matrix_ = true;
        }

        // this->full_proj_transform_ = (this->world_view_transform_.unsqueeze(0).bmm(
        //     this->projection_matrix_.unsqueeze(0))).squeeze(0);
        this->full_proj_transform_ = this->world_view_transform_.mm(this->projection_matrix_);

        this->camera_center_ = this->world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});
    }
    else if (!this->set_pose_ && this->set_camera_) {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because POSE is not set!" << std::endl;
    }
    else if (!this->set_camera_) {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because CAMERA is not set!" << std::endl;
    }
    else {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because POSE and CAMERA are not set!" << std::endl;
    }
}

Eigen::Matrix4f
GaussianKeyframe::getWorld2View2(
    const Eigen::Vector3f& trans,
    float scale)
{
    Eigen::Matrix4f Rt;
    Rt.setZero();
    Eigen::Matrix3f R = this->R_quaternion_.toRotationMatrix().cast<float>();
    Rt.topLeftCorner<3, 3>() = R;
    Eigen::Vector3f t = this->t_.cast<float>();
    Rt.topRightCorner<3, 1>() = t;
    Rt(3, 3) = 1.0f;

    Eigen::Matrix4f C2W = Rt.inverse();
    Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
    cam_center += trans;
    cam_center *= scale;
    C2W.block<3, 1>(0, 3) = cam_center;
    Rt = C2W.inverse();
    return Rt;
}

torch::Tensor
GaussianKeyframe::getProjectionMatrix(
    float znear,
    float zfar,
    float fovX,
    float fovY,
    torch::DeviceType device_type)
{
    float tanHalfFovY = std::tan(fovY / 2);
    float tanHalfFovX = std::tan(fovX / 2);

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({4, 4}, torch::TensorOptions().device(device_type));

    float z_sign = 1.0f;

    P.index({0, 0}) = 2.0 * znear / (right - left);
    P.index({1, 1}) = 2.0 * znear / (top - bottom);
    P.index({0, 2}) = (right + left) / (right - left);
    P.index({1, 2}) = (top + bottom) / (top - bottom);
    P.index({3, 2}) = z_sign;
    P.index({2, 2}) = z_sign * zfar / (zfar - znear);
    P.index({2, 3}) = -(zfar * znear) / (zfar - znear);
    return P;
}

int GaussianKeyframe::getCurrentGausPyramidLevel()
{
    for (int i = 0; i < gaus_pyramid_times_of_use_.size(); ++i) {
        if (gaus_pyramid_times_of_use_[i]) {
            --gaus_pyramid_times_of_use_[i];
            return i;
        }
    }
    // If all sub levels has been used up
    return num_gaus_pyramid_sub_levels_;
}

// void GaussianKeyframe::undistortHR(cv::Mat& img, torch::DeviceType device_type){
//     if(!kf_params_->has_undistort_) return;
//     cv::Mat img_distort;
//     cv::remap(
//         img,
//         img_distort,
//         kf_params_->undistort_map1_,
//         kf_params_->undistort_map2_,
//         cv::InterpolationFlags::INTER_LINEAR
//     );
//     this->hr_image_undist_ = tensor_utils::cvMat2TorchTensor_Float32(img_distort, device_type);
// }

bool GaussianKeyframe::validLieAlgebraUpdate(
    bool has_thresh, 
    float theta_thresh, 
    float rho_thresh,
    bool normalize
){
    torch::NoGradGuard no_grad;

    bool has_nan = torch::isnan(this->theta_).any().item<bool>() && torch::isnan(this->rho_).any().item<bool>();
    bool has_inf = torch::isinf(this->rho_).any().item<bool>() && torch::isinf(this->theta_).any().item<bool>();

    if(has_nan || has_inf)
        return false;

    auto theta_norm = this->theta_.norm().item<float>();
    auto rho_norm = this->rho_.norm().item<float>();

    if(theta_norm < 1e-4f && rho_norm < 1e-4f)
        return false;

    bool valid = true;
    if(has_thresh){
        float theta_thr = theta_thresh>0.f ? theta_thresh : kf_params_->theta_lr_*10.f;
        float rho_thr = rho_thresh>0.f ? rho_thresh : kf_params_->rho_lr_*10.f;
        if(normalize){
            if(theta_norm>theta_thr)
                this->theta_.mul_(theta_thr / theta_norm);
            if(rho_norm>rho_thr)
                this->rho_.mul_(rho_thr / rho_norm);
        }
        else if(!(theta_norm<theta_thr && rho_norm<rho_thr))
            valid = false;
    }

    return valid;
}

bool GaussianKeyframe::updateBasePose(
    bool has_thresh, 
    float theta_thresh, 
    float rho_thresh
){
    /*
    torch::NoGradGuard no_grad;

    bool valid = this->validLieAlgebraUpdate(has_thresh, theta_thresh, rho_thresh);
    
    torch::Tensor delta_pose;
    if(kf_params_->align_pose_ && kf_params_->align_global_pose_ && valid){
        delta_pose = tensor_utils::se3_exp(this->theta_, this->rho_);

        if(!torch::isnan(delta_pose).any().item<bool>() && !torch::isinf(delta_pose).any().item<bool>()){
            this->base_pose_ = delta_pose.mm(this->base_pose_);
            this->Tcw_ = tensor_utils::TensorTransformation2SE3f(this->base_pose_).cast<double>();
            this->dense_init_transformations_ = this->base_pose_.inverse().mm(delta_pose.mm(this->base_pose_));
        }
        else valid = false;
    }

    // if(this->need_dense_init_optimization_ && valid)
    //     this->updateDenseInitJointState(delta_pose);

    this->theta_.zero_(); this->rho_.zero_();
    return valid;
    */
    
    torch::Tensor delta_pose;
    return this->updateBasePose(
        delta_pose,
        has_thresh, 
        theta_thresh, 
        rho_thresh
    );
}

bool GaussianKeyframe::updateBasePose(
    torch::Tensor& delta_pose,
    bool has_thresh, 
    float theta_thresh, 
    float rho_thresh
){
    torch::NoGradGuard no_grad;

    bool valid = this->validLieAlgebraUpdate(has_thresh, theta_thresh, rho_thresh);
    
    if(kf_params_->align_pose_ && kf_params_->align_global_pose_ && valid){
        delta_pose = tensor_utils::se3_exp(this->theta_, this->rho_);

        if(!torch::isnan(delta_pose).any().item<bool>() && !torch::isinf(delta_pose).any().item<bool>()){
            this->base_pose_ = delta_pose.mm(this->base_pose_);
            this->Tcw_ = tensor_utils::TensorTransformation2SE3f(this->base_pose_).cast<double>();
            this->dense_init_transformations_ = this->base_pose_.inverse().mm(delta_pose.mm(this->base_pose_));
        }
        else valid = false;
    }

    if(!valid) delta_pose = torch::eye(4, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    this->theta_.zero_(); this->rho_.zero_();
    return valid;
}

bool GaussianKeyframe::updateGlobalDeltaPose(
    bool has_thresh, 
    float theta_thresh, 
    float rho_thresh
){
    torch::NoGradGuard no_grad;

    bool valid = this->validLieAlgebraUpdate(has_thresh, theta_thresh, rho_thresh);

    if(kf_params_->align_pose_ && kf_params_->align_global_pose_ && valid){
        auto delta_pose = tensor_utils::se3_exp(this->theta_, this->rho_);
        this->global_delta_pose_ = delta_pose.mm(this->global_delta_pose_);
    }

    this->theta_.zero_(); this->rho_.zero_();
    return valid;
}

bool GaussianKeyframe::updateLocalDeltaPose(
    bool has_thresh, 
    float theta_thresh, 
    float rho_thresh
){
    /*
    torch::NoGradGuard no_grad;

    bool valid = this->validLieAlgebraUpdate(has_thresh, theta_thresh, rho_thresh);

    if(kf_params_->align_pose_ && kf_params_->align_local_pose_ && valid){
        auto delta_pose = tensor_utils::se3_exp(this->theta_, this->rho_);
        this->local_delta_pose_ = delta_pose.mm(this->local_delta_pose_);
    }

    this->theta_.zero_(); this->rho_.zero_();
    return valid;*/

    torch::Tensor delta_pose;

    return this->updateLocalDeltaPose(
        delta_pose,
        has_thresh, 
        theta_thresh, 
        rho_thresh
    );
}

bool GaussianKeyframe::updateLocalDeltaPose(
    torch::Tensor& delta_pose,
    bool has_thresh, 
    float theta_thresh, 
    float rho_thresh
){
    torch::NoGradGuard no_grad;

    bool valid = this->validLieAlgebraUpdate(has_thresh, theta_thresh, rho_thresh);

    if(kf_params_->align_pose_ && kf_params_->align_local_pose_ && valid){
        delta_pose = tensor_utils::se3_exp(this->theta_, this->rho_);
        this->local_delta_pose_ = delta_pose.mm(this->local_delta_pose_);
    }

    this->theta_.zero_(); this->rho_.zero_();
    return valid;
}

torch::Tensor GaussianKeyframe::getBasePose(){
    return this->base_pose_;
}

torch::Tensor GaussianKeyframe::getGlobalDeltaPose(){
    return this->global_delta_pose_;
}

torch::Tensor GaussianKeyframe::getLocalDeltaPose(){
    return this->local_delta_pose_;
}

torch::Tensor GaussianKeyframe::getFullDeltaPose(){
    return this->local_delta_pose_.mm(
            this->local_delta_pose_init_.mm(this->global_delta_pose_)
        );
}

void GaussianKeyframe::setBasePose(torch::Tensor pose){
    this->base_pose_ = pose.clone();
}


void GaussianKeyframe::setGlobalDeltaPose(torch::Tensor pose){
    this->global_delta_pose_ = pose.clone();
}

void GaussianKeyframe::setLocalDeltaPose(torch::Tensor pose){
    this->local_delta_pose_ = pose.clone();
}

void GaussianKeyframe::setLocalDeltaPoseInit(torch::Tensor pose){
    this->local_delta_pose_init_ = pose.clone();
}

void GaussianKeyframe::updateRenderMatrix(bool is_hr){
    if(!kf_params_->align_pose_) return;

    if (!this->set_projection_matrix_){
        this->base_hr_proj_ = this->getProjectionMatrix(
            this->znear_, this->zfar_,
            kf_params_->hr_fovx_, kf_params_->hr_fovy_,
            torch::kCUDA);
            
        this->base_lr_proj_ = this->getProjectionMatrix(
            this->znear_, this->zfar_,
            this->FoVx_, this->FoVy_,
            torch::kCUDA);
        this->set_projection_matrix_ = true;
    }
    
    if(is_hr)
        this->world_view_transform_ = this->local_delta_pose_.mm(
                this->local_delta_pose_init_.mm(this->global_delta_pose_.mm(this->base_pose_))
            ).transpose(0, 1);
    else
        this->world_view_transform_ = this->base_pose_.transpose(0, 1);

    if(is_hr)
        this->projection_matrix_ = this->base_hr_proj_.transpose(0, 1);
    else
        this->projection_matrix_ = this->base_lr_proj_.transpose(0, 1);

    this->full_proj_transform_ = this->world_view_transform_.mm(this->projection_matrix_);

    this->camera_center_ = this->world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});
}

void GaussianKeyframe::stepOptimizer(bool step_pose, bool step_exposure){
    if(step_pose)
        this->pose_optimizer_->step();
    
    if(step_exposure)
        this->exposure_optimizer_->step();
}

void GaussianKeyframe::zeroOptimizerGrad(bool zero_pose, bool zero_exposure){
    if(zero_pose)
        this->pose_optimizer_->zero_grad(true);
    
    if(zero_exposure)
        this->exposure_optimizer_->zero_grad(true);
}

void GaussianKeyframe::resetOptimizer(bool with_lr, float theta_lr, float rho_lr){
    float set_theta_lr, set_rho_lr;
    if(with_lr){
        set_theta_lr = theta_lr > 0.f ? theta_lr : kf_params_->theta_lr_;
        set_rho_lr = rho_lr > 0.f ? rho_lr : kf_params_->rho_lr_;
    } else {
        set_theta_lr = kf_params_->theta_lr_;
        set_rho_lr = kf_params_->rho_lr_;
    }

    auto options_rot = torch::optim::AdamOptions(set_theta_lr).betas({0.9, 0.999});
    auto options_trans = torch::optim::AdamOptions(set_rho_lr).betas({0.9, 0.999});

    std::vector<torch::optim::OptimizerParamGroup> param_groups;

    param_groups.emplace_back(
        std::vector<torch::Tensor>{this->theta_},
        std::make_unique<torch::optim::AdamOptions>(options_rot)
    );
    param_groups.emplace_back(
        std::vector<torch::Tensor>{this->rho_},
        std::make_unique<torch::optim::AdamOptions>(options_trans)
    );

    this->pose_optimizer_ = std::make_unique<torch::optim::Adam>(param_groups);
}

void GaussianKeyframe::updateOptimizer(float theta_lr, float rho_lr){
    auto& group_theta = this->pose_optimizer_->param_groups()[0];
    auto& group_rho   = this->pose_optimizer_->param_groups()[1];

    auto& theta_opts = static_cast<torch::optim::AdamOptions&>(group_theta.options());
    auto& rho_opts   = static_cast<torch::optim::AdamOptions&>(group_rho.options());

    theta_opts.lr(theta_lr);
    rho_opts.lr(rho_lr);
}

void GaussianKeyframe::updateOptimizer(float decay_ratio){
    auto& group_theta = this->pose_optimizer_->param_groups()[0];
    auto& group_rho   = this->pose_optimizer_->param_groups()[1];

    auto& theta_opts = static_cast<torch::optim::AdamOptions&>(group_theta.options());
    auto& rho_opts   = static_cast<torch::optim::AdamOptions&>(group_rho.options());

    theta_opts.lr(theta_opts.lr() * decay_ratio);
    rho_opts.lr(rho_opts.lr() * decay_ratio);
}

void GaussianKeyframe::resetFullExposure(){
    this->exposure_a_ = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));
    this->exposure_b_ = torch::zeros({1}, torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA));

    std::vector<torch::optim::OptimizerParamGroup> exposure_param_groups;
    auto options_exposure_a = torch::optim::AdamOptions(kf_params_->exposure_a_lr_).betas({0.9, 0.999});
    auto options_exposure_b = torch::optim::AdamOptions(kf_params_->exposure_b_lr_).betas({0.9, 0.999});

    exposure_param_groups.push_back( torch::optim::OptimizerParamGroup(
        {this->exposure_a_}, std::make_unique<torch::optim::AdamOptions>(options_exposure_a)
    ));
    exposure_param_groups.push_back( torch::optim::OptimizerParamGroup(
        {this->exposure_b_}, std::make_unique<torch::optim::AdamOptions>(options_exposure_b)
    ));

    this->exposure_optimizer_ = std::make_unique<torch::optim::Adam>(exposure_param_groups);
}

void GaussianKeyframe::setDenseInitJointOptimization(bool flag){
    if(this->need_dense_init_optimization_ == flag)
        return;
    
    if(flag){
        this->dense_init_xyz_ = this->dense_init_xyz_.to(torch::kCUDA);
        this->dense_init_rgb_ = this->dense_init_rgb_.to(torch::kCUDA);
        
        tensor_utils::initGaussianOpacity(this->dense_init_xyz_, this->dense_init_opcity_, kf_params_->lr_local_init_opacity_);
        tensor_utils::initGaussianRotation(this->dense_init_xyz_, this->dense_init_rotation_);
        tensor_utils::initGaussianScaling(this->dense_init_xyz_, this->dense_init_scale_);
        tensor_utils::initGaussianFeatures(this->dense_init_rgb_, this->dense_init_feat_dc_, this->dense_init_feat_rest_);

        this->need_dense_init_optimization_ = true;
    }
    else {
        this->dense_init_xyz_ = torch::Tensor();
        this->dense_init_rgb_ = torch::Tensor();
        this->dense_init_opcity_ = torch::Tensor();
        this->dense_init_scale_ = torch::Tensor();
        this->dense_init_rotation_ = torch::Tensor();
        this->dense_init_feat_dc_ = torch::Tensor();
        this->dense_init_feat_rest_ = torch::Tensor();

        this->need_dense_init_optimization_ = false;
    }
}

void GaussianKeyframe::updateDenseInitJointState(torch::Tensor delta_pose){
    auto xyz_homo = torch::cat({this->dense_init_xyz_, torch::ones({this->dense_init_xyz_.size(0), 1}, this->dense_init_xyz_.options())}, 1);
    auto whole_transformation = this->base_pose_.inverse().mm(delta_pose.mm(this->base_pose_));
    auto transformed_xyz_homo = whole_transformation.mm(xyz_homo.transpose(0, 1));
    this->dense_init_xyz_ = transformed_xyz_homo.transpose(0, 1).index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}).contiguous();
}

void GaussianKeyframe::updateDenseInitJointGaussians(torch::Tensor& xyz, torch::Tensor indices){
    torch::NoGradGuard no_grad;

    auto kf_indices = torch::where(indices == int(this->fid_))[0];
    auto kf_xyz = xyz.index_select(0, kf_indices);
    auto kf_homo = torch::cat({kf_xyz, torch::ones({kf_xyz.size(0), 1}, kf_xyz.options())}, 1);
    auto transformed_kf_homo = this->dense_init_transformations_.mm(kf_homo.transpose(0,1));
    transformed_kf_homo = transformed_kf_homo.transpose(0, 1).index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}).contiguous();

    xyz.index_copy_(0, kf_indices, transformed_kf_homo);
}

int GaussianKeyframe::matchInitHROrbGMS(){
    const auto& kp_lr = this->orb_keypoints_lr_;
    const auto& descr_lr = this->orb_descriptors_lr_;

    cv::Size size_lr(this->kf_params_->lr_width_, this->kf_params_->lr_height_);
    cv::Size size_hr(
        int(kf_params_->hr_width_ * (float(kf_params_->lr_height_)/float(kf_params_->hr_height_))),
        kf_params_->lr_height_
    );
    float hr_ratio = float(size_hr.height) / float(this->kf_params_->hr_height_);

    auto orb_hr_key = std::make_tuple(
        size_hr.width,
        size_hr.height,
        hr_ratio
    );
    auto orb_hr_it = this->orb_multisizes_hr_.find(orb_hr_key);
    if(orb_hr_it == this->orb_multisizes_hr_.end())
        throw std::runtime_error("[GaussianKeyframe::matchInitHROrbGMS] Cannot find matched hr orb features for keyframe "+std::to_string(this->fid_));
    const auto& descr_hr = std::get<0>(orb_hr_it->second);
    const auto& kp_hr = std::get<1>(orb_hr_it->second);

    std::vector<int> vnMatches_lr2hr(kp_lr.size(), -1);
    std::vector<int> vnMatches_hr2lr(kp_hr.size(), -1);
    std::vector<bool> vbInliers;

    if(kp_lr.empty()||kp_hr.empty()||descr_lr.empty()||descr_hr.empty())
        return 0;

    // Brute-force, Hammin
    std::vector<cv::DMatch> matches_all;
    cv::BFMatcher matcher(NORM_HAMMING);
    matcher.match(descr_lr, descr_hr, matches_all);

    if(kf_params_->debug_){
        std::vector<cv::DMatch> matches_print;
        matches_print = std::vector<cv::DMatch>(
            matches_all.begin(), 
            matches_all.begin() + std::min<int>(matches_all.size(), 100)
        );
        cv::Mat img_matches;
        cv::Mat lr_print, hr_print;
        this->img_undist_.convertTo(lr_print, CV_8UC3, 255.f);
        hr_print = this->hr_undist_mat_.clone();
        cv::resize(hr_print, hr_print, size_hr);
        hr_print.convertTo(hr_print, CV_8UC3, 255.f);
        cv::drawMatches(
            lr_print, kp_lr,
            hr_print, kp_hr,
            matches_print, //matches_print,
            img_matches
        );
        cv::cvtColor(img_matches, img_matches, cv::COLOR_BGR2RGB);
        cv::cvtColor(lr_print, lr_print, cv::COLOR_BGR2RGB);
        cv::cvtColor(hr_print, hr_print, cv::COLOR_BGR2RGB);
        cv::drawKeypoints(lr_print, kp_lr, lr_print, cv::Scalar(0,0,255));
        cv::drawKeypoints(hr_print, kp_hr, hr_print, cv::Scalar(0,0,255));

        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((kf_params_->result_dir_ / kf_params_->debug_dir_ / "orb_matches"));
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "orb_matches" /
            (std::to_string(this->fid_)+".jpg"), img_matches);
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "orb_matches" /
            (std::to_string(this->fid_)+"_lr.jpg"), lr_print);
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "orb_matches" /
            (std::to_string(this->fid_)+"_hr.jpg"), hr_print);
    }

    if (matches_all.empty()) return 0; 

    gms_matcher gms(kp_lr, size_lr, kp_hr, size_hr, matches_all);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);

    if (num_inliers <= 0) return 0;

    this->orb_matches_lr2hr_ = vnMatches_lr2hr;
    this->orb_matches_hr2lr_ = vnMatches_hr2lr;
    int nGoodMatches = general_utils::gms_matcher_selector(
        vbInliers, matches_all,
        kp_lr, kp_hr,
        this->orb_matches_lr2hr_, this->orb_matches_hr2lr_
    );

    if(kf_params_->debug_){
        // std::cout << "[debug] orb_matches_lr2hr_: ";
        // for (const auto& match : this->orb_matches_lr2hr_) {
        //     std::cout << match << " ";
        // }
        // std::cout << std::endl;
        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < this->orb_matches_lr2hr_.size(); ++i) {
            if (this->orb_matches_lr2hr_[i] >= 0) {
                good_matches.emplace_back(i, this->orb_matches_lr2hr_[i], 0);
            }
        }

        cv::Mat img_good_matches;
        cv::Mat lr_print, hr_print;
        this->img_undist_.convertTo(lr_print, CV_8UC3, 255.f);
        hr_print = this->hr_undist_mat_.clone();
        cv::resize(hr_print, hr_print, size_hr);
        hr_print.convertTo(hr_print, CV_8UC3, 255.f);
        cv::drawMatches(
            lr_print, kp_lr,
            hr_print, kp_hr,
            good_matches, img_good_matches
        );
        cv::cvtColor(img_good_matches, img_good_matches, cv::COLOR_BGR2RGB);

        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "orb_matches" /
            (std::to_string(this->fid_)+"_gms.jpg"), img_good_matches);
    }

    return nGoodMatches;
}

int GaussianKeyframe::matchHROrbGMS( // render (rgbd) -> gt (rgb)
    torch::Tensor render_hr_tensor, float hr_resize_ratio,
    std::vector<cv::KeyPoint>& orb_keypoints_render_hr,
    std::vector<cv::KeyPoint>& orb_keypoints_gt_hr,
    std::vector<int>& vnMatches_render2gt,
    std::vector<int>& vnMatches_gt2render
){
    auto render_hr_cv = tensor_utils::torchTensor2CvMat_Float32(render_hr_tensor);
    render_hr_cv.convertTo(render_hr_cv, CV_8UC3, 255.f);

    // Compute ORB features for rendered hr image
    cv::Mat mImGray;
    cv::cvtColor(render_hr_cv, mImGray, cv::COLOR_BGR2GRAY);
    auto orb_frame_render_hr = ORB_SLAM3::Frame(
        mImGray,
        render_hr_cv,
        this->hr_timestamp_,
        this->kf_params_->orb_extractor_hr_,
        this->kf_params_->orb_vocabulary_,
        this->kf_params_->orb_camera_hr_,
        this->kf_params_->orb_camera_dist_,
        this->kf_params_->orb_bf_,
        this->kf_params_->orb_thdepth_
    );
    orb_keypoints_render_hr = orb_frame_render_hr.mvKeysUn;
    auto& orb_descriptors_render_hr = orb_frame_render_hr.mDescriptors;

    // Check and compute ORB features for gt hr image at the same size
    cv::Mat orb_descriptors_gt_hr;
    // std::vector<cv::KeyPoint> orb_keypoints_gt_hr;
    cv::Size size_render_hr(
        int(kf_params_->hr_width_ * hr_resize_ratio),
        int(kf_params_->hr_height_ * hr_resize_ratio)
    );
    auto orb_hr_key = std::make_tuple(
        size_render_hr.width,
        size_render_hr.height,
        hr_resize_ratio
    );
    auto orb_hr_it = this->orb_multisizes_hr_.find(orb_hr_key);
    if(orb_hr_it == this->orb_multisizes_hr_.end()){
        cv::Mat mImGray_gt, hr_resized;
        cv::Mat hr_undist_mat = this->hr_undist_mat_.clone();
        // std::cout<<"[debug] hr shape "<<this->hr_undist_mat_.size()<<" size "<<size_render_hr.width<<" x "<<size_render_hr.height<<std::endl;
        cv::resize(hr_undist_mat, hr_resized, size_render_hr);
        cv::cvtColor(hr_resized, mImGray_gt, cv::COLOR_BGR2GRAY);
        mImGray_gt.convertTo(mImGray_gt, CV_8U, 255.f);
        auto orb_frame_gt_hr = ORB_SLAM3::Frame(
            mImGray_gt,
            hr_resized,
            this->hr_timestamp_,
            this->kf_params_->orb_extractor_hr_,
            this->kf_params_->orb_vocabulary_,
            this->kf_params_->orb_camera_hr_,
            this->kf_params_->orb_camera_dist_,
            this->kf_params_->orb_bf_,
            this->kf_params_->orb_thdepth_
        );
        orb_descriptors_gt_hr = orb_frame_gt_hr.mDescriptors.clone();
        orb_keypoints_gt_hr = orb_frame_gt_hr.mvKeysUn;
        this->orb_multisizes_hr_[orb_hr_key] = std::make_tuple(
            orb_descriptors_gt_hr,
            orb_keypoints_gt_hr
        );
    }
    else{
        orb_descriptors_gt_hr = std::get<0>(orb_hr_it->second);
        orb_keypoints_gt_hr = std::get<1>(orb_hr_it->second);
    }

    if(orb_keypoints_render_hr.empty()||orb_keypoints_gt_hr.empty()) return 0;

    // std::vector<int> vnMatches_render2gt(orb_keypoints_render_hr.size(), -1);
    // std::vector<int> vnMatches_gt2render(orb_keypoints_gt_hr.size(), -1);
    vnMatches_render2gt = std::vector<int>(orb_keypoints_render_hr.size(), -1);
    vnMatches_gt2render = std::vector<int>(orb_keypoints_gt_hr.size(), -1);

    // Brute-force, Hammin
    std::vector<cv::DMatch> matches_all;
    cv::BFMatcher matcher(NORM_HAMMING);
    matcher.match(orb_descriptors_render_hr, orb_descriptors_gt_hr, matches_all);
    if (matches_all.empty()) return 0;

    std::vector<bool> vbInliers;
    gms_matcher gms(orb_keypoints_render_hr, size_render_hr, orb_keypoints_gt_hr, size_render_hr, matches_all);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    if (num_inliers <= 0) return 0;

    int nGoodMatches = general_utils::gms_matcher_selector(
        vbInliers, matches_all,
        orb_keypoints_render_hr, orb_keypoints_gt_hr,
        vnMatches_render2gt, vnMatches_gt2render
    );

    if(kf_params_->debug_){
        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < vnMatches_render2gt.size(); ++i) {
            if (vnMatches_render2gt[i] >= 0) {
                good_matches.emplace_back(i, vnMatches_render2gt[i], 0);
            }
        }

        cv::Mat img_good_matches, gt_hr_cv;
        gt_hr_cv = this->hr_undist_mat_.clone();
        gt_hr_cv.convertTo(gt_hr_cv, CV_8UC3, 255.f);
        cv::resize(gt_hr_cv, gt_hr_cv, size_render_hr);
        cv::cvtColor(render_hr_cv, render_hr_cv, cv::COLOR_BGR2RGB);
        cv::drawMatches(
            render_hr_cv, orb_keypoints_render_hr,
            gt_hr_cv, orb_keypoints_gt_hr,
            good_matches, img_good_matches
        );
        cv::cvtColor(img_good_matches, img_good_matches, cv::COLOR_BGR2RGB);

        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((kf_params_->result_dir_ / kf_params_->debug_dir_ / "hr_local_matches"));
        // std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count() % 10000);
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "hr_local_matches" /
            (std::to_string(this->fid_) + ".jpg"), img_good_matches);
        // std::cout<<"[debug] fid_"<<this->fid_<<" timestamp_"<<timestamp<<std::endl;
    }

    return nGoodMatches;
}

int GaussianKeyframe::matchLROrbGMS( // gt (rgbd) -> render (rgb)
    torch::Tensor render_lr_tensor, 
    std::vector<cv::KeyPoint>& orb_keypoints_render_lr,
    std::vector<cv::KeyPoint>& orb_keypoints_gt_lr,
    std::vector<int>& vnMatches_render2gt,
    std::vector<int>& vnMatches_gt2render
){
    auto render_lr_cv = tensor_utils::torchTensor2CvMat_Float32(render_lr_tensor);
    render_lr_cv.convertTo(render_lr_cv, CV_8UC3, 255.f);

    // Compute ORB features for rendered lr image
    cv::Mat mImGray;
    cv::cvtColor(render_lr_cv, mImGray, cv::COLOR_BGR2GRAY);
    auto orb_frame_render_lr = ORB_SLAM3::Frame(
        mImGray,
        render_lr_cv,
        this->lr_timestamp_,
        this->kf_params_->orb_extractor_lr_,
        this->kf_params_->orb_vocabulary_,
        this->kf_params_->orb_camera_lr_,
        this->kf_params_->orb_camera_dist_,
        this->kf_params_->orb_bf_,
        this->kf_params_->orb_thdepth_
    );
    orb_keypoints_render_lr = orb_frame_render_lr.mvKeysUn;
    auto& orb_descriptors_render_lr = orb_frame_render_lr.mDescriptors;

    // Use precomputed ORB features for gt lr image
    const auto& orb_descriptors_gt_lr = this->orb_descriptors_lr_;
    orb_keypoints_gt_lr = this->orb_keypoints_lr_;
    
    // std::cout<<"[debug::!!] orb_keypoints_lr_ "<<this->orb_keypoints_lr_.size()<<std::endl;
    // std::cout<<"[debug::!!] orb_keypoints_gt_lr "<<orb_keypoints_gt_lr.size()<<std::endl;
    // std::cout<<"[GaussianKeyframe::matchLROrbGMS] fid_"<<this->fid_<<" orb_keypoints_render_lr size "<<orb_keypoints_render_lr.size()
    //     <<" orb_keypoints_gt_lr size "<<orb_keypoints_gt_lr.size()<<std::endl;

    if(orb_keypoints_render_lr.empty() || orb_keypoints_gt_lr.empty()) return 0;

    vnMatches_render2gt = std::vector<int>(orb_keypoints_render_lr.size(), -1);
    vnMatches_gt2render = std::vector<int>(orb_keypoints_gt_lr.size(), -1);

    // Brute-force, Hammin
    std::vector<cv::DMatch> matches_all;
    cv::BFMatcher matcher(NORM_HAMMING);
    matcher.match(orb_descriptors_gt_lr, orb_descriptors_render_lr, matches_all); // gt (3d) -> render
    // matcher.match(orb_descriptors_render_lr, orb_descriptors_gt_lr, matches_all); // render (3d) -> gt

    if(kf_params_->debug_){
        std::vector<cv::DMatch> matches_print;
        matches_print = std::vector<cv::DMatch>(
            matches_all.begin(), 
            matches_all.begin() + std::min<int>(matches_all.size(), 100)
        );
        cv::Mat img_matches;
        cv::Mat lr_cv;
        this->img_undist_.convertTo(lr_cv, CV_8UC3, 255.f);
        cv::drawMatches(
            lr_cv, orb_keypoints_gt_lr,
            render_lr_cv, orb_keypoints_render_lr,
            matches_print, //matches_print,
            img_matches
        );
        cv::cvtColor(img_matches, img_matches, cv::COLOR_BGR2RGB);
        cv::cvtColor(lr_cv, lr_cv, cv::COLOR_BGR2RGB);
        cv::cvtColor(render_lr_cv, render_lr_cv, cv::COLOR_BGR2RGB);
        cv::drawKeypoints(lr_cv, orb_keypoints_gt_lr, lr_cv, cv::Scalar(0,0,255));
        cv::drawKeypoints(render_lr_cv, orb_keypoints_render_lr, render_lr_cv, cv::Scalar(0,0,255));

        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((kf_params_->result_dir_ / kf_params_->debug_dir_ / "lr_local_matches"));
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "lr_local_matches" /
            (std::to_string(this->fid_)+".jpg"), img_matches);
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "lr_local_matches" /
            (std::to_string(this->fid_)+"_gt_lr.jpg"), lr_cv);
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "lr_local_matches" /
            (std::to_string(this->fid_)+"_render_lr.jpg"), render_lr_cv);
    }

    if (matches_all.empty()) return 0;

    std::vector<bool> vbInliers;
    cv::Size size_lr(this->kf_params_->lr_width_, this->kf_params_->lr_height_);
    gms_matcher gms(orb_keypoints_gt_lr, size_lr, orb_keypoints_render_lr, size_lr, matches_all);
    // gms_matcher gms(orb_keypoints_render_lr, size_lr, orb_keypoints_gt_lr, size_lr, matches_all);

    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    if (num_inliers <= 0) return 0;

    int nGoodMatches = general_utils::gms_matcher_selector(
        vbInliers, matches_all,
        orb_keypoints_gt_lr, orb_keypoints_render_lr,
        vnMatches_gt2render, vnMatches_render2gt
    );
    // int nGoodMatches = general_utils::gms_matcher_selector(
    //     vbInliers, matches_all,
    //     orb_keypoints_render_lr, orb_keypoints_gt_lr,
    //     vnMatches_render2gt, vnMatches_gt2render
    // );

    if(kf_params_->debug_){
        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < vnMatches_gt2render.size(); ++i) {
            if (vnMatches_gt2render[i] >= 0) {
                good_matches.emplace_back(i, vnMatches_gt2render[i], 0);
            }
        }

        cv::Mat img_good_matches, gt_lr_cv;
        gt_lr_cv = this->img_undist_.clone();
        gt_lr_cv.convertTo(gt_lr_cv, CV_8UC3, 255.f);
        cv::cvtColor(render_lr_cv, render_lr_cv, cv::COLOR_BGR2RGB);
        cv::drawMatches(
            gt_lr_cv, orb_keypoints_gt_lr,
            render_lr_cv, orb_keypoints_render_lr,
            // gt_lr_cv, orb_keypoints_gt_lr,
            good_matches, img_good_matches
        );
        cv::cvtColor(img_good_matches, img_good_matches, cv::COLOR_BGR2RGB);

        // CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((kf_params_->result_dir_ / kf_params_->debug_dir_ / "orb_matches"));
        // std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count() % 10000);
        cv::imwrite(kf_params_->result_dir_ / kf_params_->debug_dir_ / "lr_local_matches" /
            (std::to_string(this->fid_) + "_gms.jpg"), img_good_matches);
        // std::cout<<"[debug] fid_"<<this->fid_<<" timestamp_"<<timestamp<<std::endl;
    }

    return nGoodMatches;
}
