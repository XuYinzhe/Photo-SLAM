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

#include "include/gaussian_mapper.h"

GaussianMapper::GaussianMapper(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        throw std::runtime_error("Please run on devices with cuda!");
        // std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        // device_type_ = torch::kCPU;
        // model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    this->kf_params_.result_dir_ = result_dir_;
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::cout << "[Gaussian Mapper]Pose optimization: "<<kf_params_.align_pose_<< std::endl;
    std::cout << "[Gaussian Mapper]Align pose between HR and LR: "<<kf_params_.render_aligned_<< std::endl;
    std::cout << "[Gaussian Mapper]Please set `KeyframeOptimization.render_aligned: 1` for HR and LR together."<< std::endl;
    std::cout << "[Gaussian Mapper]Please set `KeyframeOptimization.render_aligned: 0` for mono and rgbd."<< std::endl;

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModel>(model_params_, this->kf_params_.debug_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // Mode
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Sensors
    switch (pSLAM->getSensorType())
    {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // Cameras
    // TODO: not only monocular
    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    std::cout << "[Gaussian Mapper]Number of cameras in SLAM: " << vpCameras.size() << std::endl;
    for (auto& SLAM_camera : vpCameras) {
        Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            // camera.width_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.width
            //                                              : graphics_utils::roundToIntegerMultipleOf16(
            //                                                    undistort_params.old_size_.width);
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

            // camera.height_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.height
            //                                               : graphics_utils::roundToIntegerMultipleOf16(
            //                                                     undistort_params.old_size_.height);
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            this->rendered_depthmap_factor_ = pSLAM_->getSettings()->depthMapFactor();
            kf_params_.lr_fx_ = camera.params_[0];
            kf_params_.lr_fy_ = camera.params_[1];
            kf_params_.lr_cx_ = camera.params_[2];
            kf_params_.lr_cy_ = camera.params_[3];
            kf_params_.lr_width_ = camera.width_;
            kf_params_.lr_height_ = camera.height_;
            kf_params_.lr_fps_ = pSLAM_->getSettings()->fps();

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            std::vector<cv::Mat> undistort_mask_channels;
            cv::split(camera.undistort_mask, undistort_mask_channels);
            kf_params_.lr_undistort_mask_ = undistort_mask_channels[0];
            kf_params_.lr_undistort_mask_ = (kf_params_.lr_undistort_mask_>(1.f-1e-4f));
            kf_params_.lr_undistort_mask_.convertTo(kf_params_.lr_undistort_mask_, CV_32FC1, 1.0f/255.0f);
            std::cout << "[Gaussian Mapper]LR undistort mask sum is set: kf_params_.lr_undistort_mask_" << std::endl;

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // hr camera calibrate
    if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
        std::cout << "[Gaussian Mapper]Setting HR camera intrinsics and undistort map..." << std::endl;

        cv::Mat hr_K = (
            cv::Mat_<float>(3, 3)
            <<  kf_params_.hr_fx_, 0.f, kf_params_.hr_cx_,
                0.f, kf_params_.hr_fy_, kf_params_.hr_cy_,
                0.f, 0.f, 1.f);
        cv::Mat hr_K2 = hr_K.clone();

        cv::Mat hr_dist_coeff = (cv::Mat_<float>(1, 4) << 
            kf_params_.hr_k1_, kf_params_.hr_k2_, kf_params_.hr_p1_, kf_params_.hr_p2_);

        cv::initUndistortRectifyMap(
            hr_K,
            hr_dist_coeff,
            cv::Mat::eye(3, 3, CV_32F),
            hr_K2,
            cv::Size(kf_params_.hr_width_, kf_params_.hr_height_),
            CV_32F,
            kf_params_.hr_undistort_map1_,
            kf_params_.hr_undistort_map2_);

        kf_params_.has_undistort_ = true;

        // pyramid
        kf_params_.gaus_pyramid_hr_width_.resize(num_gaus_pyramid_sub_levels_);
        kf_params_.gaus_pyramid_hr_height_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            kf_params_.gaus_pyramid_hr_width_[l] = kf_params_.hr_width_ * this->kf_gaus_pyramid_factors_[l];
            kf_params_.gaus_pyramid_hr_height_[l] = kf_params_.hr_height_ * this->kf_gaus_pyramid_factors_[l];
        }

        kf_params_.gaus_pyramid_hr_undistort_mask_.resize(num_gaus_pyramid_sub_levels_);
        cv::Mat white(cv::Size(kf_params_.hr_width_, kf_params_.hr_height_), 
            CV_32FC3, cv::Vec3f(1.0f, 1.0f, 1.0f));
        cv::remap(
            white, kf_params_.hr_undistort_mask_,
            kf_params_.hr_undistort_map1_, kf_params_.hr_undistort_map2_,
            cv::InterpolationFlags::INTER_LINEAR
        );
        kf_params_.lr_undistort_mask_tensor_ = tensor_utils::cvMat2TorchTensor_Float32(
            kf_params_.hr_undistort_mask_, device_type_);
        std::cout << "[Gaussian Mapper]HR undistort mask sum is set: kf_params_.hr_undistort_mask_" << std::endl;

        cv::cuda::GpuMat undistort_mask_gpu;
        undistort_mask_gpu.upload(kf_params_.hr_undistort_mask_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat undistort_mask_gpu_resized;
            cv::cuda::resize(undistort_mask_gpu, undistort_mask_gpu_resized,
                             cv::Size(kf_params_.gaus_pyramid_hr_width_[l], kf_params_.gaus_pyramid_hr_height_[l]));
            kf_params_.gaus_pyramid_hr_undistort_mask_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(undistort_mask_gpu_resized);
        }

        // orb
        this->kf_params_.orb_vocabulary_ = new ORB_SLAM3::ORBVocabulary();
        bool bVocLoad = this->kf_params_.orb_vocabulary_->loadFromTextFile(
            this->kf_params_.orb_vocab_path_);
        if(!bVocLoad)
            throw std::runtime_error("[Gaussian Mapper]Wrong path to ORB vocabulary: " + 
                this->kf_params_.orb_vocab_path_);
        else
            std::cout << "[Gaussian Mapper]ORB vocabulary loaded for HR pose optimization!" << std::endl;

        this->kf_params_.orb_extractor_lr_ = new ORB_SLAM3::ORBextractor(
            this->pSLAM_->getSettings()->nFeatures() * 10,
            this->pSLAM_->getSettings()->scaleFactor(),
            this->pSLAM_->getSettings()->nLevels(),
            this->pSLAM_->getSettings()->initThFAST(),
            this->pSLAM_->getSettings()->minThFAST());
        this->kf_params_.orb_extractor_hr_ = new ORB_SLAM3::ORBextractor(
            this->pSLAM_->getSettings()->nFeatures() * 20,
            this->pSLAM_->getSettings()->scaleFactor(),
            this->pSLAM_->getSettings()->nLevels(),
            this->pSLAM_->getSettings()->initThFAST(),
            this->pSLAM_->getSettings()->minThFAST());
        std::cout<<"[Gaussian Mapper] Base feature points for ORB extraction: "
            <<this->pSLAM_->getSettings()->nFeatures()<<std::endl;

        this->kf_params_.orb_camera_lr_ = new ORB_SLAM3::Pinhole(
            std::vector<float> {
                kf_params_.lr_fx_,
                kf_params_.lr_fy_,
                kf_params_.lr_cx_,
                kf_params_.lr_cy_
            });
        this->kf_params_.orb_camera_hr_ = new ORB_SLAM3::Pinhole(
            std::vector<float> {
                kf_params_.hr_fx_,
                kf_params_.hr_fy_,
                kf_params_.hr_cx_,
                kf_params_.hr_cy_
            });
    }

    // dense map points sampling
    // if(this->dense_map_points_){
        std::cout<<"[Gaussian Mapper]Setting dense keyframe depth map sampling..."<<std::endl;

        /* sample num of points in depth, deprecated
        float ratio = kf_params_.lr_width_ / kf_params_.lr_height_;
        float unit_num = sqrt(this->dense_sample_num_ / ratio);
        float width_step = (kf_params_.lr_width_ - 1) / (ratio * unit_num + 1);
        float height_step = (kf_params_.lr_height_ - 1) / (unit_num + 1);
        int width_num = int(ratio * unit_num);
        int height_num = int(unit_num);

        std::vector<float> width_coords(width_num);
        std::vector<float> height_coords(height_num);

        for (int i = 1; i < width_num + 1; i++)
            width_coords[i] = std::round(i * width_step);
        for (int i = 1; i < height_num + 1; i++)
            height_coords[i] = std::round(i * height_step);

        // Update actual counts (in case rounding caused bounds issues)
        this->dense_width_num_ = width_coords.size();
        this->dense_height_num_ = height_coords.size();

        // Create maps with proper dimensions
        cv::Mat width_base = cv::Mat(1, dense_width_num_, CV_32F, width_coords.data());
        cv::Mat height_base = cv::Mat(dense_height_num_, 1, CV_32F, height_coords.data());
        
        cv::repeat(width_base, dense_height_num_, 1, this->dense_width_map_);
        cv::repeat(height_base, 1, dense_width_num_, this->dense_height_map_);
        */

        // opencv based grids
        // cv::Mat width_base = cv::Mat::zeros(1, kf_params_.lr_width_, CV_32F);
        // for (int c = 0; c < kf_params_.lr_width_; ++c)
        //     width_base.at<float>(0, c) = static_cast<float>(c);
        // this->dense_width_map_ = cv::repeat(width_base, kf_params_.lr_height_, 1);
        // this->dense_width_map_ = this->dense_width_map_.reshape(1, 1); 

        // cv::Mat height_base = cv::Mat::zeros(kf_params_.lr_height_, 1, CV_32F);
        // for (int r = 0; r < kf_params_.lr_height_; ++r)
        //     height_base.at<float>(r, 0) = static_cast<float>(float(r));
        // this->dense_height_map_ = cv::repeat(height_base, 1, kf_params_.lr_width_);
        // this->dense_height_map_ = this->dense_height_map_.reshape(1, 1); 

        // cv::Mat width_base, height_base;
        // cv::linspace(0, kf_params_.lr_width_ - 1, kf_params_.lr_width_, width_base);
        // cv::linspace(0, kf_params_.lr_height_ - 1, kf_params_.lr_height_, height_base);

        // cv::repeat(width_base, kf_params_.lr_height_, 1, this->dense_width_map_);
        // cv::repeat(height_base, 1, kf_params_.lr_width_, this->dense_height_map_);
        
        // cacheKeyframeDepthMap
        this->dense_width_map_ = Eigen::ArrayXf::LinSpaced(kf_params_.lr_width_, 
            0, kf_params_.lr_width_-1).replicate(kf_params_.lr_height_, 1).reshaped<Eigen::RowMajor>();
        this->dense_height_map_ = Eigen::ArrayXf::LinSpaced(kf_params_.lr_height_, 
            0, kf_params_.lr_height_-1).replicate(1, kf_params_.lr_width_).reshaped<Eigen::RowMajor>();
        
        // std::cout<<"[debug] dense_width_map_ "<<dense_width_map_.rows()<<" "<<dense_width_map_.cols()<<std::endl;
    // }

    this->dense_width_map_tensor_ = torch::arange(this->kf_params_.lr_width_, torch::kFloat32).unsqueeze(0).repeat({this->kf_params_.lr_height_, 1}).to(device_type_);
    this->dense_height_map_tensor_ = torch::arange(this->kf_params_.lr_height_, torch::kFloat32).unsqueeze(1).repeat({1, this->kf_params_.lr_width_}).to(device_type_);
    this->dense_width_map_tensor_ = this->dense_width_map_tensor_.flatten();
    this->dense_height_map_tensor_ = this->dense_height_map_tensor_.flatten();

    // if(kf_params_.debug_){
    //     this->lpips_model_ = metrics_utils::load_lpips_model(std::filesystem::absolute("../third_party/lpips/lpips_vgg.pt"), device_type_);
    // }

    this->lpips_model_= std::make_shared<torch::jit::script::Module>();

    // std::cout<<"[debug!!!] check"<<std::endl;
    // Eigen::MatrixXf test_eigen(4,4);
    // test_eigen << 1,2,3,4,
    //              5,6,7,8,
    //              9,10,11,12,
    //              13,14,15,16;
    // std::cout<<test_eigen<<std::endl;
    // auto test_eigen2torch = tensor_utils::EigenMatrix2TorchTensor(test_eigen, device_type_);
    // std::cout<<test_eigen2torch<<std::endl;
    // auto test_torch2eigen = tensor_utils::TorchTensor2EigenMatrix(test_eigen2torch);
    // std::cout<<test_torch2eigen<<std::endl;
}

GaussianMapper::GaussianMapper(
    std::shared_ptr<CuVSLAMTracker> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pCuVSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    //=== consistent to the other constructor
    std::srand(seed);
    torch::manual_seed(seed);

    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        throw std::runtime_error("Please run on devices with cuda!");
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    this->kf_params_.result_dir_ = result_dir_;
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::cout << "[Gaussian Mapper]Pose optimization: "<<kf_params_.align_pose_<< std::endl;
    std::cout << "[Gaussian Mapper]Align pose between HR and LR: "<<kf_params_.render_aligned_<< std::endl;
    std::cout << "[Gaussian Mapper]Please set `KeyframeOptimization.render_aligned: 1` for HR and LR together."<< std::endl;
    std::cout << "[Gaussian Mapper]Please set `KeyframeOptimization.render_aligned: 0` for mono and rgbd."<< std::endl;

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    gaussians_ = std::make_shared<GaussianModel>(model_params_, this->kf_params_.debug_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    //=== updated for CuVSLAMTracker
    if(!this->pCuVSLAM_){
        // NO SLAM
        return;
    }

    if (this->wait_frontend_finish_)
        this->pCuVSLAM_->SetWaitFinishSlam();

    // lr camera calibrate
    this->pCuVSLAM_->GetCameraParameters(
        kf_params_.lr_width_, kf_params_.lr_height_,
        kf_params_.lr_fx_, kf_params_.lr_fy_,
        kf_params_.lr_cx_, kf_params_.lr_cy_,
        kf_params_.lr_fps_, this->rendered_depthmap_factor_
    );

    this->kf_params_.lr_depth_factor_ = this->rendered_depthmap_factor_;

    cv::Mat lr_K = (
        cv::Mat_<float>(3, 3)
            << kf_params_.lr_fx_, 0.f, kf_params_.lr_cx_,
                0.f, kf_params_.lr_fy_, kf_params_.lr_cy_,
                0.f, 0.f, 1.f
    );

    Camera camera;
    camera.camera_id_ = 0; // single camera for CuVSLAMTracker L515
    camera.setModelId(Camera::CameraModelType::PINHOLE);  
    camera.width_ = kf_params_.lr_width_;
    camera.height_ = kf_params_.lr_height_;
    camera.params_[0]/*fx*/= kf_params_.lr_fx_;
    camera.params_[1]/*fy*/= kf_params_.lr_fy_;
    camera.params_[2]/*cx*/= kf_params_.lr_cx_;
    camera.params_[3]/*cy*/= kf_params_.lr_cy_;
    camera.dist_coeff_ = cv::Mat::zeros(1, 4, CV_32F); // no distortion for CuVSLAMTracker L515

    camera.initUndistortRectifyMapAndMask(lr_K, cv::Size(kf_params_.lr_width_, kf_params_.lr_height_), lr_K, true);
    
    std::vector<cv::Mat> undistort_mask_channels;
    cv::split(camera.undistort_mask, undistort_mask_channels);
    kf_params_.lr_undistort_mask_ = undistort_mask_channels[0];
    kf_params_.lr_undistort_mask_ = (kf_params_.lr_undistort_mask_>(1.f-1e-4f));
    kf_params_.lr_undistort_mask_.convertTo(kf_params_.lr_undistort_mask_, CV_32FC1, 1.0f/255.0f);
    std::cout << "[Gaussian Mapper]LR undistort mask sum is set: kf_params_.lr_undistort_mask_" << std::endl;

    this->undistort_mask_[camera.camera_id_] = tensor_utils::cvMat2TorchTensor_Float32(camera.undistort_mask, device_type_);

    cv::Mat viewer_sub_undistort_mask;
    int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
    int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
    cv::resize(camera.undistort_mask, viewer_sub_undistort_mask, cv::Size(viewer_image_width_, viewer_image_height_));
    viewer_sub_undistort_mask_[camera.camera_id_] = tensor_utils::cvMat2TorchTensor_Float32(viewer_sub_undistort_mask, device_type_);

    cv::Mat viewer_main_undistort_mask;
    int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
    int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
    cv::resize(camera.undistort_mask, viewer_main_undistort_mask, cv::Size(viewer_image_width_main_, viewer_image_height_main_));
    viewer_main_undistort_mask_[camera.camera_id_] = tensor_utils::cvMat2TorchTensor_Float32(viewer_main_undistort_mask, device_type_);

    if (!viewer_camera_id_set_) {
        viewer_camera_id_ = camera.camera_id_;
        viewer_camera_id_set_ = true;
    }

    this->scene_->addCamera(camera);

    // hr camera calibrate
    std::cout << "[Gaussian Mapper]Setting HR camera intrinsics and undistort map..." << std::endl;
    cv::Mat hr_K = (
        cv::Mat_<float>(3, 3)
        <<  kf_params_.hr_fx_, 0.f, kf_params_.hr_cx_,
            0.f, kf_params_.hr_fy_, kf_params_.hr_cy_,
            0.f, 0.f, 1.f);
    cv::Mat hr_K2 = hr_K.clone();

    cv::Mat hr_dist_coeff = (cv::Mat_<float>(1, 4) << 
        kf_params_.hr_k1_, kf_params_.hr_k2_, kf_params_.hr_p1_, kf_params_.hr_p2_);

    cv::initUndistortRectifyMap(
        hr_K,
        hr_dist_coeff,
        cv::Mat::eye(3, 3, CV_32F),
        hr_K2,
        cv::Size(kf_params_.hr_width_, kf_params_.hr_height_),
        CV_32F,
        kf_params_.hr_undistort_map1_,
        kf_params_.hr_undistort_map2_
    );
    
    cv::Mat white(cv::Size(kf_params_.hr_width_, kf_params_.hr_height_), CV_32FC3, cv::Vec3f(1.0f, 1.0f, 1.0f));
    cv::remap(white, kf_params_.hr_undistort_mask_,
              kf_params_.hr_undistort_map1_, kf_params_.hr_undistort_map2_,
              cv::InterpolationFlags::INTER_LINEAR);
    kf_params_.lr_undistort_mask_tensor_ = tensor_utils::cvMat2TorchTensor_Float32(kf_params_.hr_undistort_mask_, device_type_);
    
    kf_params_.has_undistort_ = true;

    std::cout << "[Gaussian Mapper]HR undistort mask sum is set: kf_params_.hr_undistort_mask_" << std::endl;

    // orb
    this->kf_params_.orb_vocabulary_ = new ORB_SLAM3::ORBVocabulary();
    bool bVocLoad = this->kf_params_.orb_vocabulary_->loadFromTextFile(this->kf_params_.orb_vocab_path_);
    if(!bVocLoad)
        throw std::runtime_error("[Gaussian Mapper]Wrong path to ORB vocabulary: " + this->kf_params_.orb_vocab_path_);
    else
        std::cout << "[Gaussian Mapper]ORB vocabulary loaded for HR pose optimization!" << std::endl;

    this->kf_params_.orb_extractor_lr_ = new ORB_SLAM3::ORBextractor(
        this->pCuVSLAM_->orb_nFeatures_ * 10,
        this->pCuVSLAM_->orb_scaleFactor_,
        this->pCuVSLAM_->orb_nLevels_,
        this->pCuVSLAM_->orb_iniThFAST_,
        this->pCuVSLAM_->orb_minThFAST_
    );
    this->kf_params_.orb_extractor_hr_ = new ORB_SLAM3::ORBextractor(
        this->pCuVSLAM_->orb_nFeatures_ * 20,
        this->pCuVSLAM_->orb_scaleFactor_,
        this->pCuVSLAM_->orb_nLevels_,
        this->pCuVSLAM_->orb_iniThFAST_,
        this->pCuVSLAM_->orb_minThFAST_
    );
    std::cout << "[Gaussian Mapper]Base feature points for ORB extraction: " << this->pCuVSLAM_->orb_nFeatures_ << std::endl;

    this->kf_params_.orb_camera_lr_ = new ORB_SLAM3::Pinhole(
        std::vector<float> {
            kf_params_.lr_fx_,
            kf_params_.lr_fy_,
            kf_params_.lr_cx_,
            kf_params_.lr_cy_
        }
    );
    this->kf_params_.orb_camera_hr_ = new ORB_SLAM3::Pinhole(
        std::vector<float> {
            kf_params_.hr_fx_,
            kf_params_.hr_fy_,
            kf_params_.hr_cx_,
            kf_params_.hr_cy_
        }
    );

    // dense map points sampling
    this->dense_width_map_ = Eigen::ArrayXf::LinSpaced(kf_params_.lr_width_, 
        0, kf_params_.lr_width_-1).replicate(kf_params_.lr_height_, 1).reshaped<Eigen::RowMajor>();
    this->dense_height_map_ = Eigen::ArrayXf::LinSpaced(kf_params_.lr_height_, 
        0, kf_params_.lr_height_-1).replicate(1, kf_params_.lr_width_).reshaped<Eigen::RowMajor>();
    this->dense_width_map_tensor_ = torch::arange(this->kf_params_.lr_width_, torch::kFloat32).unsqueeze(0).repeat({this->kf_params_.lr_height_, 1}).to(device_type_);
    this->dense_height_map_tensor_ = torch::arange(this->kf_params_.lr_height_, torch::kFloat32).unsqueeze(1).repeat({1, this->kf_params_.lr_width_}).to(device_type_);
    this->dense_width_map_tensor_ = this->dense_width_map_tensor_.flatten();
    this->dense_height_map_tensor_ = this->dense_height_map_tensor_.flatten();

    //=== other
    this->lpips_model_= std::make_shared<torch::jit::script::Module>();

}

GaussianMapper::~GaussianMapper()
{
    if(this->pCuVSLAM_){
        pCuVSLAM_->Shutdown();
    }
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_rendered_opacity_ = 
        (settings_file["Record.record_rendered_opacity"].operator int()) != 0;
    record_rendered_depth_ = 
        (settings_file["Record.record_rendered_depth"].operator int()) != 0;
    record_rendered_depth_vis_ = 
        (settings_file["Record.record_rendered_depth_vis"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    wait_frontend_finish_ = 
        (settings_file["Optimization.wait_frontend_finish"].operator int()) != 0;
    init_train_iter_ =
        settings_file["Optimization.init_train_iter"].operator int();
    global_align_lr_pose_iter_ =
        settings_file["Optimization.global_align_lr_pose_iter"].operator int();
    global_align_lr_iter_ =
        settings_file["Optimization.global_align_lr_iter"].operator int();
    global_align_lr_depth_lambda_ =
        settings_file["Optimization.global_align_lr_depth_lambda"].operator float();
    global_align_hr_iter_ =
        settings_file["Optimization.global_align_hr_iter"].operator int();
    global_align_hr_warmup_iter_ =
        settings_file["Optimization.global_align_hr_warmup_iter"].operator int();
    global_align_hr_warmup_lr_dump_ =
        settings_file["Optimization.global_align_hr_warmup_lr_dump"].operator float();
    global_align_hr_resize_ratio_ =
        settings_file["Optimization.global_align_hr_resize_ratio"].operator float();
    global_align_time_window_ratio_ =
        settings_file["Optimization.global_align_time_window_ratio"].operator float();
    global_align_opacity_thr_ =
        settings_file["Optimization.global_align_opacity_thr"].operator float();
    global_align_hr_opacity_thr_ =
        settings_file["Optimization.global_align_hr_opacity_thr"].operator float();
    global_align_hr_pose_lambda_ =
        settings_file["Optimization.global_align_hr_pose_lambda"].operator float();
    global_align_hr_pose_reg_lambda_ =
        settings_file["Optimization.global_align_hr_pose_reg_lambda"].operator float();
    global_align_hr_pose_depth_lambda_ =
        settings_file["Optimization.global_align_hr_pose_depth_lambda"].operator float();
    global_align_hr_pose_iter_ =
        settings_file["Optimization.global_align_hr_pose_iter"].operator int();
    global_align_hr_color_iter_ =
        settings_file["Optimization.global_align_hr_color_iter"].operator int();
    global_align_hr_color_lambda_ =
        settings_file["Optimization.global_align_hr_color_lambda"].operator float();

    local_align_lr_pose_iter_ =
        settings_file["Optimization.local_align_lr_pose_iter"].operator int();
    local_align_lr_iter_ =
        settings_file["Optimization.local_align_lr_iter"].operator int();
    local_align_lr_opcacity_thr_ =
        settings_file["Optimization.local_align_lr_opcacity_thr"].operator float();
    local_align_hr_resize_ratio_ =
        settings_file["Optimization.local_align_hr_resize_ratio"].operator float();
    local_align_lr_joint_pose_iter_ =
        settings_file["Optimization.local_align_lr_joint_pose_iter"].operator int();
    local_align_lr_joint_pose_dump_ =
        settings_file["Optimization.local_align_lr_joint_pose_dump"].operator float();
    local_align_hr_pose_iter_ = 
        settings_file["Optimization.local_align_hr_pose_iter"].operator int();
    local_align_hr_opacity_thr_ =
        settings_file["Optimization.local_align_hr_opacity_thr"].operator float();
    local_align_hr_pose_reg_lambda_ =
        settings_file["Optimization.local_align_hr_pose_reg_lambda"].operator float();
    local_align_hr_pose_opacity_lambda_ =
        settings_file["Optimization.local_align_hr_pose_opacity_lambda"].operator float();
    local_align_hr_pose_depth_lambda_ = 
        settings_file["Optimization.local_align_hr_pose_depth_lambda"].operator float();
    local_align_hr_color_iter_ =    
        settings_file["Optimization.local_align_hr_color_iter"].operator int();
    local_align_hr_color_loss_thr_ =
        settings_file["Optimization.local_align_hr_color_loss_thr"].operator float();
    local_align_batch_size_ =
        settings_file["Optimization.local_align_batch_size"].operator int();
    local_align_batch_fillholes_opacity_thr_ =
        settings_file["Optimization.local_align_batch_fillholes_opacity_thr"].operator float();
    local_align_batch_conn_comp_min_size_ =
        settings_file["Optimization.local_align_batch_conn_comp_min_size"].operator int();
    local_align_batch_color_periter_ =
        settings_file["Optimization.local_align_batch_color_periter"].operator float();

    dense_map_points_ =
        (settings_file["Optimization.dense_map_points"].operator int()) != 0;
    dense_sample_num_ =
        settings_file["Optimization.dense_sample_num"].operator float();
    dense_diff_thld_ =
        settings_file["Optimization.dense_diff_threshold"].operator float();
    dense_diff_thld_ratio_ =
        settings_file["Optimization.dense_diff_threshold_ratio"].operator float();

    // Keyframe Optimization Parameters
    kf_params_.debug_ = 
        (settings_file["KeyframeOptimization.debug"].operator int()) != 0;
    kf_params_.debug_dir_ =
        settings_file["KeyframeOptimization.debug_dir"].operator std::string();
    kf_params_.align_pose_ = 
        (settings_file["KeyframeOptimization.align_pose"].operator int()) != 0;
    kf_params_.render_aligned_ = 
        (settings_file["KeyframeOptimization.render_aligned"].operator int()) != 0;
    kf_params_.align_global_pose_ = 
        (settings_file["KeyframeOptimization.align_global_pose"].operator int()) != 0;
    kf_params_.align_local_pose_ = 
        (settings_file["KeyframeOptimization.align_local_pose"].operator int()) != 0;
    kf_params_.align_exposure_ = 
        (settings_file["KeyframeOptimization.align_exposure"].operator int()) != 0;
    kf_params_.exposure_a_lr_ = 
        settings_file["KeyframeOptimization.exposure_a_lr"].operator float();
    kf_params_.exposure_b_lr_ =
        settings_file["KeyframeOptimization.exposure_b_lr"].operator float();
    kf_params_.theta_lr_ = 
        settings_file["KeyframeOptimization.theta_lr"].operator float();
    kf_params_.rho_lr_ =
        settings_file["KeyframeOptimization.rho_lr"].operator float();
    kf_params_.hr_color_theta_lr_ = 
        settings_file["KeyframeOptimization.hr_color_theta_lr"].operator float(); // change
    kf_params_.hr_color_rho_lr_ =
        settings_file["KeyframeOptimization.hr_color_rho_lr"].operator float(); // change
    kf_params_.hr_pixel_samples_ =
        settings_file["KeyframeOptimization.hr_pixel_samples"].operator int();
    kf_params_.hr_width_ =
        settings_file["KeyframeOptimization.hr_width"].operator int();
    kf_params_.hr_height_ =
        settings_file["KeyframeOptimization.hr_height"].operator int();
    kf_params_.hr_fps_ =
        settings_file["KeyframeOptimization.hr_fps"].operator float();
    kf_params_.hr_fx_ = 
        settings_file["KeyframeOptimization.hr_fx"].operator float();
    kf_params_.hr_fy_ = 
        settings_file["KeyframeOptimization.hr_fy"].operator float();
    kf_params_.hr_cx_ = 
        settings_file["KeyframeOptimization.hr_cx"].operator float();
    kf_params_.hr_cy_ = 
        settings_file["KeyframeOptimization.hr_cy"].operator float();
    kf_params_.hr_k1_ = 
        settings_file["KeyframeOptimization.hr_k1"].operator float();
    kf_params_.hr_k2_ = 
        settings_file["KeyframeOptimization.hr_k2"].operator float();
    kf_params_.hr_p1_ = 
        settings_file["KeyframeOptimization.hr_p1"].operator float();
    kf_params_.hr_p2_ = 
        settings_file["KeyframeOptimization.hr_p2"].operator float();
    kf_params_.hr_k3_ = 
        settings_file["KeyframeOptimization.hr_k3"].operator float();
    
    kf_params_.orb_vocab_path_ = 
        settings_file["KeyframeOptimization.ORB_Vocabulary"].operator std::string();

    kf_params_.hr_fovx_ = graphics_utils::focal2fov(kf_params_.hr_fx_, kf_params_.hr_width_);
    kf_params_.hr_fovy_ = graphics_utils::focal2fov(kf_params_.hr_fy_, kf_params_.hr_height_);

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapper::setOtherData(const std::vector<std::string>& paths, 
    const std::vector<double>& timestamps,
    const std::vector<std::vector<double>>& lr_gt_poses,
    const std::vector<std::vector<double>>& hr_gt_poses){

    this->vstrHRImagePaths_ = paths;

    if(!paths.empty() && paths.size() == timestamps.size()){
        this->vHRTimestamps_ = timestamps;
        this->hr_timestamps_exist_ = true;
    }
    else{
        float time_gap = 1.0f / kf_params_.hr_fps_;
        for(int i=0; i<paths.size(); i++)
            this->vHRTimestamps_.push_back(i * time_gap);
        std::cout<<"[Warning] HR timestamps not provided, using approximate timestamps with fps "<<kf_params_.hr_fps_<<std::endl;
    }

    if(!lr_gt_poses.empty() && !hr_gt_poses.empty()){
        this->vvLRGTPose_ = lr_gt_poses;
        this->vvHRGTPose_ = hr_gt_poses;
        this->gt_pose_exist_ = true;
    }
}

// deprecated
/*
cv::Mat GaussianMapper::sampleDepthMap(const cv::Mat& depth) {
    if (depth.empty() || 
        dense_width_map_.empty() || 
        dense_height_map_.empty() || 
        dense_width_num_ <= 0 || 
        dense_height_num_ <= 0) 
    {
        std::cout<<"[ERROR] Invalid depth sampling!"<<std::endl;
        return cv::Mat();
    }

    cv::Mat sampledValues;
    cv::remap(depth, sampledValues, 
              dense_width_map_, 
              dense_height_map_, 
              cv::INTER_NEAREST, 
              cv::BORDER_CONSTANT, 0);

    int totalPoints = dense_height_num_ * dense_width_num_;
    cv::Mat result(totalPoints, 3, CV_32F);
    
    cv::Mat mapX = dense_width_map_.reshape(0, totalPoints);
    cv::Mat mapY = dense_height_map_.reshape(0, totalPoints);
    
    cv::Mat valuesFlat = sampledValues.reshape(0, totalPoints);

    mapX.col(0).copyTo(result.col(0));
    mapY.col(0).copyTo(result.col(1));
    valuesFlat.col(0).copyTo(result.col(2));

    return result;
}
*/

// deprecated
/*
void GaussianMapper::cacheSampledDepthMap(std::shared_ptr<GaussianKeyframe> pkf, Sophus::SE3<float>& pose){
    auto means = pkf->hr_image_.mean({1, 2});
    // std::cout<<"[debug] clr mean\n"<<means<<std::endl;

    auto sampled_mat = this->sampleDepthMap(pkf->img_auxiliary_undist_);

    pkf->dense_depth_num_ = 0;
    for (int i = 0; i < sampled_mat.rows; ++i) {
        float d = sampled_mat.at<float>(i, 2);
        if(d<1e-5) continue;
        float u = sampled_mat.at<float>(i, 0);
        float v = sampled_mat.at<float>(i, 1);
        float x = (u - kf_params_.lr_cx_) * d / kf_params_.lr_fx_;
        float y = (v - kf_params_.lr_cy_) * d / kf_params_.lr_fy_;

        // mRwc * x3Dc + mOw;
        Eigen::Vector3f x3Dc(x, y, d);
        x3Dc = pose.inverse().rotationMatrix() * x3Dc + pose.translation();
                            
        Point3D point3D;
        point3D.xyz_(0) = x3Dc[0];
        point3D.xyz_(1) = x3Dc[1];
        point3D.xyz_(2) = x3Dc[2];
        point3D.color_(0) = means[0].item<float>();
        point3D.color_(1) = means[1].item<float>();
        point3D.color_(2) = means[2].item<float>();
        scene_->cachePoint3D(scene_->getPointNumber(), point3D);

        pkf->dense_depth_num_++;
    }
}
*/

cv::Mat GaussianMapper::getDepthRelated(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
    torch::Tensor pose1, torch::Tensor pose2, torch::Tensor give_depth)
{
    Sophus::SE3f se3f_pose1 = tensor_utils::TensorTransformation2SE3f(pose1);
    Sophus::SE3f se3f_pose2 = tensor_utils::TensorTransformation2SE3f(pose2);

    auto R1_inv = se3f_pose1.inverse().rotationMatrix();
    auto t1_inv = se3f_pose1.inverse().translation();
    auto R2 = se3f_pose2.rotationMatrix();
    auto t2 = se3f_pose2.translation();

    const auto& depth1_premask = pkf1->img_auxiliary_undist_;
    cv::Mat depth1 = depth1_premask.mul(pkf1->depth_undist_valid_mask_);

    cv::Mat depth2 = tensor_utils::torchTensor2CvMat_Float32(give_depth);

    int pixels_num = kf_params_.lr_width_ * kf_params_.lr_height_;
    Eigen::Map<Eigen::ArrayXf> z1(reinterpret_cast<float*>(depth1.data), pixels_num);
    // Eigen::Map<Eigen::ArrayXf> z2_gt(reinterpret_cast<float*>(depth2.data), pixels_num);

    // kf1 <- kf2
    Eigen::ArrayXf x1 = (this->dense_width_map_ - kf_params_.lr_cx_) * z1 / kf_params_.lr_fx_;
    Eigen::ArrayXf y1 = (this->dense_height_map_ - kf_params_.lr_cy_) * z1 / kf_params_.lr_fy_;
    
    Eigen::MatrixXf Pc1(3, pixels_num);
    Pc1.row(0) = x1.eval().transpose();
    Pc1.row(1) = y1.eval().transpose();
    Pc1.row(2) = z1.eval().transpose();

    Eigen::MatrixXf Pw1 = (R1_inv * Pc1).colwise() + t1_inv;
    Eigen::MatrixXf Pc2 = (R2 * Pw1).colwise() + t2;

    Eigen::ArrayXf z2 = Pc2.row(2).transpose().array();
    Eigen::ArrayXf x2 = Pc2.row(0).transpose().array();
    Eigen::ArrayXf y2 = Pc2.row(1).transpose().array();

    Eigen::ArrayXf u2 = x2 * kf_params_.lr_fx_ / z2 + kf_params_.lr_cx_;
    Eigen::ArrayXf v2 = y2 * kf_params_.lr_fy_ / z2 + kf_params_.lr_cy_;
    
    cv::Mat mapx(1, pixels_num, CV_32F, (void*)u2.data());
    cv::Mat mapy(1, pixels_num, CV_32F, (void*)v2.data());
    mapx = mapx.reshape(1, kf_params_.lr_height_);
    mapy = mapy.reshape(1, kf_params_.lr_height_);

    cv::Mat depth2_hat;
    cv::remap(depth2, depth2_hat, mapx, mapy, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    return depth2_hat;
}

cv::Mat GaussianMapper::getDepthRelated(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
    Sophus::SE3f pose1, Sophus::SE3f pose2)
{
    auto R1_inv = pose1.inverse().rotationMatrix();
    auto t1_inv = pose1.inverse().translation();
    auto R2 = pose2.rotationMatrix();
    auto t2 = pose2.translation();

    const auto& depth1_premask = pkf1->img_auxiliary_undist_;
    const auto& depth2_premask = pkf2->img_auxiliary_undist_;

    cv::Mat depth1 = depth1_premask.mul(pkf1->depth_undist_valid_mask_);
    cv::Mat depth2 = depth2_premask.mul(pkf2->depth_undist_valid_mask_);

    int pixels_num = kf_params_.lr_width_ * kf_params_.lr_height_;
    Eigen::Map<Eigen::ArrayXf> z1(reinterpret_cast<float*>(depth1.data), pixels_num);
    Eigen::Map<Eigen::ArrayXf> z2_gt(reinterpret_cast<float*>(depth2.data), pixels_num);

    // kf1 <- kf2
    Eigen::ArrayXf x1 = (this->dense_width_map_ - kf_params_.lr_cx_) * z1 / kf_params_.lr_fx_;
    Eigen::ArrayXf y1 = (this->dense_height_map_ - kf_params_.lr_cy_) * z1 / kf_params_.lr_fy_;
    
    Eigen::MatrixXf Pc1(3, pixels_num);
    Pc1.row(0) = x1.eval().transpose();
    Pc1.row(1) = y1.eval().transpose();
    Pc1.row(2) = z1.eval().transpose();

    Eigen::MatrixXf Pw1 = (R1_inv * Pc1).colwise() + t1_inv;
    Eigen::MatrixXf Pc2 = (R2 * Pw1).colwise() + t2;

    Eigen::ArrayXf z2 = Pc2.row(2).transpose().array();
    Eigen::ArrayXf x2 = Pc2.row(0).transpose().array();
    Eigen::ArrayXf y2 = Pc2.row(1).transpose().array();

    Eigen::ArrayXf u2 = x2 * kf_params_.lr_fx_ / z2 + kf_params_.lr_cx_;
    Eigen::ArrayXf v2 = y2 * kf_params_.lr_fy_ / z2 + kf_params_.lr_cy_;
    
    cv::Mat mapx(1, pixels_num, CV_32F, (void*)u2.data());
    cv::Mat mapy(1, pixels_num, CV_32F, (void*)v2.data());
    mapx = mapx.reshape(1, kf_params_.lr_height_);
    mapy = mapy.reshape(1, kf_params_.lr_height_);

    cv::Mat depth2_hat;
    cv::remap(depth2, depth2_hat, mapx, mapy, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    return depth2_hat;
}

// compute depth difference between two keyframes
cv::Mat GaussianMapper::getDepthDiff(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
    bool give_poses, Sophus::SE3f pose1, Sophus::SE3f pose2)
{
    if(!give_poses){
        pose1 = pkf1->getPosef();
        pose2 = pkf2->getPosef();
    }

    cv::Mat depth2_hat = this->getDepthRelated(pkf1, pkf2, pose1, pose2);

    const auto& depth1_premask = pkf1->img_auxiliary_undist_;
    cv::Mat depth1 = depth1_premask.mul(pkf1->depth_undist_valid_mask_);

    cv::Mat diff_kf2_to_kf1;
    cv::absdiff(depth2_hat, depth1, diff_kf2_to_kf1);

    return diff_kf2_to_kf1;
}

cv::Mat GaussianMapper::getDepthDiff(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
    torch::Tensor pose1, torch::Tensor pose2)
{
    Sophus::SE3f se3f_pose1 = tensor_utils::TensorTransformation2SE3f(pose1);
    Sophus::SE3f se3f_pose2 = tensor_utils::TensorTransformation2SE3f(pose2);

    return this->getDepthDiff(pkf1, pkf2, true, se3f_pose1, se3f_pose2);
}

void GaussianMapper::cacheKeyframeDepthMap(std::vector<std::size_t> kfids){
    std::cout<<"[Gaussian Mapper::cacheKeyframeDepthMap] Caching keyframe depth maps..."<<std::endl;
    std::vector<std::size_t> keyframes_ids;
    if (kfids.empty()) {
        this->scene_->getKeyframeIds(keyframes_ids);
    } else {
        keyframes_ids = kfids;
    }

    for (auto id : keyframes_ids){
        std::cout<<"[debug] caching depth for keyframe "<<id<<std::endl;
    }


    int iid0, iid1, iid2;
    for(int i=this->dense_fid_offset_; i>=0; i--){
        iid0 = i;
        iid1 = int(keyframes_ids.size()/3.f) + i;
        iid2 = int(keyframes_ids.size()/3.f*2.f) + i;
        if(iid2 < keyframes_ids.size()) break;
    }

    std::size_t id0 = keyframes_ids[iid0];
    std::size_t id1 = keyframes_ids[iid1];
    std::size_t id2 = keyframes_ids[iid2];

    this->dense_init_fids_.push_back(id0);
    this->dense_init_fids_.push_back(id1);
    this->dense_init_fids_.push_back(id2);
    const auto& pkf0 = this->scene_->getKeyframe(id0);
    const auto& pkf1 = this->scene_->getKeyframe(id1);
    const auto& pkf2 = this->scene_->getKeyframe(id2);
    const auto pose0 = pkf0->getPosef();
    const auto pose1 = pkf1->getPosef();
    const auto pose2 = pkf2->getPosef();

    std::cout<<"[debug] keyframes_ids_ "<<id0<<" "<<id1<<" "<<id2<<std::endl;
    std::cout<<"[debug] kf timestamps \n"<<pkf0->lr_timestamp_<<" "<<pkf1->lr_timestamp_<<" "<<pkf2->lr_timestamp_<<std::endl;
    // const auto pose0 = pkf0->getGTPosef();
    // const auto pose1 = pkf1->getGTPosef();
    // const auto pose2 = pkf2->getGTPosef();
    // std::cout<<"[debug] gt pose0 \n"<<pkf0->getGTPosef().unit_quaternion()<<std::endl;
    // std::cout<<"[debug] gt pose1 \n"<<pkf1->getGTPosef().unit_quaternion()<<std::endl;
    // std::cout<<"[debug] pose0 \n"<<pkf0->getPosef().unit_quaternion()<<" "<<pkf0->getPosef().translation().transpose()<<std::endl;
    // std::cout<<"[debug] pose1 \n"<<pkf1->getPosef().unit_quaternion()<<" "<<pkf1->getPosef().translation().transpose()<<std::endl;
    // std::cout<<"[debug] pose2 \n"<<pkf2->getPosef().unit_quaternion()<<" "<<pkf2->getPosef().translation().transpose()<<std::endl;
    std::cout<<"[debug] pose0 \n"<<pkf0->getPosef().matrix()<<std::endl;
    std::cout<<"[debug] pose1 \n"<<pkf1->getPosef().matrix()<<std::endl;
    std::cout<<"[debug] pose2 \n"<<pkf2->getPosef().matrix()<<std::endl;
    // std::cout<<"[debug] pose related \n"<<(pkf1->getPosef().translation()-pkf0->getPosef().translation())<<std::endl;
    // std::cout<<"[debug] gt pose related \n"<<(pkf1->getGTPosef().translation()-pkf0->getGTPosef().translation())<<std::endl;
    std::cout<<"[debug] kf paths \n"<<pkf0->img_filename_<<"\n"<<pkf1->img_filename_<<"\n"<<pkf2->img_filename_<<std::endl;

    const auto& depth0 = pkf0->img_auxiliary_undist_;
    const auto& depth1 = pkf1->img_auxiliary_undist_;
    const auto& depth2 = pkf2->img_auxiliary_undist_;
    auto depth0_valid_mask = pkf0->depth_undist_valid_mask_.clone();
    auto depth1_valid_mask = pkf1->depth_undist_valid_mask_.clone();
    auto depth2_valid_mask = pkf2->depth_undist_valid_mask_.clone();
    const auto& rgb0_raw = pkf0->img_undist_;
    const auto& rgb1_raw = pkf1->img_undist_;
    const auto& rgb2_raw = pkf2->img_undist_;

    // cv::Mat rand_mask(depth0_valid_mask.size(), CV_32F);
    // cv::randu(rand_mask, 0.0f, 1.0f);
    // cv::Mat mask = rand_mask > 0.92f;
    // depth0_valid_mask.setTo(0.f, ~mask);
    // depth1_valid_mask.setTo(0.f, ~mask);
    // depth2_valid_mask.setTo(0.f, ~mask);

    int pixels_num = kf_params_.lr_width_ * kf_params_.lr_height_;
    Eigen::Map<Eigen::ArrayXf> depth0_eigen(reinterpret_cast<float*>(depth0.data), pixels_num);
    Eigen::Map<Eigen::ArrayXf> depth1_eigen(reinterpret_cast<float*>(depth1.data), pixels_num);
    Eigen::Map<Eigen::ArrayXf> depth2_eigen(reinterpret_cast<float*>(depth2.data), pixels_num);
    cv::Mat rgb0_cv = rgb0_raw.reshape(3, pixels_num);
    cv::Mat rgb1_cv = rgb1_raw.reshape(3, pixels_num);
    cv::Mat rgb2_cv = rgb2_raw.reshape(3, pixels_num);

    
    Eigen::MatrixXf Pw0;
    general_utils::projectEigen_depth2pcd(
        Pw0, kf_params_.lr_cx_, kf_params_.lr_cy_, kf_params_.lr_fx_, kf_params_.lr_fy_,
        this->dense_width_map_, this->dense_height_map_, depth0_eigen,
        pose0
    );

    Eigen::MatrixXf Pw1;
    general_utils::projectEigen_depth2pcd(
        Pw1, kf_params_.lr_cx_, kf_params_.lr_cy_, kf_params_.lr_fx_, kf_params_.lr_fy_,
        this->dense_width_map_, this->dense_height_map_, depth1_eigen,
        pose1
    );

    Eigen::MatrixXf Pw2;
    general_utils::projectEigen_depth2pcd(
        Pw2,  kf_params_.lr_cx_, kf_params_.lr_cy_, kf_params_.lr_fx_, kf_params_.lr_fy_,
        this->dense_width_map_, this->dense_height_map_, depth2_eigen,
        pose2
    );

    cv::Mat depth_diff_Pw0_to_Pw1 = this->getDepthDiff(pkf1, pkf0);
    cv::Mat depth_diff_Pw0_to_Pw2 = this->getDepthDiff(pkf2, pkf0);
    cv::Mat depth_diff_Pw1_to_Pw2 = this->getDepthDiff(pkf2, pkf1);

    float threshold_Pw0_to_Pw1 = this->dense_diff_thld_;
    float threshold_Pw0_to_Pw2 = this->dense_diff_thld_;
    float threshold_Pw1_to_Pw2 = this->dense_diff_thld_;
    if(this->dense_diff_thld_<1e-5f) {
        threshold_Pw0_to_Pw1 = (cv::mean(depth_diff_Pw0_to_Pw1))[0] * this->dense_diff_thld_ratio_;
        threshold_Pw0_to_Pw2 = (cv::mean(depth_diff_Pw0_to_Pw2))[0] * this->dense_diff_thld_ratio_;
        threshold_Pw1_to_Pw2 = (cv::mean(depth_diff_Pw1_to_Pw2))[0] * this->dense_diff_thld_ratio_;
        std::cout<<"[debug] depth diff mean thld "<<threshold_Pw0_to_Pw1<<" "<<threshold_Pw0_to_Pw2<<" "<<threshold_Pw1_to_Pw2<<std::endl;
    }

    // auto iter_start_timing2 = std::chrono::steady_clock::now();

    torch::Tensor Pw0_tensor = torch::from_blob(Pw0.data(), {pixels_num, 3}, torch::kFloat).clone();
    torch::Tensor Pw1_tensor = torch::from_blob(Pw1.data(), {pixels_num, 3}, torch::kFloat).clone();
    torch::Tensor Pw2_tensor = torch::from_blob(Pw2.data(), {pixels_num, 3}, torch::kFloat).clone();

    torch::Tensor rgb0_tensor = torch::from_blob(rgb0_cv.data, {pixels_num, 3}, torch::kFloat).clone();
    torch::Tensor rgb1_tensor = torch::from_blob(rgb1_cv.data, {pixels_num, 3}, torch::kFloat).clone();
    torch::Tensor rgb2_tensor = torch::from_blob(rgb2_cv.data, {pixels_num, 3}, torch::kFloat).clone();

    torch::Tensor depth0_valid_mask_tensor = torch::from_blob(depth0_valid_mask.data, {pixels_num}, torch::kFloat).clone();
    torch::Tensor depth1_valid_mask_tensor = torch::from_blob(depth1_valid_mask.data, {pixels_num}, torch::kFloat).clone();
    torch::Tensor depth2_valid_mask_tensor = torch::from_blob(depth2_valid_mask.data, {pixels_num}, torch::kFloat).clone();

    torch::Tensor depth_diff_Pw0_to_Pw1_tensor = torch::from_blob(depth_diff_Pw0_to_Pw1.data, {pixels_num}, torch::kFloat).clone();
    torch::Tensor depth_diff_Pw0_to_Pw2_tensor = torch::from_blob(depth_diff_Pw0_to_Pw2.data, {pixels_num}, torch::kFloat).clone();
    torch::Tensor depth_diff_Pw1_to_Pw2_tensor = torch::from_blob(depth_diff_Pw1_to_Pw2.data, {pixels_num}, torch::kFloat).clone();

    torch::Tensor full_valid_mask0 = depth0_valid_mask_tensor > 1e-5f;
    torch::Tensor full_valid_mask1 = (depth1_valid_mask_tensor > 1e-5f) & 
        (depth_diff_Pw0_to_Pw1_tensor > threshold_Pw0_to_Pw1);
    torch::Tensor full_valid_mask2 = (depth2_valid_mask_tensor > 1e-5f) & 
        (depth_diff_Pw0_to_Pw2_tensor > threshold_Pw0_to_Pw2) & 
        (depth_diff_Pw1_to_Pw2_tensor > threshold_Pw1_to_Pw2);

    torch::Tensor id0_tensor = torch::full({pixels_num}, int(id0), torch::kInt32);
    torch::Tensor id1_tensor = torch::full({pixels_num}, int(id1), torch::kInt32);
    torch::Tensor id2_tensor = torch::full({pixels_num}, int(id2), torch::kInt32);

    std::vector<torch::Tensor> xyz_list, rgb_list, idx_list;
    if (full_valid_mask0.any().item<bool>()) {
        xyz_list.push_back(Pw0_tensor.index_select(0, torch::nonzero(full_valid_mask0).squeeze()));
        rgb_list.push_back(rgb0_tensor.index_select(0, torch::nonzero(full_valid_mask0).squeeze()));
        idx_list.push_back(id0_tensor.index_select(0, torch::nonzero(full_valid_mask0).squeeze()));
    }
    if (full_valid_mask1.any().item<bool>()) {
        xyz_list.push_back(Pw1_tensor.index_select(0, torch::nonzero(full_valid_mask1).squeeze()));
        rgb_list.push_back(rgb1_tensor.index_select(0, torch::nonzero(full_valid_mask1).squeeze()));
        idx_list.push_back(id1_tensor.index_select(0, torch::nonzero(full_valid_mask1).squeeze()));
    }
    if (full_valid_mask2.any().item<bool>()) {
        xyz_list.push_back(Pw2_tensor.index_select(0, torch::nonzero(full_valid_mask2).squeeze()));
        rgb_list.push_back(rgb2_tensor.index_select(0, torch::nonzero(full_valid_mask2).squeeze()));
        idx_list.push_back(id2_tensor.index_select(0, torch::nonzero(full_valid_mask2).squeeze()));
    }

    this->dense_init_pcd_xyz_ = torch::cat(xyz_list, 0);
    this->dense_init_pcd_rgb_ = torch::cat(rgb_list, 0);
    this->dense_init_pcd_idx_ = torch::cat(idx_list, 0);

    // auto iter_start_timing3 = std::chrono::steady_clock::now();
    // auto iter_time1 = std::chrono::duration_cast<std::chrono::milliseconds>(
    //                 iter_start_timing2 - iter_start_timing1).count();
    // auto iter_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(
    //                 iter_start_timing3 - iter_start_timing2).count();
    // std::cout<<"[debug] cacheKeyframeDepthMap timing: loop "<<iter_time1
    //          <<" ms, vectorize "<<iter_time2<<" ms."<<std::endl;

    // draw
    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth"))
       {
        cv::Mat depth0_vis = depth0 * this->rendered_depthmap_factor_;
        depth0_vis.convertTo(depth0_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth0.jpg", depth0_vis);
       } 
       {
        cv::Mat depth1_vis = depth1 * this->rendered_depthmap_factor_;
        depth1_vis.convertTo(depth1_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth1.jpg", depth1_vis);
       } 
       {
        cv::Mat depth2_vis = depth2 * this->rendered_depthmap_factor_;
        depth2_vis.convertTo(depth2_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth2.jpg", depth2_vis);
       }
       {
        cv::Mat depth_diff_Pw0_to_Pw1_vis = depth_diff_Pw0_to_Pw1 * this->rendered_depthmap_factor_;
        depth_diff_Pw0_to_Pw1_vis.convertTo(depth_diff_Pw0_to_Pw1_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth_diff_Pw0_to_Pw1_vis.jpg", depth_diff_Pw0_to_Pw1_vis);
       }
       {
        cv::Mat depth_diff_Pw0_to_Pw2_vis = depth_diff_Pw0_to_Pw2 * this->rendered_depthmap_factor_;
        depth_diff_Pw0_to_Pw2_vis.convertTo(depth_diff_Pw0_to_Pw2_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth_diff_Pw0_to_Pw2_vis.jpg", depth_diff_Pw0_to_Pw2_vis);
       }
       {
        cv::Mat depth_diff_Pw1_to_Pw2_vis = depth_diff_Pw1_to_Pw2 * this->rendered_depthmap_factor_;
        depth_diff_Pw1_to_Pw2_vis.convertTo(depth_diff_Pw1_to_Pw2_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth_diff_Pw1_to_Pw2_vis.jpg", depth_diff_Pw1_to_Pw2_vis);
       }
       { // depth1_eigen[i]>1e-5f && depth_diff_Pw0_to_Pw1_eigen[i]>threshold_Pw0_to_Pw1
        cv::Mat depth1_mask = (depth1>1e-5f);
        depth1_mask.convertTo(depth1_mask, CV_32FC1, 1.f/255.f);
        cv::Mat depth1_diff_mask = (depth_diff_Pw0_to_Pw1>threshold_Pw0_to_Pw1);
        depth1_diff_mask.convertTo(depth1_diff_mask, CV_32FC1, 1.f/255.f);
        cv::Mat depth_diff_Pw0_to_Pw1_vis = depth_diff_Pw0_to_Pw1 * this->rendered_depthmap_factor_;
        depth_diff_Pw0_to_Pw1_vis = depth_diff_Pw0_to_Pw1_vis.mul(depth1_mask.mul(depth1_diff_mask));
        depth_diff_Pw0_to_Pw1_vis.convertTo(depth_diff_Pw0_to_Pw1_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth_diff_Pw0_to_Pw1_vis_masked.jpg", depth_diff_Pw0_to_Pw1_vis);
       }
       { // depth2_eigen[i]>1e-5f && depth_diff_Pw0_to_Pw2_eigen[i]>threshold_Pw0_to_Pw2 && !(depth_diff_Pw1_to_Pw2_eigen[i]>threshold_Pw1_to_Pw2)
        cv::Mat depth2_mask = (depth2>1e-5f);
        depth2_mask.convertTo(depth2_mask, CV_32FC1, 1.f/255.f);
        cv::Mat depth2_diff_mask = (depth_diff_Pw0_to_Pw2>threshold_Pw0_to_Pw2);
        depth2_diff_mask.convertTo(depth2_diff_mask, CV_32FC1, 1.f/255.f);
        cv::Mat depth2_diff_Pw1_to_Pw2_mask = (depth_diff_Pw1_to_Pw2>threshold_Pw1_to_Pw2);
        depth2_diff_Pw1_to_Pw2_mask.convertTo(depth2_diff_Pw1_to_Pw2_mask, CV_32FC1, 1.f/255.f);
        cv::Mat depth_diff_Pw0_to_Pw2_vis = depth_diff_Pw0_to_Pw2 * this->rendered_depthmap_factor_;
        depth_diff_Pw0_to_Pw2_vis = depth_diff_Pw0_to_Pw2_vis.mul(depth2_mask.mul(depth2_diff_mask.mul(1.f - depth2_diff_Pw1_to_Pw2_mask)));
        depth_diff_Pw0_to_Pw2_vis.convertTo(depth_diff_Pw0_to_Pw2_vis, CV_8UC1, 255.0f/6000.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "depth_diff_Pw0_to_Pw2_vis_masked.jpg", depth_diff_Pw0_to_Pw2_vis);
       }
    //    {
    //     cv::Mat mean_depth0, mean_depth1, mean_depth2;
    //     cv::Mat mean_sqr_depth0, mean_sqr_depth1, mean_sqr_depth2;
    //     cv::Mat var_depth0, var_depth1, var_depth2;
    //     cv::boxFilter(depth0_premask, mean_depth0, CV_32F, cv::Size(7,7));
    //     cv::boxFilter(depth1_premask, mean_depth1, CV_32F, cv::Size(7,7));
    //     cv::boxFilter(depth2_premask, mean_depth2, CV_32F, cv::Size(7,7));
    //     cv::sqrBoxFilter(depth0, mean_sqr_depth0, CV_32F, cv::Size(7,7));
    //     cv::sqrBoxFilter(depth1, mean_sqr_depth1, CV_32F, cv::Size(7,7));
    //     cv::sqrBoxFilter(depth2, mean_sqr_depth2, CV_32F, cv::Size(7,7));
    //     var_depth0 = mean_sqr_depth0 - mean_depth0.mul(mean_depth0);
    //     var_depth1 = mean_sqr_depth1 - mean_depth1.mul(mean_depth1);
    //     var_depth2 = mean_sqr_depth2 - mean_depth2.mul(mean_depth2);

    //     cv::Mat var_depth0_vis = var_depth0 * 600.f;
    //     cv::Mat var_depth1_vis = var_depth1 * 600.f;
    //     cv::Mat var_depth2_vis = var_depth2 * 600.f;
    //     var_depth0_vis.convertTo(var_depth0_vis, CV_8UC1, 1.f, 0.f);
    //     var_depth1_vis.convertTo(var_depth1_vis, CV_8UC1, 1.f, 0.f);
    //     var_depth2_vis.convertTo(var_depth2_vis, CV_8UC1, 1.f, 0.f);
    //     cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "var_depth0_vis.jpg", var_depth0_vis);
    //     cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "var_depth1_vis.jpg", var_depth1_vis);
    //     cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "init_depth" / "var_depth2_vis.jpg", var_depth2_vis);
    //    }
    }
}

void GaussianMapper::testCacheAllKeyframeDepthMap(){
    std::cout<<"[Gaussian Mapper::testCacheAllKeyframeDepthMap] Testing cached keyframe depth maps..."<<std::endl;

    int test_num = 300;
    while (1)
    {
        std::vector<std::size_t> tracking_ids, mapping_ids;
        this->pCuVSLAM_->GetFrameIds(tracking_ids);
        this->scene_->getKeyframeIds(mapping_ids);

        if(mapping_ids.size()>test_num) break;
        
        std::cout<<"[debug] size of tracking_ids "<<tracking_ids.size()<<" size of mapping_ids "<<mapping_ids.size()<<std::endl;
        // diff tracking_ids and mapping_ids
        if(tracking_ids.size()>mapping_ids.size()){
            std::vector<std::size_t> diff_ids;
            std::set_difference(tracking_ids.begin(), tracking_ids.end(), mapping_ids.begin(), mapping_ids.end(), std::back_inserter(diff_ids));
            std::cout<<"[debug] size of diff_ids "<<diff_ids.size()<<std::endl;
            for(std::size_t id : diff_ids){
                std::cout<<"[debug] insert keyframe id "<<id<<" to cache"<<std::endl;
                this->handleKeyframeFrontend(id);
                auto& pkf = this->scene_->keyframes_.at(id);
                auto gt_image = pkf->getGTLRImg(true);
                auto gt_depth = pkf->getGTLRDpt(true);
                auto valid_depth_mask = pkf->getGTLRDptMsk(true);
                auto valid_depth = gt_depth * valid_depth_mask;
                auto rand_mask = torch::rand_like(valid_depth);
                auto mask = (rand_mask > 0.92f).to(torch::kFloat32).cuda();
                valid_depth = valid_depth * mask;
                this->gaussians_->increaseKeyframeInitPcd(pkf, valid_depth, gt_image, this->kf_params_);
            }
        }
        // sleep for 100 ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    this->savePly(this->result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "ply_lr_test", false);
    throw std::runtime_error("testCacheAllKeyframeDepthMap finished!");
}

void GaussianMapper::generatePyramidSizes(std::shared_ptr<GaussianKeyframe> pkf, const Camera& camera){
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
        pkf->gaus_pyramid_height_ = kf_params_.gaus_pyramid_hr_height_;
        pkf->gaus_pyramid_width_ = kf_params_.gaus_pyramid_hr_width_;
    }
    else{
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    }
}

void GaussianMapper::generatePyramidFrames(std::shared_ptr<GaussianKeyframe> pkf){
    cv::cuda::GpuMat img_gpu;
    if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
        auto hr_img = pkf->getGTHRImg();
        img_gpu.upload(tensor_utils::torchTensor2CvMat_Float32(hr_img));
    }
    else 
        img_gpu.upload(pkf->img_undist_);

    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::cuda::GpuMat img_resized;
        cv::cuda::resize(img_gpu, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
        pkf->gaus_pyramid_original_image_[l] =
            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
    }
}

void GaussianMapper::getAvgGlobalPose(std::vector<std::size_t>& kfids, bool soften, float soften_ratio){
    torch::NoGradGuard no_grad;

    torch::Tensor avg_pose;
    this->getAvgGlobalPose(kfids, avg_pose, soften, soften_ratio);
}

void GaussianMapper::getAvgGlobalPose(std::vector<std::size_t>& kfids, torch::Tensor& avg_pose, bool soften, float soften_ratio){
    torch::NoGradGuard no_grad;

    torch::Tensor theta_avg = torch::zeros(3, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor rho_avg = torch::zeros(3, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    std::vector<bool> valid_poses;

    int valid_num = 0;
    for(int id = 0; id<kfids.size(); id++){
        auto& pkf = scene_->keyframes().at(kfids.at(id));

        auto pose = pkf->getGlobalDeltaPose();
        auto se3_lie = tensor_utils::se3_log(pose);

        auto theta = std::get<0>(se3_lie);
        auto rho = std::get<1>(se3_lie);

        if(theta.isnan().any().item<bool>() || rho.isnan().any().item<bool>())
            if(theta.isinf().any().item<bool>() || rho.isinf().any().item<bool>()){
                std::cout<<"[warning] `getAvgGlobalPose`: invalid pose kf "<<pkf->fid_<<" pose\n"<<pose<<" theta\n"<<theta<<" rho\n"<<rho<<std::endl;
                valid_poses.push_back(false);
                continue;
            }

        theta_avg += theta;
        rho_avg += rho;

        valid_num++;
        valid_poses.push_back(true);

    }

    if(valid_num==0){
        std::cout<<"[error] `getAvgGlobalPose`: no valid keyframe poses!"<<std::endl;
        throw std::runtime_error("`getAvgGlobalPose` no valid keyframe poses!");
    }
    else{
        theta_avg /= float(valid_num);
        rho_avg /= float(valid_num);
    }

    avg_pose = tensor_utils::se3_exp(theta_avg, rho_avg);

    if(soften){
        for(int id = 0; id<kfids.size(); id++){
            auto& pkf = scene_->keyframes().at(kfids.at(id));

            if(!valid_poses.at(id)){
                pkf->setGlobalDeltaPose(avg_pose);
                continue;
            }
            
            auto pose = pkf->getGlobalDeltaPose();
            auto se3_lie = tensor_utils::se3_log(pose);

            auto theta = std::get<0>(se3_lie);
            auto rho = std::get<1>(se3_lie);

            theta = theta * soften_ratio + theta_avg * (1.f - soften_ratio);
            rho = rho * soften_ratio + rho_avg * (1.f - soften_ratio);

            auto softened_pose = tensor_utils::se3_exp(theta, rho);
            pkf->setGlobalDeltaPose(softened_pose);
        }
    }
    // else{
    //     for(int id = 0; id<kfids.size(); id++){
    //         auto& pkf = scene_->keyframes().at(kfids.at(id));
    //         pkf->setGlobalDeltaPose(avg_pose);
    //     }
    // }
}

void GaussianMapper::run()
{
    // First loop: Initial gaussian mapping
    while (!isStopped()) {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            pSLAM_->getAtlas()->clearMappingOperation();

            // Get initial sparse map
            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                if(!this->dense_map_points_){
                    vpMPs = pMap->GetAllMapPoints();
                    // std::cout<<"[debug] vpmaps size "<<vpMPs.size()<<std::endl; // 1500
                    for (const auto& pMP : vpMPs){
                        Point3D point3D;
                        auto pos = pMP->GetWorldPos();
                        point3D.xyz_(0) = pos.x();
                        point3D.xyz_(1) = pos.y();
                        point3D.xyz_(2) = pos.z();
                        auto color = pMP->GetColorRGB();
                        point3D.color_(0) = color(0);
                        point3D.color_(1) = color(1);
                        point3D.color_(2) = color(2);
                        scene_->cachePoint3D(pMP->mnId, point3D);
                    }
                }

                vpKFs = pMap->GetAllKeyFrames();
                for (const auto& pKF : vpKFs){
                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(
                        pKF->mnId, getIteration(), &this->kf_params_);
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        // Camera
                        Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        cv::Mat imgRGB = pKF->imgLeftRGB;
                        if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        // Auxiliary Image
                        cv::Mat imgAux = pKF->imgAuxiliary;
                        if (this->sensor_type_ == RGBD)
                            camera.undistortImage(imgAux, imgAux_undistorted);
                        else
                            imgAux_undistorted = imgAux;

                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        new_kf->img_filename_ = pKF->mNameFile;

                        this->generatePyramidSizes(new_kf, camera);

                        if(kf_params_.debug_){
                            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_undist"))
                            cv::Mat imgRGB_undistorted_vis, imgRGB_vis;
                            imgRGB_undistorted.convertTo(imgRGB_undistorted_vis, CV_8UC3, 255.0f, 0.f);
                            imgRGB.convertTo(imgRGB_vis, CV_8UC3, 255.0f, 0.f);
                            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB_undistorted.jpg"), imgRGB_undistorted_vis);
                            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB.jpg"), imgRGB_vis);

                            cv::Mat imgAux_undistorted_vis, imgAux_vis;
                            imgAux_undistorted.convertTo(imgAux_undistorted_vis, CV_8UC1, 4000.0f/6000.0f*255.0f, 0.f);
                            imgAux.convertTo(imgAux_vis, CV_8UC1, 4000.0f/6000.0f*255.0f, 0.f);
                            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux_undistorted.jpg"), imgAux_undistorted_vis);
                            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux.jpg"), imgAux_vis);
                        }

                        // new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                        // if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
                        //     new_kf->gaus_pyramid_height_ = kf_params_.gaus_pyramid_hr_height_;
                        //     new_kf->gaus_pyramid_width_ = kf_params_.gaus_pyramid_hr_width_;
                        // }
                        // else{
                        //     new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                        //     new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                        // }
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }
                    // new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // Features
                    // std::vector<float> pixels;
                    // std::vector<float> pointsLocal;
                    // pKF->GetKeypointInfo(pixels, pointsLocal);
                    // new_kf->kps_pixel_ = std::move(pixels);
                    // new_kf->kps_point_local_ = std::move(pointsLocal);
                    // new_kf->img_undist_ = imgRGB_undistorted;
                    
                    // new_kf->setGTLRImg(pKF->imgLeftRGB);
                    new_kf->setGTLRImg(imgRGB_undistorted); // !!! color not need undistort?
                    // new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                    // new_kf->setGTLRDpt(pKF->imgAuxiliary);
                    new_kf->setGTLRDpt(imgAux_undistorted); // !!! depth not need undistort?

                    // align
                    if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
                        new_kf->lr_timestamp_ = pKF->mTimeStamp;

                        //gt
                        if(this->gt_pose_exist_){
                            auto& gt_lr = this->vvLRGTPose_[new_kf->fid_];
                            // auto& gt_hr = this->vvHRGTPose_[new_kf->hr_fid_];
                            new_kf->setGTPose(gt_lr[6], gt_lr[3], gt_lr[4], gt_lr[5], // qw,qx,qy,qz
                                gt_lr[0], gt_lr[1], gt_lr[2]); // tx, ty, tz
                            // std::cout<<"[debug] gt pose "<<new_kf->fid_<<" "<<gt_lr[6]<<" "<<gt_lr[3]<<" "<<gt_lr[4]<<" "<<gt_lr[5]<<" "<<gt_lr[0]<<" "<<gt_lr[1]<<" "<<gt_lr[2]<<std::endl;
                        }
                    }

                    // if(this->dense_map_points_)
                        // this->cacheSampledDepthMap(new_kf, pose);
                }

                if(this->dense_map_points_)
                    this->cacheKeyframeDepthMap();
            }

            // std::vector<int> valid_select_fids;
            // Prepare multi resolution images for training
            // for (auto& kfit : scene_->keyframes()) {
            //     this->global_align_select_fids_.push_back(kfit.first);
            //     auto pkf = kfit.second;
            //     if (device_type_ == torch::kCUDA) {
            //         // use lr or hr rgb
            //         cv::cuda::GpuMat img_gpu;
            //         // if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_)
            //         //     // img_gpu.upload(pkf->hr_image_undist_);
            //         //     img_gpu = tensor_utils::torchTensor2CvGpuMat_Float32(pkf->hr_image_undist_);
            //         // else 
            //             img_gpu.upload(pkf->img_u
            //             ndist_);
            //         pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            //         for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            //             cv::cuda::GpuMat img_resized;
            //             cv::cuda::resize(img_gpu, img_resized,
            //                             cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            //             pkf->gaus_pyramid_original_image_[l] =
            //                 tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            //         }
            //     }
            //     else {
            //         throw std::runtime_error("Please run on devices with cuda!");
            //         // pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            //         // for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            //         //     // use lr or hr rgb
            //         //     cv::Mat img_resized;
            //         //     if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_)
            //         //         cv::resize(pkf->hr_image_undist_, img_resized,
            //         //                 cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            //         //     else
            //         //         cv::resize(pkf->img_undist_, img_resized,
            //         //                 cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            //         //     pkf->gaus_pyramid_original_image_[l] =
            //         //         tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            //         // }
            //     }
            // }

            // Prepare for training
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                // gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                gaussians_->createFromPcd(this->dense_init_pcd_xyz_, this->dense_init_pcd_rgb_, this->dense_init_pcd_idx_, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }

            savePly(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "ply_lr_init", false);
            // throw std::runtime_error("[GaussianMapper::run]debug!");

            // Invoke training once
            // for(int ini = 0; ini<this->init_train_iter_; ini++)
            //     trainForOneIteration();

            if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_)
                this->optimizeGlobalAlign();

            // throw std::runtime_error("[GaussianMapper::run]debug!");

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()){
                auto pkf = kfit.second;
                if(!pkf->has_hr_fid_) continue;

                this->generatePyramidFrames(pkf);

                // use lr or hr rgb
                // cv::cuda::GpuMat img_gpu;
                // if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
                //     std::cout<<"[debug] using hr image for gaus pyramid "<<pkf->fid_<<" "<<pkf->hr_fid_<<std::endl;

                //     auto hr_img = pkf->getGTHRImg();
                //     std::cout<<"[debug] hr img size "<<hr_img.sizes()<<std::endl;
                //     img_gpu.upload(tensor_utils::torchTensor2CvMat_Float32(hr_img));
                // }
                // else 
                //     img_gpu.upload(pkf->img_undist_);

                // pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                // for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                //     std::cout<<"[debug] gaus pyramid level "<<l<<" size "<<pkf->gaus_pyramid_width_[l]<<" "<<pkf->gaus_pyramid_height_[l]<<std::endl;

                //     cv::cuda::GpuMat img_resized;
                //     cv::cuda::resize(img_gpu, img_resized,
                //                     cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                //     pkf->gaus_pyramid_original_image_[l] =
                //         tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                // }
            }
            
            // Invoke training once
            for(int ini = 0; ini<this->init_train_iter_; ini++)
                trainForOneIteration();

            savePly(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "ply_lr_train", false);
            // throw std::runtime_error("[GaussianMapper::run]debug!");

            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
            break;
        }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            // combineMappingOperations();
            this->insertNewKeyframesFromSLAM();
            // throw std::runtime_error("[GaussianMapper::insertNewKeyframesFromSLAM] debug throw!");
            if (cull_keyframes_)
                cullKeyframes();
        }

        // Invoke training once
        // trainForOneIteration();

        if (pSLAM_->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

    // // Third loop: Tail gaussian optimization
    // int densify_interval = densifyInterval();
    // int n_delay_iters = densify_interval * 0.8;
    // while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
    //     trainForOneIteration();
    //     densify_interval = densifyInterval();
    //     n_delay_iters = densify_interval * 0.8;
    // }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply", false);
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

void GaussianMapper::run_cuvslam(){
    const int init_mapping_frames = 10; // !!! debug

    //=== CuVSLAM integrated mapping ===//

    // First loop: Initial gaussian mapping
    std::cout<<"[GaussianMapper::run_cuvslam] Initial mapping loop start, waiting for CuVSLAM to accumulate enough frames..."<<std::endl;
    if (this->wait_frontend_finish_)
        std::cout<<"[GaussianMapper::run_cuvslam] Wait for CuVSLAM frontend tracking to finish before starting initial mapping."<<std::endl;
    
    while (!this->isStopped()){
        if (this->wait_frontend_finish_ && !this->pCuVSLAM_->IsTrackingFinished()){
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout<<"[GaussianMapper::run_cuvslam] CuVSLAM frontend tracking, current frame count: "<<this->pCuVSLAM_->GetFrameCount()<<std::endl;
        }
        else if (this->pCuVSLAM_->GetFrameCount() < init_mapping_frames)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        else {
            this->initial_mapped_ = true;
            break;
        }
    }

    if(this->wait_frontend_finish_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / "cuvslam_tracking"))
        this->pCuVSLAM_->SaveTrajectory(result_dir_ / "cuvslam_tracking");
    }

    std::cout<<"[GaussianMapper::run_cuvslam] Initial mapping loop end, start processing CuVSLAM frames..."<<std::endl;
    {
        std::unique_lock<std::mutex> lock_render(this->mutex_render_);

        std::vector<std::size_t> kfids;
        this->pCuVSLAM_->GetFrameIds(kfids);

        if (kfids.size() > init_mapping_frames && this->wait_frontend_finish_){
            kfids.resize(init_mapping_frames);
            std::cout<<"[GaussianMapper::run_cuvslam] CuVSLAM frame count "<<this->pCuVSLAM_->GetFrameCount()<<" exceeds initial mapping frames "<<init_mapping_frames<<", only process the first "<<init_mapping_frames<<" frames for initial mapping."<<std::endl;
        }

        for(int i=0; i<kfids.size(); i++){
            auto frame_id = kfids.at(i);
            this->handleKeyframeFrontend(frame_id);
        }

        // obtain initial dense pcd 
        this->cacheKeyframeDepthMap(kfids);
    }
    
    {
        std::unique_lock<std::mutex> lock_render(this->mutex_render_);
        this->scene_->cameras_extent_ = std::get<1>(this->scene_->getNerfppNorm());
        // test dense xyz rgb idx to create gs
        // project all 1's depth map by identity pose to get xyz
        // cv::Mat depth_ones = cv::Mat::ones(this->kf_params_.lr_height_, this->kf_params_.lr_width_, CV_32FC1);
        // int pixels_num = kf_params_.lr_width_ * kf_params_.lr_height_;
        // Eigen::Map<Eigen::ArrayXf> depth_ones_eigen(reinterpret_cast<float*>(depth_ones.data), pixels_num);
        // Eigen::MatrixXf Pw;
        // Sophus::SE3f eye = Sophus::SE3f();
        // std::cout<<"[debug] fxfycxcy "<<kf_params_.lr_fx_<<" "<<kf_params_.lr_fy_<<" "<<kf_params_.lr_cx_<<" "<<kf_params_.lr_cy_<<std::endl;
        // general_utils::projectEigen_depth2pcd(
        //     Pw, kf_params_.lr_cx_, kf_params_.lr_cy_, kf_params_.lr_fx_, kf_params_.lr_fy_,
        //     this->dense_width_map_, this->dense_height_map_, depth_ones_eigen,
        //     eye
        // );
        // auto test_xyz = torch::from_blob(Pw.data(), {pixels_num, 3}, torch::kFloat32).clone();
        // auto test_rgb = torch::ones({pixels_num, 3}, torch::kFloat32);
        // // change center pixel color to red
        // int center_idx = (kf_params_.lr_height_ / 2) * kf_params_.lr_width_ + (kf_params_.lr_width_ / 2);
        // test_rgb[center_idx][1] = 0.f;
        // test_rgb[center_idx][2] = 0.f;
        // auto test_idx = torch::zeros({pixels_num}, torch::kInt64);
        // this->gaussians_->createFromPcd(test_xyz, test_rgb, test_idx, 1.f);

        this->gaussians_->createFromPcd(this->dense_init_pcd_xyz_, this->dense_init_pcd_rgb_, this->dense_init_pcd_idx_, this->scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(this->mutex_settings_);
        this->gaussians_->trainingSetup(opt_params_);

        if(this->kf_params_.debug_)
            this->savePly(this->result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "ply_lr_init", false);
    }
    // {
    //     std::unique_lock<std::mutex> lock_render(this->mutex_render_);
    //     torch::NoGradGuard no_grad;
    //     auto& pkf_first = this->scene_->keyframes().at(0);
    //     std::cout<<"[debug] test pose:\n"<<pkf_first->getBasePose()<<std::endl;
    //     auto render_pkg = GaussianRenderer::render(
    //         pkf_first,
    //         this->kf_params_.lr_height_,
    //         this->kf_params_.lr_width_,
    //         this->gaussians_,
    //         this->pipe_params_,
    //         this->background_,
    //         this->override_color_,
    //         false, true, true, true
    //     );
    //     auto rendered_image = std::get<0>(render_pkg);

    //     auto rendered_image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
    //     cv::cvtColor(rendered_image_cv, rendered_image_cv, CV_RGB2BGR);
    //     rendered_image_cv.convertTo(rendered_image_cv, CV_8UC3, 255.0f);
    //     CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "lr_test"))
    //     cv::imwrite(result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "lr_test" / (std::to_string(pkf_first->fid_) + "_render.jpg"), rendered_image_cv);

    //     auto ones_tensor = torch::ones_like(rendered_image);
    //     auto diff_image = torch::abs(rendered_image - ones_tensor);
    //     std::cout<<"[debug] diff mean "<<diff_image.mean().item<float>()<<std::endl;
    //     auto diff_image_cv = tensor_utils::torchTensor2CvMat_Float32(diff_image);
    //     cv::cvtColor(diff_image_cv, diff_image_cv, CV_RGB2BGR);
    //     diff_image_cv.convertTo(diff_image_cv, CV_8UC3, 255.0f);
    //     cv::imwrite(result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "lr_test" / (std::to_string(pkf_first->fid_) + "_diff.jpg"), diff_image_cv);  

    //     throw std::runtime_error("[GaussianMapper::run_cuvslam] debug throw after initial render!");
    // }

    // test cache all keyframe depth map
    // this->testCacheAllKeyframeDepthMap();

    // compute initial global alignment pose and timestamp
    this->optimizeGlobalAlign();

    if(this->kf_params_.debug_)
        this->savePly(this->result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "ply_lr_aligned", false);
    
    // Second loop: Incremental gaussian mapping
    while(!this->isStopped()){
        std::vector<std::size_t> tracking_ids, mapping_ids;
        this->pCuVSLAM_->GetFrameIds(tracking_ids);
        this->scene_->getKeyframeIds(mapping_ids);

        if(tracking_ids.size() > mapping_ids.size() + this->local_align_batch_size_){
            std::vector<std::size_t> diff_ids;
            std::set_difference(tracking_ids.begin(), tracking_ids.end(), mapping_ids.begin(), mapping_ids.end(), std::back_inserter(diff_ids));
            
            if(diff_ids.size() < this->local_align_batch_size_){
                std::cerr << "[GaussianMapper::run_cuvslam] Warning: Expected new frames but the number of new frames (" << diff_ids.size() << ") is less than batch size (" << this->local_align_batch_size_ << ")." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            std::cout<<"[GaussianMapper::run_cuvslam] Inserting new batch of keyframes, id range ["<<diff_ids.front()<<", "<<diff_ids.back()<<"], size "<<diff_ids.size()<<std::endl;
            std::vector<std::size_t> batch_ids(diff_ids.begin(), diff_ids.begin() + this->local_align_batch_size_);
            this->insertBatchKeyframes(batch_ids);
        }
        else{
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
    }

}

float GaussianMapper::optimizeGlobalLRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose)
{
    torch::Tensor loss;
    if(use_differential_pose){
        for(int i=0; i<this->global_align_lr_pose_iter_; i++){
            auto render_pkg = GaussianRenderer::render(
                pkf,
                this->kf_params_.lr_height_,
                this->kf_params_.lr_width_,
                this->gaussians_,
                this->pipe_params_,
                this->background_,
                this->override_color_,
                false, true, true, true
            );

            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);

            auto gt_image = pkf->getGTLRImg(true);
            auto gt_depth = pkf->getGTLRDpt(true);
            auto gt_depth_mask = pkf->getGTLRDptMsk(true); 

            auto loss = loss_utils::get_loss_rgbd(
                rendered_image, gt_image,
                rendered_depth, gt_depth,
                this->lambdaDssim(),
                this->global_align_lr_depth_lambda_,
                pkf->exposure_a_, pkf->exposure_b_,
                gt_depth_mask,
                device_type_
            );

            loss.backward();

            {
                torch::NoGradGuard no_grad;

                gaussians_->optimizer_->zero_grad(true);
                pkf->stepOptimizer(true);
                pkf->zeroOptimizerGrad(true, true);

                pkf->updateBasePose();

                if(i % 10 == 0) 
                    pkf->updateOptimizer(0.5f);

                if(kf_params_.debug_ && (i==this->global_align_lr_pose_iter_-1 || i==0)){
                    auto masked_gt_image = gt_image * gt_depth_mask;
                    auto masked_gt_depth = gt_depth * gt_depth_mask;
                    auto masked_rendered_image = rendered_image * gt_depth_mask;
                    auto masked_rendered_depth = rendered_depth * gt_depth_mask;
                    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose"))
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                    std::cout<<"[debug] lr pose optim fid "<<pkf->fid_<<" iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }
            }
        }
    }
    else{

        float confidence = -1.f;
        int render_attempts = 0;
        while(confidence <= 0.f && render_attempts < this->max_pnp_render_attempts_){
            torch::Tensor rendered_image, rendered_depth, gt_depth, gt_image, gt_depth_mask, valid_mask;
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                torch::NoGradGuard no_grad;

                auto render_pkg = GaussianRenderer::render(pkf,
                    this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    false, true, true, true
                );

                rendered_image = std::get<0>(render_pkg);
                rendered_depth = std::get<4>(render_pkg);
                auto rendered_opacity = std::get<5>(render_pkg);

                auto opacity_mask = (rendered_opacity > 0.5f).to(torch::kFloat32).squeeze();
                std::cout<<"[debug] rendered opacity mask sum "<<opacity_mask.sum().item<float>()<<std::endl;

                gt_image = pkf->getGTLRImg(true);
                gt_depth = pkf->getGTLRDpt(true);
                gt_depth_mask = pkf->getGTLRDptMsk(true).to(torch::kFloat32); 

                valid_mask = opacity_mask.squeeze() * gt_depth_mask.squeeze();

                loss = loss_utils::get_loss_rgbd(
                    rendered_image, gt_image,
                    rendered_depth, gt_depth,
                    this->lambdaDssim(),
                    this->global_align_lr_depth_lambda_,
                    pkf->exposure_a_, pkf->exposure_b_,
                    valid_mask,
                    device_type_
                );

                rendered_image = valid_mask.unsqueeze(0) * rendered_image;
            }

            if(kf_params_.debug_){
                auto masked_gt_image = gt_image * valid_mask;
                auto masked_gt_depth = gt_depth * valid_mask;
                auto masked_rendered_image = rendered_image * valid_mask;
                auto masked_rendered_depth = rendered_depth * valid_mask;
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_before.jpg"), image_cv);
                std::cout<<"[debug] lr pose optim fid "<<pkf->fid_<<" final loss "<<loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }

            std::vector<cv::KeyPoint> kpts_rendered, kpts_gt;
            std::vector<int> matches_rendered2gt, matches_gt2rendered;
            int good_matches = pkf->matchLROrbGMS(rendered_image, 
                kpts_rendered, kpts_gt,
                matches_rendered2gt, matches_gt2rendered
            );

            std::cout<<"[GaussianMapper::optimizeGlobalLRPose] classic pose fid "<<pkf->fid_<<" found "<<good_matches<<" good matches for GMS pose estimation."<<std::endl;

            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            cv::Mat K_lr = (cv::Mat_<float>(3,3) << 
                kf_params_.lr_fx_, 0.f, kf_params_.lr_cx_,
                0.f, kf_params_.lr_fy_, kf_params_.lr_cy_,
                0.f, 0.f, 1.f
            );

            auto depth_cpu = gt_depth.squeeze().to(torch::kCPU).contiguous();
            cv::Mat depth_cv(depth_cpu.size(0), depth_cpu.size(1), CV_32F);
            std::memcpy(depth_cv.data, depth_cpu.data_ptr<float>(), sizeof(float)*depth_cpu.size(0)*depth_cpu.size(1));

            confidence = this->getRelatedPoseGMS(
                pkf,
                kpts_gt, kpts_rendered,
                matches_gt2rendered,
                depth_cv,
                K_lr,
                rvec, tvec,
                inliers
            );

            if(confidence <= 0.f){
                render_attempts++;
                std::cout<<"[warning][GaussianMapper::optimizeGlobalLRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed, re-rendering "<<render_attempts<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
                continue;
            }

            auto delta_pose = general_utils::vecs2transformation(rvec, tvec);
            auto delta_pose_torch = torch::from_blob(delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
            delta_pose_torch = delta_pose_torch.inverse();

            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                auto updated_base_pose = delta_pose_torch.mm(pkf->getBasePose());
                std::cout<<"[debug 02201235] delta_pose_torch "<<delta_pose_torch<<" id "<<pkf->fid_<<std::endl;
                pkf->setBasePose(updated_base_pose);
            }

            if(kf_params_.debug_){
                auto rendered_image_before = rendered_image.clone();
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    torch::NoGradGuard no_grad;
                    auto render_pkg = GaussianRenderer::render(
                        pkf,
                        this->kf_params_.lr_height_,
                        this->kf_params_.lr_width_,
                        this->gaussians_,
                        this->pipe_params_,
                        this->background_,
                        this->override_color_,
                        false, true, true, true
                    );

                    rendered_image = std::get<0>(render_pkg);
                    rendered_depth = std::get<4>(render_pkg);
                    auto rendered_opacity = std::get<5>(render_pkg);

                    auto opacity_mask = (rendered_opacity > 0.5f).to(torch::kFloat32).squeeze();
                    valid_mask = opacity_mask * gt_depth_mask.squeeze();
                }

                rendered_image = valid_mask.unsqueeze(0) * rendered_image;

                auto masked_gt_image = gt_image * valid_mask;
                auto masked_gt_depth = gt_depth * valid_mask;
                auto masked_rendered_image = rendered_image * valid_mask;
                auto masked_rendered_depth = rendered_depth * valid_mask;

                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_after.jpg"), image_cv);
                auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
                cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                std::cout<<"[debug] lr pose optim fid "<<pkf->fid_<<" final loss "<<loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                auto diff_image = torch::abs(rendered_image_before - rendered_image);
                auto diff_image_cv = tensor_utils::torchTensor2CvMat_Float32(diff_image);
                cv::cvtColor(diff_image_cv, diff_image_cv, CV_RGB2BGR);
                diff_image_cv.convertTo(diff_image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_diff.jpg"), diff_image_cv); 
            }
        }

        if(confidence <= 0.f){
            std::cout<<"[error][GaussianMapper::optimizeGlobalLRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed after "<<this->max_pnp_render_attempts_<<" attempts!"<<std::endl;
            throw std::runtime_error("[GaussianMapper::optimizeGlobalLRPose] PnP pose estimation failed!");
        }

    }

    return loss.item<float>();
}

float GaussianMapper::optimizeGlobalHRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose)
{
    torch::Tensor loss;
    int hr_height_resize, hr_width_resize;
    float hr_resize_ratio = this->getRsizedHRScale(this->global_align_hr_resize_ratio_, hr_width_resize, hr_height_resize);

    if(use_differential_pose){
        pkf->resetOptimizer(true, kf_params_.hr_color_theta_lr_, kf_params_.hr_color_rho_lr_);
        pkf->resetFullExposure();
        torch::Tensor opacity_mask;
        for(int i=0; i<this->global_align_hr_pose_iter_; i++){
            auto render_pkg = GaussianRenderer::render(pkf,
                hr_height_resize, hr_width_resize,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, true, true, true
            );

            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);

            if(i==0) opacity_mask = (rendered_opacity > this->global_align_hr_opacity_thr_).to(torch::kFloat32);
                
            auto gt_image = pkf->getGTHRImg(hr_resize_ratio, true);

            auto loss_rgb = loss_utils::get_loss_rgb(rendered_image, gt_image,
                this->lambdaDssim(), pkf->exposure_a_, pkf->exposure_b_, opacity_mask);

            loss = loss_rgb;
            loss.backward();
                
            {
                torch::NoGradGuard no_grad;

                // gaussians_->optimizer_->step();
                gaussians_->optimizer_->zero_grad(true); 

                pkf->stepOptimizer(true, true);
                pkf->zeroOptimizerGrad(true, true);

                pkf->updateLocalDeltaPose(true);

                if(i==this->global_align_hr_pose_iter_/2)
                    pkf->updateOptimizer(0.7f);

                if(kf_params_.debug_ && (i==this->global_align_hr_pose_iter_-1 || i==0)){
                    auto masked_gt_image = gt_image * opacity_mask;
                    auto masked_rendered_image = rendered_image * opacity_mask;
                    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_train_pose"))
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_ ) / "hr_train_pose" / (std::to_string(pkf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                    std::cout<<"[debug] hr pose optim fid "<<pkf->fid_<<" iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }
            }

        }
    }
    else{
        hr_resize_ratio = this->getRsizedHRScale(1.f, hr_width_resize, hr_height_resize);
        torch::Tensor rendered_image, rendered_depth, opacity_mask, gt_image;
        torch::Tensor local_delta_pose1, local_delta_pose2;
        torch::Tensor updated_loss, final_loss;

        cv::Mat K_hr = (cv::Mat_<float>(3,3) <<
            kf_params_.hr_fx_ * hr_resize_ratio, 0.f, kf_params_.hr_cx_ * hr_resize_ratio,
            0.f, kf_params_.hr_fy_ * hr_resize_ratio, kf_params_.hr_cy_ * hr_resize_ratio,
            0.f, 0.f, 1.f
        );

        float confidence = -1.f;
        int render_attempts = 0;
        while(confidence <= 0.f && render_attempts < this->max_pnp_render_attempts_){
            {
                torch::NoGradGuard no_grad;

                auto render_hr_pkg = GaussianRenderer::render(
                    pkf,
                    hr_height_resize, hr_width_resize,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    true, true, true, true
                );

                rendered_image = std::get<0>(render_hr_pkg);
                rendered_depth = std::get<4>(render_hr_pkg);
                auto rendered_opacity = std::get<5>(render_hr_pkg);

                opacity_mask = (rendered_opacity > this->global_align_hr_opacity_thr_).to(torch::kFloat32).squeeze();
                
                gt_image = pkf->getGTHRImg(hr_resize_ratio, true).cuda();

                loss = loss_utils::get_loss_rgb(
                    rendered_image, gt_image,
                    this->lambdaDssim(),
                    pkf->exposure_a_, pkf->exposure_b_,
                    opacity_mask,
                    device_type_
                );
            }

            if(kf_params_.debug_){
                auto masked_gt_image = gt_image * opacity_mask;
                auto masked_rendered_image = rendered_image * opacity_mask;
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_ ) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_before.jpg"), image_cv);
                auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
                cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_ ) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                std::cout<<"[debug] hr pose optim fid "<<pkf->fid_<<" final loss "<<loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
            
            std::vector<cv::KeyPoint> kpts_rendered, kpts_gt;
            std::vector<int> matches_rendered2gt, matches_gt2rendered;
            int good_matches = pkf->matchHROrbGMS(rendered_image, hr_resize_ratio,
                kpts_rendered, kpts_gt,
                matches_rendered2gt, matches_gt2rendered
            );

            auto depth_cpu = rendered_depth.squeeze().to(torch::kCPU).contiguous();
            cv::Mat depth_cv(depth_cpu.size(0), depth_cpu.size(1), CV_32F);
            std::memcpy(depth_cv.data, depth_cpu.data_ptr<float>(), sizeof(float)*depth_cpu.size(0)*depth_cpu.size(1));

            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            confidence = this->getRelatedPoseGMS(
                pkf,
                kpts_rendered, kpts_gt,
                matches_rendered2gt,
                depth_cv,
                K_hr,
                rvec, tvec,
                inliers,
                false
            );
            
            if(confidence > 0.f){
                std::cout<<"[GaussianMapper::optimizeGlobalHRPose] rendered RGBD pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                    <<" inliers "<<inliers.size()<<"/"<<good_matches<<std::endl;        
                auto delta_pose = general_utils::vecs2transformation(rvec, tvec);
                auto delta_pose_torch = torch::from_blob(delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);

                auto updated_local_pose = delta_pose_torch.mm(pkf->getLocalDeltaPose());
                local_delta_pose1 = pkf->getLocalDeltaPose().clone();
                pkf->setLocalDeltaPose(updated_local_pose);
                local_delta_pose2 = pkf->getLocalDeltaPose().clone();
                break;
            }
            else{
                // std::cout<<"[GaussianMapper::optimizeGlobalHRPose] rendered RGBD HR pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                //     <<" inliers "<<inliers.size()<<"/"<<good_matches<<", skip pose update."<<std::endl;
                std::cout<<"[warning][GaussianMapper::optimizeGlobalHRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed, re-rendering "<<render_attempts<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
                render_attempts++;
                // return loss.item<float>();
            }
        }

        if(confidence <= 0.f){
            std::cout<<"[error][GaussianMapper::optimizeGlobalHRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed after "<<this->max_pnp_render_attempts_<<" attempts!"<<std::endl;
            // throw std::runtime_error("[GaussianMapper::optimizeGlobalHRPose] PnP pose estimation failed!");
            return loss.item<float>();
        }
        else{
            confidence = -1.f;
            render_attempts = 0;
        }

        while(confidence <= 0.f && render_attempts < this->max_pnp_render_attempts_){
            {
                torch::NoGradGuard no_grad;

                auto render_hr_pkg = GaussianRenderer::render(
                    pkf,
                    hr_height_resize, hr_width_resize,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    true, true, true, true
                );

                rendered_image = std::get<0>(render_hr_pkg);
                rendered_depth = std::get<4>(render_hr_pkg);
                auto rendered_opacity = std::get<5>(render_hr_pkg);

                opacity_mask = (rendered_opacity > this->global_align_hr_opacity_thr_).to(torch::kFloat32).squeeze();
                
                updated_loss = loss_utils::get_loss_rgb(
                    rendered_image, gt_image,
                    this->lambdaDssim(),
                    pkf->exposure_a_, pkf->exposure_b_,
                    opacity_mask,
                    device_type_
                );

                if(updated_loss.item<float>() > loss.item<float>()){
                    std::cout<<"[GaussianMapper::optimizeGlobalHRPose] rendered RGBD HR pose update increases loss from "<<loss.item<float>()<<" to "<<updated_loss.item<float>()<<", revert pose update."<<std::endl;
                    pkf->setLocalDeltaPose(local_delta_pose1);
                }
            }    

            if(kf_params_.debug_){
                auto masked_gt_image = gt_image * opacity_mask;
                auto masked_rendered_image = rendered_image * opacity_mask;
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_ ) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_after.jpg"), image_cv);
                std::cout<<"[debug] hr pose optim fid "<<pkf->fid_<<" final loss "<<updated_loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
            
            std::vector<cv::KeyPoint> kpts_rendered, kpts_gt;
            std::vector<int> matches_rendered2gt, matches_gt2rendered;
            int good_matches = pkf->matchHROrbGMS(rendered_image, hr_resize_ratio,
                kpts_rendered, kpts_gt,
                matches_rendered2gt, matches_gt2rendered
            );

            auto depth_cpu = rendered_depth.squeeze().to(torch::kCPU).contiguous();
            cv::Mat depth_cv(depth_cpu.size(0), depth_cpu.size(1), CV_32F);
            std::memcpy(depth_cv.data, depth_cpu.data_ptr<float>(), sizeof(float)*depth_cpu.size(0)*depth_cpu.size(1));

            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            confidence = this->getRelatedPoseGMS(
                pkf,
                kpts_rendered, kpts_gt,
                matches_rendered2gt,
                depth_cv,
                K_hr,
                rvec, tvec,
                inliers,
                false,
                hr_resize_ratio,
                true
            );

            if(confidence > 0.f){
                std::cout<<"[GaussianMapper::optimizeGlobalHRPose] HRLR cross RGBD refined pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                    <<" inliers "<<inliers.size()<<"/"<<good_matches
                    <<" loss before "<<loss.item<float>()
                    <<" loss after "<<updated_loss.item<float>()<<std::endl;     
                auto delta_pose = general_utils::vecs2transformation(rvec, tvec);
                auto delta_pose_torch = torch::from_blob(delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
                
                auto updated_local_pose = delta_pose_torch.mm(pkf->getLocalDeltaPose());
                pkf->setLocalDeltaPose(updated_local_pose);
                break;
            }
            else{
                // std::cout<<"[GaussianMapper::optimizeGlobalHRPose] HRLR cross RGBD refined pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                //     <<" inliers "<<inliers.size()<<"/"<<good_matches<<", skip pose update."<<std::endl;
                std::cout<<"[warning][GaussianMapper::optimizeGlobalHRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed, re-rendering "<<render_attempts<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
                render_attempts++;
                // return updated_loss.item<float>();
            }
        }
        
        if(confidence <= 0.f){
            std::cout<<"[error][GaussianMapper::optimizeGlobalHRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed after "<<this->max_pnp_render_attempts_<<" attempts!"<<std::endl;
            // throw std::runtime_error("[GaussianMapper::optimizeGlobalHRPose] PnP pose estimation failed!");
            return updated_loss.item<float>();
        }

        {
            torch::NoGradGuard no_grad;

            auto render_hr_pkg = GaussianRenderer::render(
                pkf,
                hr_height_resize, hr_width_resize,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, true, true, true
            );

            rendered_image = std::get<0>(render_hr_pkg);
            rendered_depth = std::get<4>(render_hr_pkg);
            auto rendered_opacity = std::get<5>(render_hr_pkg);

            opacity_mask = (rendered_opacity > this->global_align_hr_opacity_thr_).to(torch::kFloat32).squeeze();
                
            final_loss = loss_utils::get_loss_rgb(
                rendered_image, gt_image,
                this->lambdaDssim(),
                pkf->exposure_a_, pkf->exposure_b_,
                opacity_mask,
                device_type_
            );

            if(final_loss.item<float>() > updated_loss.item<float>()){
                std::cout<<"[GaussianMapper::optimizeGlobalHRPose] HRLR cross RGBD HR refined pose update increases loss from "<<updated_loss.item<float>()<<" to "<<final_loss.item<float>()<<", revert pose update."<<std::endl;
                pkf->setLocalDeltaPose(local_delta_pose2);
            }

            if(kf_params_.debug_){
                auto masked_gt_image = gt_image * opacity_mask;
                auto masked_rendered_image = rendered_image * opacity_mask;
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_ ) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_final.jpg"), image_cv);
                std::cout<<"[debug] hr pose optim fid "<<pkf->fid_<<" final loss "<<final_loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
        }

    }

    return loss.item<float>();
}

float GaussianMapper::optimizeGlobalLRImg(std::shared_ptr<GaussianKeyframe> pkf, int i){
    torch::Tensor gt_image, gt_depth, gt_depth_mask, rendered_image, rendered_depth, loss;

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // 1st 1/3: sh=1, 2nd 1/3: sh=2, last 1/3: sh=3 
        gaussians_->setShDegree(int(
            float(i) / float(this->global_align_lr_iter_) * float(gaussians_->max_sh_degree_) + 1
        ));

        auto render_pkg = GaussianRenderer::render(
            pkf,
            this->kf_params_.lr_height_,
            this->kf_params_.lr_width_,
            this->gaussians_,
            this->pipe_params_,
            this->background_,
            this->override_color_,
            false
        );

        rendered_image = std::get<0>(render_pkg);
        // auto viewspace_point_tensor = std::get<1>(render_pkg);
        // auto visibility_filter = std::get<2>(render_pkg);
        rendered_depth = std::get<4>(render_pkg);

        gt_image = pkf->getGTLRImg(true);
        gt_depth = pkf->getGTLRDpt(true);
        gt_depth_mask = pkf->getGTLRDptMsk(true); 

        loss = loss_utils::get_loss_rgbd(
            rendered_image, gt_image,
            rendered_depth, gt_depth,
            this->lambdaDssim(),
            this->global_align_lr_depth_lambda_,
            pkf->exposure_a_, pkf->exposure_b_,
            gt_depth_mask,
            device_type_
        );

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true); 

            pkf->zeroOptimizerGrad(true, true);
        }
    }

    if(kf_params_.debug_ && (i==this->global_align_lr_iter_-1 || i==0)){
        auto masked_gt_image = gt_image * gt_depth_mask;
        auto masked_gt_depth = gt_depth * gt_depth_mask;
        auto masked_rendered_image = rendered_image * gt_depth_mask;
        auto masked_rendered_depth = rendered_depth * gt_depth_mask;

        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_img"))
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_img" / (std::to_string(pkf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
        auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
        cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
        gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "lr_train_img" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
        std::cout<<"[debug] lr img optim fid "<<pkf->fid_<<" iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
        metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
    }

    return loss.item<float>();
}

void GaussianMapper::optimizeGlobalAlign(){
    // std::unique_lock<std::mutex> lock_render(mutex_render_);

    auto iter_start_timing1 = std::chrono::steady_clock::now();
    std::vector<std::size_t> keyframes_ids;
    this->scene_->getKeyframeIds(keyframes_ids);
    this->local_mapping_batch_ids_[0] = keyframes_ids;

    std::cout<<"[GaussianMapper::optimizeGlobalAlign] step0"<<std::endl;
    // 0. optim 3 init keyframes pose 
    for(int id = 0; id<keyframes_ids.size(); id++){
    // for(int id = 0; id<this->dense_init_fids_.size(); id++){
        auto& pkf = scene_->keyframes().at(keyframes_ids.at(id));
        // auto& pkf = scene_->keyframes().at(this->dense_init_fids_.at(id));
        this->optimizeGlobalLRPose(pkf, false); // [test20260213] disable LR pnp
    }

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);    
        for(int id = 0; id<keyframes_ids.size(); id++)
        // for(int id = 0; id<this->dense_init_fids_.size(); id++)
            scene_->keyframes().at(keyframes_ids.at(id))->resetOptimizer();
            // scene_->keyframes().at(this->dense_init_fids_.at(id))->resetOptimizer();
        gaussians_->resetOptimizer(opt_params_); // !!! notes try to add scale 1.2 for sequence `wall`
    }

    std::cout<<"[GaussianMapper::optimizeGlobalAlign] step1"<<std::endl;
    // 1. converge 3 init keyframes
    for(int i=0; i<this->global_align_lr_iter_; i++)
    for(int id = 0; id<this->dense_init_fids_.size(); id++){
        auto& pkf = scene_->keyframes().at(this->dense_init_fids_.at(id));
        this->optimizeGlobalLRImg(pkf, i);
    }

    // return; // [test20260213] disable HR

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        for(int id = 0; id<this->dense_init_fids_.size(); id++)
            scene_->keyframes().at(this->dense_init_fids_.at(id))->resetOptimizer();
        gaussians_->resetOptimizer(opt_params_);
    }

    auto iter_start_timing2 = std::chrono::steady_clock::now();

    std::cout<<"[GaussianMapper::optimizeGlobalAlign] step2"<<std::endl;
    // 2. estimate global time offset
    int hr_height_resize, hr_width_resize;
    float hr_resize_ratio;
    if(this->global_align_hr_resize_ratio_>0.99999f){
        hr_height_resize = this->kf_params_.hr_height_;
        hr_width_resize = this->kf_params_.hr_width_;
        hr_resize_ratio = 1.f;
    }
    else{
        hr_height_resize = int(floor(float(this->kf_params_.hr_height_) * this->global_align_hr_resize_ratio_));
        hr_width_resize = int(floor(float(this->kf_params_.hr_width_) * this->global_align_hr_resize_ratio_));
        hr_resize_ratio = this->global_align_hr_resize_ratio_;
    }

    std::size_t test_frames = 7; // debug number, should move to config
    int global_time_align_frame_num = std::min(keyframes_ids.size(), test_frames);
    std::vector<std::size_t> global_time_align_fids;

    for (size_t i = 0; i < global_time_align_frame_num; i++) {
        float pos = (float(i) / (global_time_align_frame_num - 1)) * (keyframes_ids.size() - 1);
        int idx = int(std::round(pos));

        if (idx >= static_cast<int>(keyframes_ids.size())) {
            idx = static_cast<int>(keyframes_ids.size()) - 1;
        } else if (idx < 0) {
            idx = 0;
        }

        global_time_align_fids.push_back(keyframes_ids[idx]);
    }

    global_time_align_fids.erase(
        std::unique(global_time_align_fids.begin(), global_time_align_fids.end()), 
        global_time_align_fids.end()
    );
    std::cout<<"[debug] selected keyframes for global time alignment: ";
    for(auto fid: global_time_align_fids) std::cout<<fid<<" ";
    std::cout<<std::endl;

    // t_window = (t_lr0 + id_lrN / lr_fps) * hr_fps * ratio_window
    auto& pkf0 = scene_->keyframes().at(keyframes_ids[0]);
    float hr_window_size = (pkf0->lr_timestamp_ + float(global_time_align_fids[global_time_align_fids.size()-1])/kf_params_.lr_fps_ ) * kf_params_.hr_fps_ * this->global_align_time_window_ratio_;
    // size_t hr_start_fid = int(pkf0->lr_timestamp_ * kf_params_.hr_fps_);
    std::cout<<"[debug] hr_window_size "<<hr_window_size<<std::endl;

    std::vector<torch::Tensor> hr_gts;
    for(int i=0; i<hr_window_size; i++){
        auto hr_undist_mat = general_utils::readRGB2cvMat(this->vstrHRImagePaths_[i], 
            true, kf_params_.hr_undistort_map1_, kf_params_.hr_undistort_map2_);
        auto hr_undist_tensor = tensor_utils::cvMat2TorchTensor_Float32(hr_undist_mat, torch::kCPU);
        if(hr_resize_ratio<1.f-1e-5f)
            hr_undist_tensor = torch::nn::functional::interpolate(
                hr_undist_tensor.unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>({hr_height_resize, hr_width_resize}))
                    .mode(torch::kNearest)
            ).squeeze(0);
        hr_gts.push_back(hr_undist_tensor);
    }

    if(kf_params_.debug_){
        std::cout<<"[debug] global time align fids: ";
        for(auto fid: global_time_align_fids) std::cout<<fid<<" ";
        std::cout<<std::endl;
    }

    std::vector<std::vector<float>> all_kf_losses(global_time_align_frame_num);
    std::vector<torch::Tensor> kf_rendered_imgs(global_time_align_frame_num);
    std::vector<size_t> kf_hr_fids(global_time_align_frame_num);

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        torch::NoGradGuard no_grad;
        for (size_t k = 0; k < global_time_align_frame_num; k++) {
            auto& pkf = scene_->keyframes().at(global_time_align_fids[k]);
            auto render_pkg = GaussianRenderer::render(
                pkf,
                hr_height_resize, hr_width_resize,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, true
            );
            kf_rendered_imgs[k] = std::get<0>(render_pkg);
        }
    }

    std::vector<bool> valid_match(global_time_align_frame_num, false);
    {
        torch::NoGradGuard no_grad;
        std::size_t search_start = 0;

        for (std::size_t k = 0; k < global_time_align_frame_num; k++) {
            std::cout<<"[debug] search start id: "<<search_start<<" for keyframe "<<k<<" (fid "<<global_time_align_fids[k]<<")\n";
            auto& rendered_img = kf_rendered_imgs[k];
            auto& cur_loss = all_kf_losses[k];

            if (search_start >= hr_gts.size()) {
                std::cout << "[warn] Keyframe " << k << " skipped: HR frames exhausted.\n";
                break;  // stop further matching
            }

            for (std::size_t i = search_start; i < hr_gts.size(); i++) {
                auto gt_image = hr_gts[i].cuda();
                float loss_val = loss_utils::l1_loss(rendered_img, gt_image).item<float>();
                cur_loss.push_back(loss_val);
            }

            // find best match
            std::size_t local_min_idx = std::distance(
                cur_loss.begin(),
                std::min_element(cur_loss.begin(), cur_loss.end()));
            std::size_t matched_hr_idx = search_start + local_min_idx;

            if (matched_hr_idx >= hr_gts.size() - 1) {
                std::cout << "[warn] Keyframe " << k << " match index out of HR range ("
                        << matched_hr_idx << " / " << hr_gts.size()
                        << "), skipping.\n";
                break;  // skip remaining frames as HR window exhausted
            }

            kf_hr_fids[k] = matched_hr_idx;
            valid_match[k] = true;

            search_start = matched_hr_idx + 1; // progress window forward
        }

        for (auto& t : hr_gts) t = torch::Tensor();
    }

    std::vector<float> delta_times;
    delta_times.reserve(global_time_align_frame_num);

    for (size_t k = 0; k < global_time_align_frame_num; k++) {
        if (!valid_match[k]) continue; // skip unusable ones

        auto& pkf = scene_->keyframes().at(global_time_align_fids[k]);
        float dt = float(kf_hr_fids[k]) / this->kf_params_.hr_fps_ - pkf->lr_timestamp_;
        delta_times.push_back(dt);

        if (kf_params_.debug_) {
            std::cout << "[debug] kf" << k << " delta_t: " << dt << std::endl;
        }
    }

    if (delta_times.empty()) {
        throw std::runtime_error("[GaussianMapper::optimizeGlobalAlign] No valid HR matches found for global alignment!");
        // std::cout << "[warn] No valid HR matches found — global alignment disabled.\n";
        // this->global_align_time_ = 0.0f;
    } else {
        this->global_align_time_ =
            std::accumulate(delta_times.begin(), delta_times.end(), 0.f) / delta_times.size();

        std::cout << "[debug] global_align_time " << this->global_align_time_
                  << " (computed from " << delta_times.size() << " valid matches)\n";
    }

    /*
    std::vector<float> kf0_loss, kf1_loss, kf2_loss;
    auto& pkf0 = scene_->keyframes().at(this->dense_init_fids_[0]);
    auto& pkf1 = scene_->keyframes().at(this->dense_init_fids_[1]);
    auto& pkf2 = scene_->keyframes().at(this->dense_init_fids_[2]);

    torch::Tensor kf0_rendered, kf1_rendered, kf2_rendered;
    size_t kf0_hr_fid, kf1_hr_fid, kf2_hr_fid;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        torch::NoGradGuard no_grad;
        auto render_pkg0 = GaussianRenderer::render(pkf0,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, true
        );
        auto render_pkg1 = GaussianRenderer::render(pkf1,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, true
        );
        auto render_pkg2 = GaussianRenderer::render(pkf2,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, true
        );

        kf0_rendered = std::get<0>(render_pkg0);
        kf1_rendered = std::get<0>(render_pkg1);
        kf2_rendered = std::get<0>(render_pkg2);
    }
    {
        torch::NoGradGuard no_grad;
        for(int i=0; i<hr_gts.size(); i++){
            auto gt_image = hr_gts[i].cuda();
            auto loss = loss_utils::l1_loss(kf0_rendered, gt_image);
            kf0_loss.push_back(loss.item<float>());
            // std::cout<<"[debug] kf0 "<<i<<" "<<loss.item<float>()<<std::endl;
        }
    
        kf0_hr_fid = std::distance(kf0_loss.begin(), std::min_element(kf0_loss.begin(), kf0_loss.end()));
        for(int i=kf0_hr_fid-1; i>=0; i--) hr_gts[i] = torch::Tensor(); // release memory
        
        for(int i=kf0_hr_fid; i<hr_gts.size(); i++){
            auto gt_image = hr_gts[i].cuda();
            auto loss = loss_utils::l1_loss(kf1_rendered, gt_image);
            kf1_loss.push_back(loss.item<float>());
            // std::cout<<"[debug] kf1 "<<i<<" "<<loss.item<float>()<<std::endl;
        }
    
        kf1_hr_fid = 1 + kf0_hr_fid + std::distance(kf1_loss.begin(), std::min_element(kf1_loss.begin(), kf1_loss.end()));
        for(int i=kf1_hr_fid-1; i>=0; i--) hr_gts[i] = torch::Tensor(); // release memory

        for(int i=kf1_hr_fid; i<hr_gts.size(); i++){
            auto gt_image = hr_gts[i].cuda();
            auto loss = loss_utils::l1_loss(kf2_rendered, gt_image);
            kf2_loss.push_back(loss.item<float>());
            // std::cout<<"[debug] kf1 "<<i<<" "<<loss.item<float>()<<std::endl;
        }

        kf2_hr_fid = 1 + kf1_hr_fid + std::distance(kf2_loss.begin(), std::min_element(kf2_loss.begin(), kf2_loss.end()));
    
        for(int i=0; i<hr_gts.size(); i++) hr_gts[i] = torch::Tensor(); // release memory

        if(kf_params_.debug_){
            std::cout<<"[debug] kf0 hr fid "<<kf0_hr_fid<<" loss "<<kf0_loss[kf0_hr_fid]<<std::endl;
            std::cout<<"[debug] kf1 hr fid "<<kf1_hr_fid<<" loss "<<kf1_loss[kf1_hr_fid - kf0_hr_fid - 1]<<std::endl;
            std::cout<<"[debug] kf2 hr fid "<<kf2_hr_fid<<" loss "<<kf2_loss[kf2_hr_fid - kf1_hr_fid - 1]<<std::endl;
        }
    }

    float f0_delta_time = float(kf0_hr_fid) / this->kf_params_.hr_fps_ - pkf0->lr_timestamp_;
    float f1_delta_time = float(kf1_hr_fid) / this->kf_params_.hr_fps_ - pkf1->lr_timestamp_;
    float f2_delta_time = float(kf2_hr_fid) / this->kf_params_.hr_fps_ - pkf2->lr_timestamp_;
    std::cout<<"[debug] kf0 delta time "<<f0_delta_time<<std::endl;
    std::cout<<"[debug] kf1 delta time "<<f1_delta_time<<std::endl;
    std::cout<<"[debug] kf2 delta time "<<f2_delta_time<<std::endl; 

    this->global_align_time_ = (f0_delta_time + f1_delta_time + f2_delta_time) / 3.f;
    std::cout<<"[debug] estimated global align time offset "<<this->global_align_time_<<std::endl;
    */

    for(int id = 0; id<keyframes_ids.size(); id++){
        auto& pkf = this->scene_->keyframes().at(keyframes_ids.at(id));
        bool has_hr_fid = false;
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            has_hr_fid = pkf->setGTHRImg(this->global_align_time_, this->vstrHRImagePaths_);
        }
        if(!has_hr_fid) continue;

        if(kf_params_.debug_){
            std::cout<<"[debug] final match [lr_id] "<<pkf->fid_<<" [lr_time] "<<pkf->lr_timestamp_<<" [est hr_id] "<<pkf->hr_fid_<<" [est hr time] "<<(pkf->lr_timestamp_ + this->global_align_time_)<<" [gt hr time] "<<this->vHRTimestamps_[pkf->hr_fid_]<<std::endl;

            torch::Tensor linear_indices;
            auto gt_image = pkf->getGTHRImg(linear_indices, hr_resize_ratio);

            torch::Tensor white = torch::ones({gt_image.size(1), gt_image.size(2), 3}, 
                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

            torch::Tensor flat_gt = white.reshape({3, -1}).permute({1, 0});

            std::cout<<"[debug] flat_gt sizes "<<flat_gt.sizes()<<" linear_indices sizes "<<linear_indices.sizes()<<" max "<<linear_indices.max().item<int>()<<" min "<<linear_indices.min().item<int>()<<std::endl;

            torch::Tensor gt_samples = flat_gt.index({linear_indices}); // [10000, 3]
    
            torch::Tensor black = torch::zeros({gt_image.size(1), gt_image.size(2), 3}, 
                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            // Flatten the black image to [H*W, 3]
            torch::Tensor black_flat = black.reshape({-1, 3});
            black_flat.index_put_({linear_indices}, gt_samples.cpu());
            // std::cout<<"[debug] black_flat sizes "<<black_flat.sizes()<<std::endl;
            cv::Mat image_cv(gt_image.size(1), gt_image.size(2), CV_32FC3, black_flat.data_ptr<float>());
            cv::cvtColor(image_cv, image_cv, cv::COLOR_RGB2BGR);
            image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_train_gt_samples"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_train_gt_samples" / (std::to_string(pkf->fid_)+".jpg"), image_cv);
        }
    }

    auto iter_start_timing3 = std::chrono::steady_clock::now();
    
    std::cout<<"[GaussianMapper::optimizeGlobalAlign] new step3: orb match"<<std::endl;

    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> rt_confidences;
    cv::Size size_hr(
        int(kf_params_.hr_width_ * (float(kf_params_.lr_height_)/float(kf_params_.hr_height_))),
        kf_params_.lr_height_
    );
    float hr_ratio = float(size_hr.height) / float(kf_params_.hr_height_);
    for(int id=0; id<keyframes_ids.size(); id++){
        auto& pkf = scene_->keyframes().at(keyframes_ids.at(id));
        if(!pkf->has_hr_fid_) continue;

        int good_matches = pkf->matchInitHROrbGMS();
        if(good_matches < 100) continue;

        auto orb_hr_key = std::make_tuple(
            size_hr.width,
            size_hr.height,
            hr_ratio
        );
        auto orb_hr_it = pkf->orb_multisizes_hr_.find(orb_hr_key);
        if(orb_hr_it == pkf->orb_multisizes_hr_.end())
            throw std::runtime_error("[error][GaussianMapper::optimizeGlobalAlign] cannot find orb hr size!");
        const auto& kp_hr = std::get<1>(orb_hr_it->second);

        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        cv::Mat K_hr = (cv::Mat_<double>(3,3) << 
            kf_params_.hr_fx_ * hr_ratio, 0, kf_params_.hr_cx_ * hr_ratio,
            0, kf_params_.hr_fy_ * hr_ratio, kf_params_.hr_cy_ * hr_ratio,
            0, 0, 1
        );

        float confidence = this->getRelatedPoseGMS(
            pkf, 
            pkf->orb_keypoints_lr_, kp_hr,
            pkf->orb_matches_lr2hr_,
            pkf->img_auxiliary_undist_,
            K_hr,
            rvec, tvec, inliers
        );

        if(confidence < 0.f){
            std::cout<<"[warning][GaussianMapper::optimizeGlobalAlign] fid "<<pkf->fid_<<" getRelatedPoseGMS failed "<<std::endl;
            continue;
        }

        rvecs.push_back(rvec.clone());
        tvecs.push_back(tvec.clone());
        rt_confidences.push_back(confidence);
    }

    if(rt_confidences.size() < 3) 
        std::cout<<"[warning][GaussianMapper::optimizeGlobalAlign] not enough valid orb pose estimations for global align: "<<rt_confidences.size()<<std::endl;
    
    // average rvec and tvec weighted by confidence
    float sum_confidence = std::accumulate(rt_confidences.begin(), rt_confidences.end(), 0.f);
    cv::Mat rvec_avg = cv::Mat::zeros(3,1,CV_32FC1);
    cv::Mat tvec_avg = cv::Mat::zeros(3,1,CV_32FC1);
    for(int i=0; i<rt_confidences.size(); i++){
        rvec_avg += rvecs[i] * (rt_confidences[i] / sum_confidence);
        tvec_avg += tvecs[i] * (rt_confidences[i] / sum_confidence);
        // std::cout<<"[debug] id "<<i<<" confidence "<<rt_confidences[i]<<" "<<(rt_confidences[i] / sum_confidence)<<" rvec \n"<<rvecs[i]<<" tvec \n"<<tvecs[i]<<std::endl;
    }

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);

        cv::Mat T_avg = general_utils::vecs2transformation(rvec_avg, tvec_avg);
        this->global_align_pose_ = torch::from_blob(T_avg.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
        std::cout<<"[debug] averaged T_lr2hr (this->global_align_pose_) \n"<<this->global_align_pose_<<std::endl;

        // this->getAvgGlobalPose(this->global_algin_fids_, this->global_align_pose_);
        for(int i=0; i<keyframes_ids.size(); i++){
            auto& pkf = scene_->keyframes().at(keyframes_ids.at(i));
            pkf->setGlobalDeltaPose(this->global_align_pose_);
        }
    }

    if(kf_params_.debug_){
    {
        torch::NoGradGuard no_grad;
        torch::Tensor rendered_image, gt_image;
        for(int id = 0; id<keyframes_ids.size(); id++){
            auto& pkf = scene_->keyframes().at(keyframes_ids.at(id));
            if(!pkf->has_hr_fid_) continue;
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                auto render_pkg = GaussianRenderer::render(pkf,
                    this->kf_params_.hr_height_, this->kf_params_.hr_width_,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    true, true, true, true
                );

                rendered_image = std::get<0>(render_pkg);
                gt_image = pkf->getGTHRImg(1.f, true);
            }
            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
            cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
            image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_aligned"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_aligned" / (std::to_string(pkf->fid_)+".jpg"), image_cv);
            
            // gt_image = gt_image.cpu();
            auto image_gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
            cv::cvtColor(image_gt_cv, image_gt_cv, CV_RGB2BGR);
            image_gt_cv.convertTo(image_gt_cv, CV_8UC3, 255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_aligned" / ("gt_" + std::to_string(pkf->fid_)+".jpg"), image_gt_cv);
        }
    }
    }

    // throw std::runtime_error("[debug] step3!");

    std::cout<<"[GaussianMapper::optimizeGlobalAlign] new step4: init hr local pose"<<std::endl;

    for(int id=0; id<keyframes_ids.size(); id++){
        auto& pkf = scene_->keyframes().at(keyframes_ids.at(id));
        if(!pkf->has_hr_fid_) continue;

        this->optimizeGlobalHRPose(pkf, false);
        
    }

    // throw std::runtime_error("[debug] step5!");

    std::cout<<"[GaussianMapper::optimizeGlobalAlign] step5: hr color refinement"<<std::endl;
    std::vector<std::size_t> random_hr_color_fids;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        gaussians_->resetOptimizer(opt_params_, 1.8f);
        
        this->getBatchShuffledFrameIds(keyframes_ids, random_hr_color_fids, this->global_align_hr_color_iter_);
    
        for(int i=0; i<keyframes_ids.size(); i++){
            auto& pkf = scene_->keyframes().at(keyframes_ids.at(i));
            // pkf->resetOptimizer();
            // pkf->resetOptimizer(true, kf_params_.hr_color_theta_lr_, kf_params_.hr_color_rho_lr_); // change
            pkf->updateOptimizer(0.7f);
            pkf->resetFullExposure();
        }
    }

    // std::cout<<"[debug] gs sh degree "<<this->gaussians_->active_sh_degree_<<std::endl;
    hr_resize_ratio = this->getRsizedHRScale(1.f, hr_width_resize, hr_height_resize);
    std::unordered_map<std::size_t, torch::Tensor> opacity_masks;
    for(int i=0; i<this->global_align_hr_color_iter_; i++){
        auto id = random_hr_color_fids.at(i); // !!! need change back to (i)
        auto& pkf = scene_->keyframes().at(id);

        torch::Tensor rendered_image, rendered_depth, rendered_opacity, gt_image, opacity_mask;
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            auto render_pkg = GaussianRenderer::render(pkf,
                hr_height_resize, hr_width_resize,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, true, false, false
            );
            
            rendered_image = std::get<0>(render_pkg);
            rendered_depth = std::get<4>(render_pkg);
            rendered_opacity = std::get<5>(render_pkg);
            
            if(!opacity_masks[id].defined())
                opacity_masks[id] = (rendered_opacity > 0.1f).to(torch::kFloat32);
            
            opacity_mask = opacity_masks[id];
            
            gt_image = pkf->getGTHRImg(hr_resize_ratio, true);
        }

        auto loss_rgb = loss_utils::get_loss_rgb(rendered_image, gt_image,
            this->lambdaDssim(), pkf->exposure_a_, pkf->exposure_b_, opacity_mask);

        auto loss = loss_rgb;

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);

            if(i<100) pkf->stepOptimizer(false, false);
            else pkf->stepOptimizer(false, true);
            pkf->zeroOptimizerGrad(true, true);

            // pkf->updateLocalDeltaPose(true);

            // if(i==50)
            //     pkf->updateOptimizer(0.5f);
            // if(i==100)
            //     pkf->updateOptimizer(0.1f);

            if(kf_params_.debug_){
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_color_pose"))
                rendered_image = rendered_image * torch::exp(pkf->exposure_a_) + pkf->exposure_b_;
                // auto cv_opacity_mask = tensor_utils::torchTensor2CvMat_Float32(opacity_mask);
                // cv_opacity_mask.convertTo(cv_opacity_mask, CV_8UC1, 255.0f);
                // cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_color_pose" / ("mask_" + std::to_string(pkf->fid_) + "_" + std::to_string(i) + ".jpg"), cv_opacity_mask);
                auto masked_rendered_image = rendered_image * opacity_mask;
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_color_pose" / ("hr_" + std::to_string(pkf->fid_) + "_" + std::to_string(i) + ".jpg"), image_cv);
                auto masked_gt = gt_image * opacity_mask;
                auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt);
                cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
                gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + kf_params_.debug_dir_) / "hr_color_pose" / ("gt_" + std::to_string(pkf->fid_) + ".jpg"), gt_image_cv);

                auto masked_gt_image = gt_image * opacity_mask;
                // auto masked_rendered_image = rendered_image * opacity_mask;
                std::cout<<"[debug] fid "<<pkf->fid_<<" iter "<<i<<" metrics: ";
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
        }
    }

    auto iter_start_timing4 = std::chrono::steady_clock::now();

    auto iter_time1 = std::chrono::duration_cast<std::chrono::milliseconds>(
                    iter_start_timing2 - iter_start_timing1).count();
    auto iter_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                    iter_start_timing3 - iter_start_timing2).count();
    auto iter_time3 = std::chrono::duration_cast<std::chrono::milliseconds>(
                    iter_start_timing4 - iter_start_timing3).count();
    // std::cout<<"[debug] gobal align step 1: "<<iter_time1<<"ms"<<std::endl;
    // std::cout<<"[debug] gobal align step 2: "<<iter_time2<<"ms"<<std::endl;
    // std::cout<<"[debug] gobal align step 3: "<<iter_time3<<"ms"<<std::endl;
    auto total_iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    iter_start_timing4 - iter_start_timing1).count();
    std::cout<<"[debug] gobal align total time: "<<total_iter_time<<"ms"<<std::endl;

    // throw std::runtime_error("[debug] finish global align!");

}

void GaussianMapper::getBatchShuffledFrameIds(
    const std::vector<std::size_t>& in_fids,
    std::vector<std::size_t>& out_fids,
    int required_iters)
{
    out_fids.clear();

    std::vector<std::size_t> temp_fids;
    for(int i=0; i<in_fids.size(); i++){
        auto& pkf = scene_->keyframes().at(in_fids.at(i));
        if(!pkf->has_hr_fid_) continue;
        temp_fids.push_back(in_fids.at(i));
    }

    int valid_count = 0;
    for(int i=0; i<required_iters; i++){
        std::size_t fid = temp_fids.at(i % temp_fids.size());
        auto& pkf = scene_->keyframes().at(fid);
        if(!pkf->has_hr_fid_) continue;
        out_fids.push_back(fid);
        valid_count++;
    }

    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    unsigned int seed = static_cast<unsigned int>(millis ^ (reinterpret_cast<uintptr_t>(&out_fids) >> 3));
    std::shuffle(out_fids.begin(), out_fids.end(), std::default_random_engine(seed + rand() % 1000));

    if (rand() % 2 == 0) std::reverse(out_fids.begin(), out_fids.end());
    if (!out_fids.empty() && rand() % 3 == 0) std::rotate(out_fids.begin(), out_fids.begin() + (rand() % out_fids.size()), out_fids.end());
    
}


float GaussianMapper::getRelatedPoseGMS(
    std::shared_ptr<GaussianKeyframe> pkf, 
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const std::vector<int>& match12,
    const cv::Mat& depths1,
    const cv::Mat& K2,
    cv::Mat& rvec, cv::Mat& tvec, 
    std::vector<int>& inliers,
    bool is_lr,
    float hr_resize_ratio,
    bool use_lr2hr_depth
){ 
    float cx, cy, fx, fy;
    if(is_lr){
        cx = kf_params_.lr_cx_;
        cy = kf_params_.lr_cy_;
        fx = kf_params_.lr_fx_;
        fy = kf_params_.lr_fy_;
    }
    else{
        cx = kf_params_.hr_cx_ * hr_resize_ratio;
        cy = kf_params_.hr_cy_ * hr_resize_ratio;
        fx = kf_params_.hr_fx_ * hr_resize_ratio;
        fy = kf_params_.hr_fy_ * hr_resize_ratio;
    }
    // frame1 with 3D points, frame2 with 2D points
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    float depth1;
    for (size_t i = 0; i < kpts1.size(); i++){
        if(match12[i] < 0 || match12[i] >= kpts2.size()) continue;

        const auto& kpt1 = kpts1[i];
        const auto& kpt2 = kpts2[match12[i]];
        if(!use_lr2hr_depth)
            depth1 = depths1.at<float>(int(kpt1.pt.y), int(kpt1.pt.x));
        else{
            torch::Tensor gt_depth_lr = pkf->getGTLRDpt(false).squeeze() * pkf->getGTLRDptMsk(false).squeeze();
            float depth_hr = depths1.at<float>(int(kpt1.pt.y), int(kpt1.pt.x));
            auto Pc_hr = torch::tensor({
                (kpt1.pt.x - cx) * depth_hr / fx,
                (kpt1.pt.y - cy) * depth_hr / fy,
                depth_hr,
                1.0f
            }, torch::kFloat32).reshape({4, 1});  // (4x1)
            auto Pc_lr = pkf->getFullDeltaPose().cpu().inverse().mm(Pc_hr);
            auto u_lr = Pc_lr[0].item<float>() * kf_params_.lr_fx_ / Pc_lr[2].item<float>() + kf_params_.lr_cx_;
            auto v_lr = Pc_lr[1].item<float>() * kf_params_.lr_fy_ / Pc_lr[2].item<float>() + kf_params_.lr_cy_;

            if(u_lr < 0 || u_lr >= gt_depth_lr.size(1) - 1 || v_lr < 0 || v_lr >= gt_depth_lr.size(0) - 1)
                depth1 = 0.f;
            else
                depth1 = gt_depth_lr[int(v_lr)][int(u_lr)].item<float>();
        }

        if(depth1 < 1e-5f) continue;

        cv::Point3f pt3d(
            (kpt1.pt.x - cx) * depth1 / fx,
            (kpt1.pt.y - cy) * depth1 / fy,
            depth1
        );
        cv::Point2f pt2d(
            kpt2.pt.x,
            kpt2.pt.y
        );

        pts3d.push_back(pt3d);
        pts2d.push_back(pt2d);
    }

    if(pts3d.size() < 6){
        std::cout<<"[GaussianMapper::getRelatedPoseGMS] not enough 2D-3D points for PnP: "<<pts3d.size()<<std::endl;
        return -1.f;
    }

    bool success = cv::solvePnPRansac(
        pts3d, pts2d, K2, cv::Mat(),
        rvec, tvec,
        false,
        100,      // number of iterations
        2.0f,     // reprojection error
        0.99,     // confidence
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );

    if(!success || inliers.size() < 6){
        std::cout<<"[GaussianMapper::getRelatedPoseGMS] cv::solvePnPRansac failed!"<<std::endl;
        return -2.f;
    }

    return float(inliers.size()) / float(pts3d.size());
}

void GaussianMapper::insertNewKeyframesFromSLAM(){
    std::map<ORB_SLAM3::MappingOperation::OprType, std::vector<std::size_t>> local_mapping_operation;
    ORB_SLAM3::MappingOperation::OprType opr_type;
    std::vector<std::size_t> new_kf_ids;

    int inserted_count = 0;
    int batch_inserted_count = 0;
    std::vector<std::shared_ptr<KeyframeFrontend>> batch_kfs;
    std::vector<double> batch_timestamps;
    std::vector<std::size_t> new_kf_batch_ids;
    const auto& frontend_keyframes = pSLAM_->getAtlas()->GetCurrentMap()->GetAllKeyFrames();
    while (this->pSLAM_->getAtlas()->hasMappingOperation()){
        ORB_SLAM3::MappingOperation opr = this->pSLAM_->getAtlas()->getAndPopMappingOperation();
        opr_type = opr.meOperationType;
        auto& opr_associated_kf = opr.associatedKeyFrames();

        std::cout<<"[GaussianMapper::insertNewKeyframesFromSLAM] get mapping operation type "<<int(opr_type)<<" with "<<opr_associated_kf.size()<<" associated keyframes."<<std::endl;
        
        for (KeyframeFrontend& kf : opr_associated_kf){
            auto kfid = std::get<0>(kf);
            std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);

            if(this->local_align_batch_fids_timestamps_map_.find(kfid) == this->local_align_batch_fids_timestamps_map_.end()){
                double timestamp = -1;
                // find timestamp from all keyframes
                auto kf_it = std::find_if(frontend_keyframes.begin(), frontend_keyframes.end(),
                    [kfid](ORB_SLAM3::KeyFrame* kf) { return kf->mnId == kfid; });
                if (kf_it != frontend_keyframes.end())
                    timestamp = (*kf_it)->mTimeStamp;
                
                this->local_align_batch_fids_timestamps_map_[kfid] = timestamp;
            }

            if(!pkf){
                // double timestamp = -1;
                // // find timestamp from all keyframes
                // auto kf_it = std::find_if(opr_keyframes.begin(), opr_keyframes.end(),
                //     [kfid](ORB_SLAM3::KeyFrame* kf) { return kf->mnId == kfid; });
                // if (kf_it != opr_keyframes.end())
                //     timestamp = (*kf_it)->mTimeStamp;
                double timestamp = this->local_align_batch_fids_timestamps_map_[kfid];
                
                if(this->local_align_batch_size_ > 1){
                    if(batch_inserted_count == this->local_align_batch_size_){
                        this->insertBatchKeyframes(batch_kfs, batch_timestamps);
                        batch_kfs.clear();
                        batch_timestamps.clear();
                        new_kf_batch_ids.clear();
                        batch_inserted_count = 0;

                        // throw std::runtime_error("[GaussianMapper::insertNewKeyframesFromSLAM] debug batch_inserted_count!");
                    }
                    
                    if(std::find(new_kf_batch_ids.begin(), new_kf_batch_ids.end(), kfid) == new_kf_batch_ids.end()){
                        if(std::find(new_kf_ids.begin(), new_kf_ids.end(), kfid) != new_kf_ids.end()){
                            std::cout<<"[GaussianMapper::insertNewKeyframesFromSLAM] keyframe id "<<kfid<<" already in current local mapping operation, skip batch insertion."<<std::endl;
                            continue;
                        }
                        std::cout<<"[debug] insert new keyframe id "<<kfid<<" to batch."<<std::endl;
                        // batch_kfs.push_back(&kf);
                        auto shared_kf = std::make_shared<KeyframeFrontend>(kf);
                        batch_kfs.push_back(shared_kf);
                        batch_timestamps.push_back(timestamp);
                        new_kf_batch_ids.push_back(kfid);
                        batch_inserted_count++;
                        std::cout<<"[debug] current batch size "<<batch_kfs.size()<<std::endl;
                    }
                }
                else this->insertOneKeyframe(kf, timestamp);
                
                if (std::find(new_kf_ids.begin(), new_kf_ids.end(), kfid) == new_kf_ids.end()){
                    new_kf_ids.push_back(kfid);
                    inserted_count++;
                }

            }
            // else
            //     std::cout<<"[GaussianMapper::insertNewKeyframesFromSLAM] keyframe id "<<kfid<<" already exists, skip insertion."<<std::endl;
        
            if(inserted_count == 60)
                throw std::runtime_error("[GaussianMapper::insertNewKeyframesFromSLAM] debug throw!");
        }
        
    }

    local_mapping_operation[opr_type] = new_kf_ids;
    this->local_mapping_operations_.push_back(local_mapping_operation);
    // std::cout<<"[GaussianMapper::insertNewKeyframesFromSLAM] inserted "<<new_kf_ids.size()<<" new keyframes."<<std::endl;
    throw std::runtime_error("[GaussianMapper::insertNewKeyframesFromSLAM] debug throw!");

}

std::size_t GaussianMapper::handleKeyframeFrontend(std::size_t kfid){
    // used for cuvslam
    std::shared_ptr<GaussianKeyframe> new_kf = 
        std::make_shared<GaussianKeyframe>(kfid, 0, &this->kf_params_);

    std::vector<double> tum_pose; // tx, ty, tz, qx, qy, qz, qw
    this->pCuVSLAM_->GetPose(kfid, tum_pose, new_kf->lr_timestamp_);
    new_kf->setPose(tum_pose[6], tum_pose[3], tum_pose[4], tum_pose[5], // qw,qx,qy,qz
                    tum_pose[0], tum_pose[1], tum_pose[2], true); // tx, ty, tz
    new_kf->zfar_ = this->z_far_;
    new_kf->znear_ = this->z_near_;

    Camera& camera = this->scene_->cameras_.at(0);
    new_kf->setCameraParams(camera);

    cv::Mat img_rgb, img_depth;
    std::string img_depth_filename;
    this->pCuVSLAM_->GetFrame(kfid, img_rgb, img_depth, new_kf->img_filename_, img_depth_filename); // already undistorted from L515
    new_kf->setGTLRImg(img_rgb, true);
    new_kf->setGTLRDpt(img_depth, true);

    this->increaseKeyframeTimesOfUse(new_kf, this->newKeyframeTimesOfUse());

    if(this->kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((this->result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "lr_undist"))
        cv::Mat imgAux_vis = new_kf->img_auxiliary_undist_.clone();
        cv::Mat imgRGB_vis = new_kf->img_undist_.clone();
        imgRGB_vis.convertTo(imgRGB_vis, CV_8UC3, 255.0f, 0.f);
        cv::cvtColor(imgRGB_vis, imgRGB_vis, cv::COLOR_BGR2RGB);
        imgAux_vis.convertTo(imgAux_vis, CV_8UC1, this->kf_params_.lr_depth_factor_/6000.0f*255.0f, 0.f);
        auto path_rgb = (this->result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB.jpg"));
        auto path_aux = (this->result_dir_ / ("run_cuvslam" + this->kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux.jpg"));
        cv::imwrite(path_rgb.string(), imgRGB_vis);
        cv::imwrite(path_aux.string(), imgAux_vis);
    }

    this->scene_->addKeyframe(new_kf, &this->kfid_shuffled_);
    return new_kf->fid_;
}

std::size_t GaussianMapper::handleKeyframeFrontend(KeyframeFrontend& kf, std::shared_ptr<GaussianKeyframe> new_kf, float timestamp){
    auto camera_id = std::get<1>(kf);
    auto& pose = std::get<2>(kf);
    auto& img = std::get<3>(kf);
    auto& dpt = std::get<5>(kf);
    new_kf->img_filename_ = std::get<8>(kf);
    new_kf->lr_timestamp_ = timestamp;

    // undistort and set camera params
    Camera& camera = scene_->cameras_.at(camera_id);
    new_kf->setCameraParams(camera);

    cv::Mat img_undistorted, dpt_undistorted;
    camera.undistortImage(img, img_undistorted);
    camera.undistortImage(dpt, dpt_undistorted);
    // img_undistorted = img;
    // dpt_undistorted = dpt; // !!! depth no need distort?

    new_kf->setGTLRImg(img_undistorted);
    new_kf->setGTLRDpt(dpt_undistorted); 
    new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(img_undistorted, device_type_);
        
    this->generatePyramidSizes(new_kf, camera);
    this->increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

    // set pose
    new_kf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    // new_kf->computeTransformTensors();

    this->scene_->addKeyframe(new_kf, &kfid_shuffled_);

    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist"))
        cv::Mat imgRGB_undistorted_vis, imgRGB_vis;
        img_undistorted.convertTo(imgRGB_undistorted_vis, CV_8UC3, 255.0f, 0.f);
        cv::cvtColor(imgRGB_undistorted_vis, imgRGB_undistorted_vis, CV_RGB2BGR);
        img.convertTo(imgRGB_vis, CV_8UC3, 255.0f, 0.f);
        cv::cvtColor(imgRGB_vis, imgRGB_vis, CV_RGB2BGR);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB_undistorted.jpg"), imgRGB_undistorted_vis);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB.jpg"), imgRGB_vis);

        cv::Mat imgAux_undistorted_vis, imgAux_vis;
        dpt_undistorted.convertTo(imgAux_undistorted_vis, CV_8UC1, 4000.0f/6000.0f*255.0f, 0.f);
        dpt.convertTo(imgAux_vis, CV_8UC1, 4000.0f/6000.0f*255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux_undistorted.jpg"), imgAux_undistorted_vis);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux.jpg"), imgAux_vis);
    }

    return new_kf->fid_;
}

float GaussianMapper::optimizeLocalLRPose(
    std::shared_ptr<GaussianKeyframe> pkf, 
    bool use_differential_pose
){
    torch::Tensor loss;
    if(use_differential_pose)
    {
        torch::Tensor opacity_mask;
        pkf->resetOptimizer();
        for(int i=0; i<this->local_align_lr_pose_iter_; ++i){
            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, true, true, true
            );

            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);

            if(i==0 || i==int(this->local_align_lr_pose_iter_*0.5f))
                opacity_mask = (rendered_opacity > 0.5f).to(torch::kFloat32).squeeze();

            auto gt_image = pkf->getGTLRImg().cuda();
            auto gt_depth = pkf->getGTLRDpt().cuda();
            auto gt_depth_mask = pkf->getGTLRDptMsk().cuda().to(torch::kFloat32); 

            auto valid_mask = opacity_mask * gt_depth_mask;

            loss = loss_utils::get_loss_rgbd(
                rendered_image, gt_image,
                rendered_depth, gt_depth,
                this->lambdaDssim(),
                this->global_align_lr_depth_lambda_,
                pkf->exposure_a_, pkf->exposure_b_,
                valid_mask,
                device_type_
            );

            loss.backward();

            {
                torch::NoGradGuard no_grad;

                gaussians_->optimizer_->zero_grad(true); 
                pkf->stepOptimizer(true);
                pkf->zeroOptimizerGrad(true, true);

                pkf->updateBasePose();

                if(i == int(this->local_align_lr_pose_iter_ * 0.5f))
                    pkf->updateOptimizer(0.5f);
                
                if(kf_params_.debug_){
                    auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
                    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose"))
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                
                    if(i==0){
                        std::cout<<"[GaussianMapper::handleKeyframeFrontend] use differential pose fid "<<pkf->fid_<<std::endl;
                        auto masked_gt_image = gt_image * valid_mask.unsqueeze(0);
                        auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
                        cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                        gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                    }
                }
            }
        }
    }
    else
    {
        float confidence = -1.f;
        int render_attempts = 0;

        torch::Tensor rendered_image, gt_depth, gt_image, gt_depth_mask, valid_mask, base_pose_copy;
        while(confidence <= 0.f && render_attempts < this->max_pnp_render_attempts_){
            {
                torch::NoGradGuard no_grad;

                auto render_pkg = GaussianRenderer::render(pkf,
                    this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    false, true, true, true
                );

                rendered_image = std::get<0>(render_pkg);
                auto rendered_depth = std::get<4>(render_pkg);
                auto rendered_opacity = std::get<5>(render_pkg);

                auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();

                gt_image = pkf->getGTLRImg(true);
                gt_depth = pkf->getGTLRDpt(true);
                gt_depth_mask = pkf->getGTLRDptMsk(true).to(torch::kFloat32); 

                valid_mask = opacity_mask * gt_depth_mask;

                loss = loss_utils::get_loss_rgbd(
                    rendered_image, gt_image,
                    rendered_depth, gt_depth,
                    this->lambdaDssim(),
                    this->global_align_lr_depth_lambda_,
                    pkf->exposure_a_, pkf->exposure_b_,
                    valid_mask,
                    device_type_
                );

                rendered_image = valid_mask.unsqueeze(0) * rendered_image;

            }    

            if(kf_params_.debug_){
                std::cout<<"[GaussianMapper::handleKeyframeFrontend] use classic pose fid "<<pkf->fid_<<std::endl;
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose"))
                auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_before.jpg"), image_cv);
                auto masked_gt_image = gt_image * valid_mask.unsqueeze(0);
                auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
                cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
            // auto time1 = std::chrono::steady_clock::now();
            std::vector<cv::KeyPoint> kpts_rendered, kpts_gt;
            std::vector<int> matches_rendered2gt, matches_gt2rendered;
            int good_matches = pkf->matchLROrbGMS(rendered_image, 
                kpts_rendered, kpts_gt,
                matches_rendered2gt, matches_gt2rendered
            );

            std::cout<<"[GaussianMapper::optimizeLocalLRPose] classic pose fid "<<pkf->fid_<<" found "<<good_matches<<" good matches for GMS pose estimation."<<std::endl;

            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            cv::Mat K_lr = (cv::Mat_<float>(3,3) << 
                kf_params_.lr_fx_, 0.f, kf_params_.lr_cx_,
                0.f, kf_params_.lr_fy_, kf_params_.lr_cy_,
                0.f, 0.f, 1.f
            );

            auto depth_cpu = gt_depth.squeeze().to(torch::kCPU).contiguous();
            cv::Mat depth_cv(depth_cpu.size(0), depth_cpu.size(1), CV_32F);
            std::memcpy(depth_cv.data, depth_cpu.data_ptr<float>(), sizeof(float)*depth_cpu.size(0)*depth_cpu.size(1));

            confidence = this->getRelatedPoseGMS(
                pkf,
                kpts_gt, kpts_rendered,
                matches_gt2rendered,
                depth_cv,
                K_lr,
                rvec, tvec,
                inliers
            );
            // auto time2 = std::chrono::steady_clock::now();
            // auto pnp_time = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
            // std::cout<<"[GaussianMapper::optimizeLocalLRPose] PnP pose estimation time for fid "<<pkf->fid_<<" took "<<pnp_time<<"ms"<<std::endl;

            if(confidence <= 0.f){
                render_attempts++;
                std::cout<<"[GaussianMapper::optimizeLocalLRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed, re-rendering "<<render_attempts<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
                continue;
            }

            auto delta_pose = general_utils::vecs2transformation(rvec, tvec);
            auto delta_pose_torch = torch::from_blob(delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
            delta_pose_torch = delta_pose_torch.inverse();

            
            auto updated_base_pose = delta_pose_torch.mm(pkf->getBasePose());
            base_pose_copy = pkf->getBasePose().clone();
            pkf->setBasePose(updated_base_pose);

            torch::Tensor updated_loss;
            {
                torch::NoGradGuard no_grad;

                auto render_pkg = GaussianRenderer::render(pkf,
                    this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    false, true, true, true
                );

                rendered_image = std::get<0>(render_pkg);
                auto rendered_depth = std::get<4>(render_pkg);
                auto rendered_opacity = std::get<5>(render_pkg);

                auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();

                valid_mask = opacity_mask * gt_depth_mask;

                updated_loss = loss_utils::get_loss_rgbd(
                    rendered_image, gt_image,
                    rendered_depth, gt_depth,
                    this->lambdaDssim(),
                    this->global_align_lr_depth_lambda_,
                    pkf->exposure_a_, pkf->exposure_b_,
                    valid_mask,
                    device_type_
                );

            }   

            if (updated_loss.item<float>() > loss.item<float>()){
                std::cout<<"[GaussianMapper::optimizeLocalLRPose] rendered RGBD LR pose update increases loss from "<<loss.item<float>()<<" to "<<updated_loss.item<float>()<<", revert pose update."<<std::endl;
                pkf->setBasePose(base_pose_copy);

                if(kf_params_.debug_){
                    std::cout<<"[GaussianMapper::optimizeLocalLRPose] classic pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                        <<" inliers "<<inliers.size()<<"/"<<good_matches
                        <<" loss before "<<loss.item<float>()
                        <<" loss after "<<updated_loss.item<float>()<<std::endl;
                }
                break;
            }

            if(kf_params_.debug_){
                auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_after.jpg"), image_cv);
                auto masked_gt_image = gt_image * valid_mask.unsqueeze(0);
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }

            
        }
        if(confidence <= 0.f){
            std::cout<<"[error][GaussianMapper::optimizeLocalLRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed after "<<this->max_pnp_render_attempts_<<" attempts!"<<std::endl;
            throw std::runtime_error("[GaussianMapper::optimizeLocalLRPose] PnP pose estimation failed!");
        }
        
        /*
        std::cout<<"[debug] test refined pose with VGICP for fid "<<pkf->fid_<<std::endl;
        std::cout<<pkf->getBasePose()<<std::endl;
        auto test_vgicp_refined_pose = this->refinePoseFastVGICP(pkf);
        std::cout<<test_vgicp_refined_pose<<std::endl;
        std::cout<<pkf->getBasePose()<<std::endl;

        torch::Tensor icp_loss;
        {
            torch::NoGradGuard no_grad;

            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, true, true, true
            );

            rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);

            auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();

            valid_mask = opacity_mask * gt_depth_mask;

            icp_loss = loss_utils::get_loss_rgbd(
                rendered_image, gt_image,
                rendered_depth, gt_depth,
                this->lambdaDssim(),
                this->global_align_lr_depth_lambda_,
                pkf->exposure_a_, pkf->exposure_b_,
                valid_mask,
                device_type_
            );

            if(kf_params_.debug_){
                auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(pkf->fid_)+"_after_icp.jpg"), image_cv);
                auto masked_gt_image = gt_image * valid_mask.unsqueeze(0);
                std::cout<<"[debug] icp loss for fid "<<pkf->fid_<<" : "<<icp_loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }

        }
        */
    }
    return loss.item<float>();
}

float GaussianMapper::optimizeLocalLRPoses(std::vector<std::size_t>& fids) {
    std::vector<std::shared_ptr<GaussianKeyframe>> pkfs;
    std::vector<torch::Tensor> rendered_images, gt_images, gt_depths, valid_masks;
    std::vector<cv::Mat> gt_depths_cv, rendered_depths_cv;
    std::vector<float> losses;

    // get last frame of previous batch
    auto prev_batch = std::prev(this->local_mapping_batch_ids_.end(), 2);
    auto prev_fid = prev_batch->second.back();
    {
        torch::NoGradGuard no_grad;

        auto pkf_prev = scene_->getKeyframe(prev_fid);
        auto render_pkg = GaussianRenderer::render(pkf_prev,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, true, true, true
        );

        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<4>(render_pkg);
        auto rendered_opacity = std::get<5>(render_pkg);

        auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();

        auto gt_image = pkf_prev->getGTLRImg(true);
        auto gt_depth = pkf_prev->getGTLRDpt(false);
        auto gt_depth_mask = pkf_prev->getGTLRDptMsk(true); 

        auto valid_mask = opacity_mask * gt_depth_mask;
        rendered_image = valid_mask.unsqueeze(0) * rendered_image;

        rendered_images.push_back(rendered_image);
        gt_images.push_back(gt_image);
        gt_depths.push_back(gt_depth);
        // valid_masks.push_back(valid_mask);

        auto gt_depth_cpu = gt_depth.squeeze().contiguous();
        cv::Mat gt_depth_cv(gt_depth_cpu.size(0), gt_depth_cpu.size(1), CV_32F);
        std::memcpy(gt_depth_cv.data, gt_depth_cpu.data_ptr<float>(), sizeof(float)*gt_depth_cpu.size(0)*gt_depth_cpu.size(1));
        gt_depths_cv.push_back(gt_depth_cv);

        // auto rendered_depth_cpu = rendered_depth.squeeze().to(torch::kCPU).contiguous();
        // cv::Mat rendered_depth_cv(rendered_depth_cpu.size(0), rendered_depth_cpu.size(1), CV_32F);
        // std::memcpy(rendered_depth_cv.data, rendered_depth_cpu.data_ptr<float>(), sizeof(float)*rendered_depth_cpu.size(0)*rendered_depth_cpu.size(1));
        // rendered_depths_cv.push_back(rendered_depth_cv);

        pkfs.push_back(std::move(pkf_prev));
    }

    for (int i=0; i<fids.size(); i++){
        torch::NoGradGuard no_grad;

        auto pkf = scene_->getKeyframe(fids[i]);

        auto render_pkg = GaussianRenderer::render(pkf,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, true, true, true
        );

        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<4>(render_pkg);
        auto rendered_opacity = std::get<5>(render_pkg);

        auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();

        auto gt_image = pkf->getGTLRImg(true);
        auto gt_depth = pkf->getGTLRDpt(false);
        auto gt_depth_mask = pkf->getGTLRDptMsk(true); 

        auto valid_mask = opacity_mask * gt_depth_mask;

        auto loss = loss_utils::get_loss_rgbd(
            rendered_image, gt_image,
            rendered_depth, gt_depth,
            this->lambdaDssim(),
            this->global_align_lr_depth_lambda_,
            pkf->exposure_a_, pkf->exposure_b_,
            valid_mask,
            device_type_
        );
        losses.push_back(loss.item<float>()); // !!! size of losses is smaller than pkfs by 1, since we don't compute loss for the first frame of previous batch

        rendered_image = valid_mask.unsqueeze(0) * rendered_image; // put it after loss computation

        rendered_images.push_back(rendered_image);
        gt_images.push_back(gt_image);
        gt_depths.push_back(gt_depth);
        // valid_masks.push_back(valid_mask);

        auto gt_depth_cpu = gt_depth.squeeze().contiguous();
        cv::Mat gt_depth_cv(gt_depth_cpu.size(0), gt_depth_cpu.size(1), CV_32F);
        std::memcpy(gt_depth_cv.data, gt_depth_cpu.data_ptr<float>(), sizeof(float)*gt_depth_cpu.size(0)*gt_depth_cpu.size(1));
        gt_depths_cv.push_back(gt_depth_cv);

        // auto rendered_depth_cpu = rendered_depth.squeeze().to(torch::kCPU).contiguous();
        // cv::Mat rendered_depth_cv(rendered_depth_cpu.size(0), rendered_depth_cpu.size(1), CV_32F);
        // std::memcpy(rendered_depth_cv.data, rendered_depth_cpu.data_ptr<float>(), sizeof(float)*rendered_depth_cpu.size(0)*rendered_depth_cpu.size(1));
        // rendered_depths_cv.push_back(rendered_depth_cv);

        pkfs.push_back(std::move(pkf));
    }

    int render_attempts = 0;
    std::vector<bool> valid_updates(fids.size(), false);
    while(render_attempts < this->max_pnp_render_attempts_){
        std::vector<torch::Tensor> final_delta_poses, prev_base_poses;
        
        for (int i=1; i<pkfs.size(); i++){
            if (valid_updates[i-1]) continue;

            auto& cur_pkf = pkfs[i];
            auto& prev_pkf = pkfs[i-1];
            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] optimizing local LR pose for fid "<<cur_pkf->fid_<<" with previous frame fid "<<prev_pkf->fid_<<std::endl;

            // pnp linkage 3d -> 2d: prev gt -> cur rendered, cur gt -> cur rendered
            std::vector<cv::KeyPoint> kpts_cur_gt, kpts_prev_gt, kpts_cur_rendered;
            std::vector<int> matches_curgt2rendered, matches_rendered2curgt, matches_prevgt2rendered, matches_rendered2prevgt;
        
            // cur gt -> cur rendered
            int good_matches_curgt2rendered = cur_pkf->matchLROrbGMS(rendered_images[i], 
                kpts_cur_rendered, kpts_cur_gt,
                matches_rendered2curgt, matches_curgt2rendered
            );
            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA cur gt -> cur found "<<good_matches_curgt2rendered<<" good matches."<<std::endl;
        
            // prev gt -> cur rendered
            kpts_cur_rendered.clear();
            int good_matches_prevgt2rendered = prev_pkf->matchLROrbGMS(rendered_images[i], 
                kpts_cur_rendered, kpts_prev_gt,
                matches_rendered2prevgt, matches_prevgt2rendered
            );
            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA prev gt -> cur found "<<good_matches_prevgt2rendered<<" good matches."<<std::endl;
            
            // temp keep it here
            if (good_matches_curgt2rendered == 0) 
                throw std::runtime_error("[GaussianMapper::optimizeLocalLRPoses] No valid matches for cur gt -> cur rendered for fid "+std::to_string(cur_pkf->fid_));

            // related pose
            cv::Mat cur_rvec, cur_tvec, prev_rvec, prev_tvec;
            std::vector<int> cur_inliers, prev_inliers;
            cv::Mat K_lr = (cv::Mat_<float>(3,3) << 
                kf_params_.lr_fx_, 0.f, kf_params_.lr_cx_,
                0.f, kf_params_.lr_fy_, kf_params_.lr_cy_,
                0.f, 0.f, 1.f
            );

            float confidence_curgt2rendered = this->getRelatedPoseGMS(
                cur_pkf,
                kpts_cur_gt, kpts_cur_rendered,
                matches_curgt2rendered,
                gt_depths_cv[i],
                K_lr,
                cur_rvec, cur_tvec,
                cur_inliers
            );
            float confidence_prevgt2rendered = this->getRelatedPoseGMS(
                prev_pkf,
                kpts_prev_gt, kpts_cur_rendered,
                matches_prevgt2rendered,
                gt_depths_cv[i-1],
                K_lr,
                prev_rvec, prev_tvec,
                prev_inliers
            );

            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA related pose confidence for fid "<<cur_pkf->fid_<<" : cur gt -> cur rendered confidence "<<confidence_curgt2rendered<<" with "<<cur_inliers.size()<<"/"<<good_matches_curgt2rendered<<" inliers, prev gt -> cur rendered confidence "<<confidence_prevgt2rendered<<" with "<<prev_inliers.size()<<"/"<<good_matches_prevgt2rendered<<" inliers."<<std::endl;

            // average the two related poses
            torch::Tensor final_delta_pose = torch::eye(4, 4).to(torch::kFloat32).to(device_type_);
            if (confidence_curgt2rendered > 0.f && confidence_prevgt2rendered > 0.f){
                auto cur_delta_pose = general_utils::vecs2transformation(cur_rvec, cur_tvec);
                auto prev_delta_pose = general_utils::vecs2transformation(prev_rvec, prev_tvec);

                auto cur_delta_pose_torch = torch::from_blob(cur_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
                auto prev_delta_pose_torch = torch::from_blob(prev_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);

                // real delta pose of cur, gt (3d) -> cur (2d), gt (3d) <- cur (2d) needs inverse
                cur_delta_pose_torch = cur_delta_pose_torch.inverse();
                prev_delta_pose_torch = prev_delta_pose_torch.inverse();

                // prev_delta_pose_torch = real_prev_delta_pose_torch.mm(prev to cur related pose)
                auto prev_to_cur_related_pose = cur_pkf->getBasePose().mm(prev_pkf->getBasePose().inverse());
                auto real_prev_delta_pose_torch = prev_delta_pose_torch.mm(prev_to_cur_related_pose);

                final_delta_pose = tensor_utils::average_poses({cur_delta_pose_torch, real_prev_delta_pose_torch}, {confidence_curgt2rendered, confidence_prevgt2rendered});
            }
            else if (confidence_curgt2rendered > 0.f){
                auto cur_delta_pose = general_utils::vecs2transformation(cur_rvec, cur_tvec);
                final_delta_pose = torch::from_blob(cur_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
            }
            else if (confidence_prevgt2rendered > 0.f){
                auto prev_delta_pose = general_utils::vecs2transformation(prev_rvec, prev_tvec);
                final_delta_pose = torch::from_blob(prev_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
            }
            else{
                std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA related pose estimation failed for fid "<<cur_pkf->fid_<<", set final delta pose to identity."<<std::endl;
            }
            final_delta_poses.push_back(final_delta_pose);
        }

        // apply the final delta poses
        for (int i=1; i<pkfs.size(); i++){
            if (valid_updates[i-1]) continue;

            auto& pkf = pkfs[i];
            prev_base_poses.push_back(pkf->getBasePose().clone());
            auto updated_base_pose = final_delta_poses[i-1].mm(pkf->getBasePose());
            pkf->setBasePose(updated_base_pose);

            {
                torch::NoGradGuard no_grad;

                auto render_pkg = GaussianRenderer::render(pkf,
                    this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    false, true, true, true
                );

                auto rendered_image = std::get<0>(render_pkg);
                auto rendered_depth = std::get<4>(render_pkg);
                auto rendered_opacity = std::get<5>(render_pkg);

                auto gt_depth_mask = pkf->getGTLRDptMsk(true);  

                auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();
                auto valid_mask = opacity_mask * gt_depth_mask;

                auto loss = loss_utils::get_loss_rgbd(
                    rendered_image, gt_images[i],
                    rendered_depth, gt_depths[i],
                    this->lambdaDssim(),
                    this->global_align_lr_depth_lambda_,
                    pkf->exposure_a_, pkf->exposure_b_,
                    valid_mask,
                    device_type_
                );

                if (loss.item<float>() > losses[i-1]){
                    std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA rendered RGBD LR pose update for fid "<<pkf->fid_<<" increases loss from "<<losses[i-1]<<" to "<<loss.item<float>()<<", revert pose update."<<std::endl;
                    pkf->setBasePose(prev_base_poses[i-1]);
                    valid_updates[i-1] = false;
                }
                else{
                    std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA rendered RGBD LR pose update for fid "<<pkf->fid_<<" decreases loss from "<<losses[i-1]<<" to "<<loss.item<float>()<<", keep pose update."<<std::endl;
                    valid_updates[i-1] = true;
                }

                rendered_images[i] = valid_mask.unsqueeze(0) * rendered_image;
                losses[i-1] = loss.item<float>();

                if(kf_params_.debug_){
                    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose_ba"))

                    auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
                    auto masked_gt_image = gt_images[i] * valid_mask.unsqueeze(0);

                    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose_ba" / (std::to_string(pkf->fid_)+"_"+std::to_string(render_attempts)+"_ba.jpg"), image_cv);
                    if (render_attempts == 0){
                        auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
                        cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                        gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose_ba" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                    }
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }
            }
        }

        // all valid break
        if (std::all_of(valid_updates.begin(), valid_updates.end(), [](bool v){ return v; })){
            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] all BA pose updates are valid after "<<render_attempts<<" attempts, break."<<std::endl;
            break;
        }
        else{ 
            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] not all BA pose updates are valid, re-rendering "<<render_attempts+1<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
            render_attempts++;
        }
    }

   /*

    // local ba for: prev, 0, 1, 2, ..., n-1
    std::vector<torch::Tensor> final_delta_poses;
    for (int i=1; i<pkfs.size(); i++){
        auto& cur_pkf = pkfs[i];
        auto& prev_pkf = pkfs[i-1];

        // pnp linkage 3d -> 2d: prev gt -> cur rendered, cur gt -> cur rendered
        std::vector<cv::KeyPoint> kpts_cur_gt, kpts_prev_gt, kpts_cur_rendered;
        std::vector<int> matches_curgt2rendered, matches_rendered2curgt, matches_prevgt2rendered, matches_rendered2prevgt;
    
        // cur gt -> cur rendered
        int good_matches_curgt2rendered = cur_pkf->matchLROrbGMS(rendered_images[i], 
            kpts_cur_rendered, kpts_cur_gt,
            matches_rendered2curgt, matches_curgt2rendered
        );
        std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA cur gt -> cur rendered for fid "<<cur_pkf->fid_<<" found "<<good_matches_curgt2rendered<<" good matches."<<std::endl;
    
        // prev gt -> cur rendered
        kpts_cur_rendered.clear();
        int good_matches_prevgt2rendered = prev_pkf->matchLROrbGMS(rendered_images[i], 
            kpts_cur_rendered, kpts_prev_gt,
            matches_rendered2prevgt, matches_prevgt2rendered
        );
        std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA prev gt -> cur rendered for fid "<<prev_pkf->fid_<<" found "<<good_matches_prevgt2rendered<<" good matches."<<std::endl;
        
        // related pose
        cv::Mat cur_rvec, cur_tvec, prev_rvec, prev_tvec;
        std::vector<int> cur_inliers, prev_inliers;
        cv::Mat K_lr = (cv::Mat_<float>(3,3) << 
            kf_params_.lr_fx_, 0.f, kf_params_.lr_cx_,
            0.f, kf_params_.lr_fy_, kf_params_.lr_cy_,
            0.f, 0.f, 1.f
        );

        float confidence_curgt2rendered = this->getRelatedPoseGMS(
            cur_pkf,
            kpts_cur_gt, kpts_cur_rendered,
            matches_curgt2rendered,
            gt_depths_cv[i],
            K_lr,
            cur_rvec, cur_tvec,
            cur_inliers
        );

        float confidence_prevgt2rendered = this->getRelatedPoseGMS(
            prev_pkf,
            kpts_prev_gt, kpts_cur_rendered,
            matches_prevgt2rendered,
            gt_depths_cv[i-1],
            K_lr,
            prev_rvec, prev_tvec,
            prev_inliers
        );

        std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA related pose confidence for fid "<<cur_pkf->fid_<<" : cur gt -> cur rendered confidence "<<confidence_curgt2rendered<<" with "<<cur_inliers.size()<<"/"<<good_matches_curgt2rendered<<" inliers, prev gt -> cur rendered confidence "<<confidence_prevgt2rendered<<" with "<<prev_inliers.size()<<"/"<<good_matches_prevgt2rendered<<" inliers."<<std::endl;
        
        // average the two related poses
        torch::Tensor final_delta_pose = torch::eye(4, 4).to(torch::kFloat32).to(device_type_);
        if (confidence_curgt2rendered > 0.f && confidence_prevgt2rendered > 0.f){
            auto cur_delta_pose = general_utils::vecs2transformation(cur_rvec, cur_tvec);
            auto prev_delta_pose = general_utils::vecs2transformation(prev_rvec, prev_tvec);

            auto cur_delta_pose_torch = torch::from_blob(cur_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
            auto prev_delta_pose_torch = torch::from_blob(prev_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);

            // real delta pose of cur, gt (3d) -> cur (2d), gt (3d) <- cur (2d) needs inverse
            cur_delta_pose_torch = cur_delta_pose_torch.inverse();
            prev_delta_pose_torch = prev_delta_pose_torch.inverse();

            // prev_delta_pose_torch = real_prev_delta_pose_torch.mm(prev to cur related pose)
            auto prev_to_cur_related_pose = cur_pkf->getBasePose().mm(prev_pkf->getBasePose().inverse());
            auto real_prev_delta_pose_torch = prev_delta_pose_torch.mm(prev_to_cur_related_pose);

            final_delta_pose = tensor_utils::average_poses({cur_delta_pose_torch, real_prev_delta_pose_torch}, {confidence_curgt2rendered, confidence_prevgt2rendered});

            std::cout<<"[debug] cur_delta_pose_torch:\n"<<cur_delta_pose_torch<<std::endl;
            std::cout<<"[debug] prev_delta_pose_torch:\n"<<prev_delta_pose_torch<<std::endl;
            std::cout<<"[debug] real_prev_delta_pose_torch:\n"<<real_prev_delta_pose_torch<<std::endl;
            std::cout<<"[debug] final_delta_pose:\n"<<final_delta_pose<<std::endl;
        }
        else if (confidence_curgt2rendered > 0.f){
            auto cur_delta_pose = general_utils::vecs2transformation(cur_rvec, cur_tvec);
            final_delta_pose = torch::from_blob(cur_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
        }
        else if (confidence_prevgt2rendered > 0.f){
            auto prev_delta_pose = general_utils::vecs2transformation(prev_rvec, prev_tvec);
            final_delta_pose = torch::from_blob(prev_delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
        }
        else{
            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA related pose estimation failed for fid "<<cur_pkf->fid_<<", set final delta pose to identity."<<std::endl;
        }
        final_delta_poses.push_back(final_delta_pose);
    }
    
    // update poses
    std::vector<torch::Tensor> prev_base_poses;
    for (int i=1; i<pkfs.size(); i++){
        auto& pkf = pkfs[i];
        prev_base_poses.push_back(pkf->getBasePose().clone());
        auto updated_base_pose = final_delta_poses[i-1].mm(pkf->getBasePose());
        pkf->setBasePose(updated_base_pose);
    }

    if(kf_params_.debug_){
        torch::NoGradGuard no_grad;
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose_ba"))

        for (int i=1; i<pkfs.size(); i++){
            auto& pkf = pkfs[i];

            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, true, true, true
            );

            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);

            auto opacity_mask = (rendered_opacity > 0.3f).to(torch::kFloat32).squeeze();

            auto gt_image = pkf->getGTLRImg(true);
            auto gt_depth = pkf->getGTLRDpt(true);
            auto gt_depth_mask = pkf->getGTLRDptMsk(true); 

            auto valid_mask = opacity_mask * gt_depth_mask;

            auto loss = loss_utils::get_loss_rgbd(
                rendered_image, gt_image,
                rendered_depth, gt_depth,
                this->lambdaDssim(),
                this->global_align_lr_depth_lambda_,
                pkf->exposure_a_, pkf->exposure_b_,
                valid_mask,
                device_type_
            );

            auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
            auto masked_gt_image = gt_image * valid_mask.unsqueeze(0);

            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
            cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
            image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose_ba" / (std::to_string(pkf->fid_)+"_ba.jpg"), image_cv);
            auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
            cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
            gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose_ba" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);

            std::cout<<"[GaussianMapper::optimizeLocalLRPoses] BA final pose for fid "<<pkf->fid_<<" loss after BA "<<loss.item<float>()<<", previous loss "<<losses[i-1].item<float>()<<std::endl;
            metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
        }
    }
    // throw std::runtime_error("[GaussianMapper::optimizeLocalLRPoses] not implemented yet!");
*/
    
    return 0.f; 
}

float GaussianMapper::optimizeLocalHRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose)
{
    torch::Tensor loss;
    int hr_height_resize, hr_width_resize;
    float hr_resize_ratio = this->getRsizedHRScale(this->local_align_hr_resize_ratio_, hr_width_resize, hr_height_resize);

    pkf->setGlobalDeltaPose(this->global_align_pose_);

    if(use_differential_pose){

        pkf->resetOptimizer();
        torch::Tensor opacity_mask, gt_image, prev_local_delta_pose;
        float prev_loss = 1e5f;
        bool has_reset_opacity = false;
        for(int i=0; i<this->local_align_hr_pose_iter_; i++){
            auto render_hr_pkg = GaussianRenderer::render(
                pkf,
                hr_height_resize, hr_width_resize,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, true, true, true
            );

            auto rendered_image = std::get<0>(render_hr_pkg);
            auto rendered_depth = std::get<4>(render_hr_pkg);
            auto rendered_opacity = std::get<5>(render_hr_pkg);

            if(i == 0)
                opacity_mask = (rendered_opacity > this->local_align_hr_opacity_thr_).to(torch::kFloat32);

            // linear sampling map
            
            torch::Tensor linear_sampling_map;
            gt_image = pkf->getGTHRImg(linear_sampling_map, hr_resize_ratio, true);

            auto loss_rgb = loss_utils::get_loss_rgb(linear_sampling_map,
                rendered_image, gt_image,
                lambdaDssim(),
                pkf->exposure_a_, pkf->exposure_b_,
                opacity_mask
            );

            auto loss_depth = loss_utils::get_loss_hr2lr(
                linear_sampling_map, rendered_depth.squeeze(), opacity_mask.squeeze(),
                pkf->getGTLRDpt(true), pkf->getGTLRDptMsk(true),
                kf_params_.hr_fx_*hr_resize_ratio, kf_params_.hr_fy_*hr_resize_ratio, 
                kf_params_.hr_cx_*hr_resize_ratio, kf_params_.hr_cy_*hr_resize_ratio,
                kf_params_.lr_fx_, kf_params_.lr_fy_, 
                kf_params_.lr_cx_, kf_params_.lr_cy_,
                pkf->getFullDeltaPose()
            );

            loss = loss_rgb * (1.f-this->local_align_hr_pose_depth_lambda_) + 
                loss_depth * this->local_align_hr_pose_depth_lambda_;
            
            
            /*
            gt_image = pkf->getGTHRImg(hr_resize_ratio, true);
            auto loss_rgb = loss_utils::get_loss_rgb(
                rendered_image, gt_image,
                this->lambdaDssim(),
                pkf->exposure_a_, pkf->exposure_b_,
                opacity_mask,
                device_type_
            );
            loss = loss_rgb;
            auto loss_depth = torch::zeros({1}, torch::kFloat32).to(device_type_);
            */

            if(i>0 && loss.item<float>() > prev_loss * 1.01f){
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] hr pose optimization early stop at iter "<<i<<" fid "<<pkf->fid_<<" loss "<<loss.item<float>()<<" rgb loss "<<loss_rgb.item<float>()<<" depth loss "<<loss_depth.item<float>()<<std::endl;
                pkf->setLocalDeltaPose(prev_local_delta_pose);
                if(kf_params_.debug_){
                    auto rendered_image_ab = rendered_image * torch::exp(pkf->exposure_a_) + pkf->exposure_b_;
                    auto masked_rendered_image = rendered_image_ab * opacity_mask;
                    auto masked_gt_image = gt_image * opacity_mask;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }
                break;
                // if(i>this->local_align_hr_pose_iter_/2 && has_reset_opacity)
                //     break;
                // else{
                //     // pkf->updateOptimizer(0.8f);
                //     opacity_mask = (rendered_opacity > this->local_align_hr_opacity_thr_).to(torch::kFloat32);
                //     prev_loss = loss.item<float>() * 1.2f;
                //     has_reset_opacity = true;
                //     continue;
                // }
            }
            else{ 
                prev_loss = loss.item<float>();
                prev_local_delta_pose = pkf->getLocalDeltaPose().clone();
            }
            
            loss.backward();

            {
                torch::NoGradGuard no_grad;

                gaussians_->optimizer_->zero_grad(true);

                pkf->stepOptimizer(true, false);
                pkf->zeroOptimizerGrad(true, true);
                pkf->updateLocalDeltaPose(true);

                if(kf_params_.debug_){
                    std::cout<<"[debug] fid "<<pkf->fid_<<" hr iter "<<i<<" loss "<<loss.item<float>()<<" rgb loss "<<loss_rgb.item<float>()<<" depth loss "<<loss_depth.item<float>()<<std::endl;
                    auto rendered_image_ab = rendered_image * torch::exp(pkf->exposure_a_) + pkf->exposure_b_;
                    auto masked_rendered_image = rendered_image_ab * opacity_mask;
                    auto masked_gt_image = gt_image * opacity_mask;

                    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose"))
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(pkf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                
                    // if((i+1) % 5 == 0){
                        std::cout<<"[debug] metrics at hr pose iter "<<i<<std::endl;
                        metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                    // }

                    if(i == 0){
                        // std::cout<<"[debug] metrics at hr pose iter "<<i<<std::endl;
                        // metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                        auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
                        cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                        gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                    }
                }
            }
        }
    }
    // auto iter_start_timing2 = std::chrono::steady_clock::now();
    else{
        hr_resize_ratio = this->getRsizedHRScale(1.f, hr_width_resize, hr_height_resize);
        torch::Tensor rendered_image, rendered_depth, opacity_mask, gt_image;
        torch::Tensor local_delta_pose1, local_delta_pose2;
        torch::Tensor updated_loss, final_loss;

        cv::Mat K_hr = (cv::Mat_<float>(3,3) <<
            kf_params_.hr_fx_ * hr_resize_ratio, 0.f, kf_params_.hr_cx_ * hr_resize_ratio,
            0.f, kf_params_.hr_fy_ * hr_resize_ratio, kf_params_.hr_cy_ * hr_resize_ratio,
            0.f, 0.f, 1.f
        );

        float confidence = -1.f;
        int render_attempts = 0;
        while(confidence <= 0.f && render_attempts < this->max_pnp_render_attempts_){
            {
                torch::NoGradGuard no_grad;

                auto render_hr_pkg = GaussianRenderer::render(
                    pkf,
                    hr_height_resize, hr_width_resize,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    true, true, true, true
                );

                rendered_image = std::get<0>(render_hr_pkg);
                rendered_depth = std::get<4>(render_hr_pkg);
                auto rendered_opacity = std::get<5>(render_hr_pkg);

                opacity_mask = (rendered_opacity > this->local_align_hr_opacity_thr_).to(torch::kFloat32).squeeze();
                
                gt_image = pkf->getGTHRImg(hr_resize_ratio, true).cuda();

                loss = loss_utils::get_loss_rgb(
                    rendered_image, gt_image,
                    this->lambdaDssim(),
                    pkf->exposure_a_, pkf->exposure_b_,
                    opacity_mask,
                    device_type_
                );
            }

            if(kf_params_.debug_){
                auto masked_rendered_image = rendered_image * opacity_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_before.jpg"), image_cv);
                auto masked_gt_image = gt_image * opacity_mask.unsqueeze(0);
                auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
                cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_cv);
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] classic pose fid "<<pkf->fid_<<" loss before optimization: "<<loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }

            std::vector<cv::KeyPoint> kpts_rendered, kpts_gt;
            std::vector<int> matches_rendered2gt, matches_gt2rendered;
            int good_matches = pkf->matchHROrbGMS(rendered_image, hr_resize_ratio,
                kpts_rendered, kpts_gt,
                matches_rendered2gt, matches_gt2rendered
            );

            auto depth_cpu = rendered_depth.squeeze().to(torch::kCPU).contiguous();
            cv::Mat depth_cv(depth_cpu.size(0), depth_cpu.size(1), CV_32F);
            std::memcpy(depth_cv.data, depth_cpu.data_ptr<float>(), sizeof(float)*depth_cpu.size(0)*depth_cpu.size(1));

            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            confidence = this->getRelatedPoseGMS(
                pkf,
                kpts_rendered, kpts_gt,
                matches_rendered2gt,
                depth_cv,
                K_hr,
                rvec, tvec,
                inliers,
                false
            );

            if(confidence > 0.f){
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] rendered RGBD pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                    <<" inliers "<<inliers.size()<<"/"<<good_matches<<std::endl;

                auto delta_pose = general_utils::vecs2transformation(rvec, tvec);
                auto delta_pose_torch = torch::from_blob(delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
                // delta_pose_torch = delta_pose_torch.inverse();
                std::cout<<"[debug] before pose "<<pkf->getLocalDeltaPose()<<std::endl;
                auto updated_local_pose = delta_pose_torch.mm(pkf->getLocalDeltaPose());
                std::cout<<"[debug] delta pose "<<delta_pose_torch<<std::endl;
                std::cout<<"[debug] updated pose "<<updated_local_pose<<std::endl;
                local_delta_pose1 = pkf->getLocalDeltaPose().clone();
                pkf->setLocalDeltaPose(updated_local_pose);
                local_delta_pose2 = pkf->getLocalDeltaPose().clone();
                break;
            }
            else{
                // std::cout<<"[GaussianMapper::optimizeLocalHRPose] rendered RGBD HR pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                //     <<" inliers "<<inliers.size()<<"/"<<good_matches<<", skip pose update."<<std::endl;
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed, re-rendering "<<render_attempts<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
                render_attempts++;
                // return loss.item<float>();
            }
        }

        if(confidence <= 0.f){
            std::cout<<"[error][GaussianMapper::optimizeLocalHRPose] For rendered RGBD HR, PnP pose estimation for fid "<<pkf->fid_<<" failed after "<<this->max_pnp_render_attempts_<<" attempts!"<<std::endl;
            throw std::runtime_error("[GaussianMapper::optimizeLocalHRPose] PnP pose estimation failed!");
            return loss.item<float>();
        }
        else{
            confidence = -1.f;
            render_attempts = 0;
        }

        while(confidence <= 0.f && render_attempts < this->max_pnp_render_attempts_){
            {
                torch::NoGradGuard no_grad;

                auto render_hr_pkg = GaussianRenderer::render(
                    pkf,
                    hr_height_resize, hr_width_resize,
                    this->gaussians_, this->pipe_params_,
                    this->background_, this->override_color_,
                    true, true, true, true
                );

                rendered_image = std::get<0>(render_hr_pkg);
                rendered_depth = std::get<4>(render_hr_pkg);
                auto rendered_opacity = std::get<5>(render_hr_pkg);

                opacity_mask = (rendered_opacity > this->local_align_hr_opacity_thr_).to(torch::kFloat32).squeeze();
                
                updated_loss = loss_utils::get_loss_rgb(
                    rendered_image, gt_image,
                    this->lambdaDssim(),
                    pkf->exposure_a_, pkf->exposure_b_,
                    opacity_mask,
                    device_type_
                );

                if(updated_loss.item<float>() > loss.item<float>()){
                    std::cout<<"[GaussianMapper::optimizeLocalHRPose] rendered RGBD HR pose update increases loss from "<<loss.item<float>()<<" to "<<updated_loss.item<float>()<<", revert pose update."<<std::endl;
                    pkf->setLocalDeltaPose(local_delta_pose1);
                }
            }    

            if(kf_params_.debug_){
                auto masked_rendered_image = rendered_image * opacity_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_after.jpg"), image_cv);
                auto masked_gt_image = gt_image * opacity_mask.unsqueeze(0);
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] rendered RGBD HR pose fid "<<pkf->fid_<<" loss after optimization: "<<updated_loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
            
            std::vector<cv::KeyPoint> kpts_rendered, kpts_gt;
            std::vector<int> matches_rendered2gt, matches_gt2rendered;
            int good_matches = pkf->matchHROrbGMS(rendered_image, hr_resize_ratio,
                kpts_rendered, kpts_gt,
                matches_rendered2gt, matches_gt2rendered
            );

            auto depth_cpu = rendered_depth.squeeze().to(torch::kCPU).contiguous();
            cv::Mat depth_cv(depth_cpu.size(0), depth_cpu.size(1), CV_32F);
            std::memcpy(depth_cv.data, depth_cpu.data_ptr<float>(), sizeof(float)*depth_cpu.size(0)*depth_cpu.size(1));

            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            confidence = this->getRelatedPoseGMS(
                pkf,
                kpts_rendered, kpts_gt,
                matches_rendered2gt,
                depth_cv,
                K_hr,
                rvec, tvec,
                inliers,
                false,
                hr_resize_ratio,
                true
            );

            if(confidence > 0.f){
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] HRLR cross RGBD HR pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                    <<" inliers "<<inliers.size()<<"/"<<good_matches
                    <<" loss before "<<loss.item<float>()
                    <<" loss after "<<updated_loss.item<float>()<<std::endl;

                auto delta_pose = general_utils::vecs2transformation(rvec, tvec);
                auto delta_pose_torch = torch::from_blob(delta_pose.ptr<float>(), {4, 4}, torch::kFloat32).to(device_type_);
                // delta_pose_torch = delta_pose_torch.inverse();
                std::cout<<"[debug] before pose "<<pkf->getLocalDeltaPose()<<std::endl;
                auto updated_local_pose = delta_pose_torch.mm(pkf->getLocalDeltaPose());
                std::cout<<"[debug] delta pose "<<delta_pose_torch<<std::endl;
                std::cout<<"[debug] updated pose "<<updated_local_pose<<std::endl;
                pkf->setLocalDeltaPose(updated_local_pose);
                break;
            }
            else{
                // std::cout<<"[GaussianMapper::optimizeLocalHRPose] HRLR cross RGBD HR pose fid "<<pkf->fid_<<" GMS confidence "<<confidence
                //     <<" inliers "<<inliers.size()<<"/"<<good_matches<<", skip pose update."<<std::endl;
                // return updated_loss.item<float>();

                std::cout<<"[GaussianMapper::optimizeLocalHRPose] PnP pose estimation for fid "<<pkf->fid_<<" failed, re-rendering "<<render_attempts<<"/"<<this->max_pnp_render_attempts_<<"..."<<std::endl;
                render_attempts++;
            }
        }

        if(confidence <= 0.f){
            std::cout<<"[error][GaussianMapper::optimizeLocalHRPose] For HRLR cross RGBD HR pose, PnP pose estimation for fid "<<pkf->fid_<<" failed after "<<this->max_pnp_render_attempts_<<" attempts!"<<std::endl;
            return updated_loss.item<float>();
        }

        {
            torch::NoGradGuard no_grad;

            auto render_hr_pkg = GaussianRenderer::render(
                pkf,
                hr_height_resize, hr_width_resize,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, true, true, true
            );

            rendered_image = std::get<0>(render_hr_pkg);
            auto rendered_opacity = std::get<5>(render_hr_pkg);
            opacity_mask = (rendered_opacity > this->local_align_hr_opacity_thr_).to(torch::kFloat32).squeeze();

            final_loss = loss_utils::get_loss_rgb(
                rendered_image, gt_image,
                this->lambdaDssim(),
                pkf->exposure_a_, pkf->exposure_b_,
                opacity_mask,
                device_type_
            );

            if(final_loss.item<float>() > updated_loss.item<float>()){
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] HRLR cross RGBD HR pose update increases loss from "<<updated_loss.item<float>()<<" to "<<final_loss.item<float>()<<", revert pose update."<<std::endl;
                pkf->setLocalDeltaPose(local_delta_pose2);
            }

            if(kf_params_.debug_){
                auto masked_rendered_image = rendered_image * opacity_mask.unsqueeze(0);
                auto masked_gt_image = gt_image * opacity_mask.unsqueeze(0);
                std::cout<<"[GaussianMapper::optimizeLocalHRPose] final metrics fid "<<pkf->fid_<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                cv::Mat image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(pkf->fid_)+"_final.jpg"), image_cv);
            }   
        }
    }

    return loss.item<float>();
}

float GaussianMapper::getRsizedHRScale(float ratio, int& out_width, int& out_height){
    float out_ratio;
    if(ratio >= 1.f-1e-5f){
        out_height = this->kf_params_.hr_height_;
        out_width = this->kf_params_.hr_width_;
        out_ratio = 1.f;
    }
    else{
        out_height = int(floor(float(this->kf_params_.hr_height_) * ratio));
        out_width = int(floor(float(this->kf_params_.hr_width_) * ratio));
        out_ratio = ratio;
    }
    return out_ratio;
}

torch::Tensor GaussianMapper::getLocalLRValidDptMsk(std::shared_ptr<GaussianKeyframe> pkf){
    int kernel_size = 3;

    torch::Tensor rendered_image, rendered_opacity, valid_mask;

    // {
    //     std::unique_lock<std::mutex> lock_render(mutex_render_);
        torch::NoGradGuard no_grad;
        auto render_pkg = GaussianRenderer::render(pkf,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, true, true, true
        );

        rendered_image = std::get<0>(render_pkg);
        rendered_opacity = std::get<5>(render_pkg);
        
        // auto gt_image = pkf->getGTLRImg(true);
        auto gt_depth = pkf->getGTLRDpt(true);
        auto gt_depth_mask = pkf->getGTLRDptMsk(true);

        auto opacity_mask = (rendered_opacity < this->local_align_batch_fillholes_opacity_thr_).to(torch::kFloat32).squeeze();

        valid_mask = general_utils::dilate_mask(opacity_mask, 3) * gt_depth_mask;
    // }

    auto labeled_mask = cc_torch::connected_components_labeling_2d(valid_mask.to(torch::kUInt8));
    auto filtered_mask = general_utils::merge_large_components(labeled_mask, this->local_align_batch_conn_comp_min_size_);

    // auto valid_depth = gt_depth * filtered_mask;

    // pkf->setGlobalDeltaPose(global_delta_pose);
    // pkf->setLocalDeltaPose(local_delta_pose);

    // this->gaussians_->increaseKeyframeInitPcd(pkf, valid_depth, gt_image, this->kf_params_);

    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_valid_dpt"))
        labeled_mask = labeled_mask.to(torch::kFloat32);
        labeled_mask = labeled_mask / labeled_mask.max();
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(labeled_mask);
        image_cv.convertTo(image_cv, CV_8UC1, 255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_valid_dpt" / (std::to_string(pkf->fid_) + "_labeled.jpg"), image_cv);
        auto image_cv2 = tensor_utils::torchTensor2CvMat_Float32(valid_mask);
        image_cv2.convertTo(image_cv2, CV_8UC1, 255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_valid_dpt" / (std::to_string(pkf->fid_) + "_valid.jpg"), image_cv2);
        auto image_cv3 = tensor_utils::torchTensor2CvMat_Float32(filtered_mask);
        image_cv3.convertTo(image_cv3, CV_8UC1, 255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_valid_dpt" / (std::to_string(pkf->fid_) + "_filtered.jpg"), image_cv3);
        
        auto image_cv4 = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
        cv::cvtColor(image_cv4, image_cv4, CV_RGB2BGR);
        image_cv4.convertTo(image_cv4, CV_8UC3, 255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_valid_dpt" / (std::to_string(pkf->fid_) + "_image.jpg"), image_cv4);    
    }

    // auto filtered_mask_cpu = filtered_mask.to(torch::kCPU).contiguous();
    // return filtered_mask_cpu;
    return filtered_mask;
}

int GaussianMapper::insertLocalLRValidDpts(std::vector<torch::Tensor>& valid_depth_masks, std::vector<std::size_t>& valid_fids){
    // std::unique_lock<std::mutex> lock_render(mutex_render_);
    torch::NoGradGuard no_grad;
    
    std::vector<std::size_t> sorted_valid_fids;
    std::vector<torch::Tensor> sorted_valid_depth_masks;
    std::vector<std::pair<int, int>> fid_pixel_counts;

    // Build a map from fid to index for valid_depth_masks
    std::unordered_map<std::size_t, int> fid_to_mask_idx;
    for (int i = 0; i < valid_fids.size(); ++i) {
        fid_to_mask_idx[valid_fids[i]] = i;
        int pixel_count = valid_depth_masks[i].sum().item<int>();
        fid_pixel_counts.emplace_back(valid_fids[i], pixel_count);
    }

    if(fid_pixel_counts.size() > 1)
        std::sort(fid_pixel_counts.begin(), fid_pixel_counts.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second; // larger pixel count first
            });

    for (const auto& [fid, count] : fid_pixel_counts)
        std::cout << "[GaussianMapper::insertLocalLRValidDpts] fid " << fid << " has " << count << " valid depth pixels." << std::endl;

    int max_pixel_fid = fid_pixel_counts.front().first;
    int max_pixel_count = fid_pixel_counts.front().second;

    if(max_pixel_count < 1){
        std::cout<<"[GaussianMapper::insertLocalLRValidDpts] no valid depth pixels found in the batch!"<<std::endl;
        return -1;
    }

    // auto& pkf_ref = this->scene_->keyframes_.at(max_pixel_fid);
    auto pkf_ref = this->scene_->getKeyframe(max_pixel_fid); // [debug 20260222]
    auto pose_ref = pkf_ref->getBasePose().clone();
    auto valid_depth_mask_ref = valid_depth_masks[fid_to_mask_idx[max_pixel_fid]];
    // auto valid_depth_ref = pkf_ref->getGTLRDpt(true);
    auto valid_depth_ref = valid_depth_mask_ref * pkf_ref->getGTLRDpt(true);
    auto gt_image_ref = pkf_ref->getGTLRImg(true);
    this->gaussians_->increaseKeyframeInitPcd(pkf_ref, valid_depth_ref, gt_image_ref, this->kf_params_);

    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff"))
        auto ref_valid_depth_mask_cv = tensor_utils::torchTensor2CvMat_Float32(valid_depth_mask_ref);
        ref_valid_depth_mask_cv.convertTo(ref_valid_depth_mask_cv, CV_8UC1, 255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(max_pixel_fid) + "_main_mask.jpg"), ref_valid_depth_mask_cv);
        auto masked_image = gt_image_ref * valid_depth_mask_ref;
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_image);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(max_pixel_fid) + "_main_image.jpg"), image_cv);
        torch::Tensor opacity_mask, gt_depth;
        {
            torch::NoGradGuard no_grad;
            auto render_pkg = GaussianRenderer::render(pkf_ref,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, true, true, true
            );
            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);
            opacity_mask = (rendered_opacity > 0.1f).to(torch::kFloat32).squeeze();
            auto rendered_image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
            gt_depth = pkf_ref->getGTLRDpt(true);
            cv::cvtColor(rendered_image_cv, rendered_image_cv, CV_RGB2BGR);
            rendered_image_cv.convertTo(rendered_image_cv, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(max_pixel_fid) + "_main_render.jpg"), rendered_image_cv);
            auto depth_diff = torch::abs(rendered_depth - gt_depth) * this->rendered_depthmap_factor_;
            depth_diff = depth_diff * opacity_mask;
            auto depth_diff_cv = tensor_utils::torchTensor2CvMat_Float32(depth_diff);
            depth_diff_cv.convertTo(depth_diff_cv, CV_8UC1, 255.0f/2000.f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(max_pixel_fid) + "_main_depth_diff.jpg"), depth_diff_cv); 
        }
        /*
        {
            auto valid_indices = torch::where(valid_depth_ref > 0.f);
            auto valid_v = valid_indices[0];
            auto valid_u = valid_indices[1];
            auto linear_idx = valid_v * kf_params_.lr_width_ + valid_u;

            auto z = valid_depth_ref.flatten().index_select(0, linear_idx);
            auto x = (valid_u.to(torch::kFloat32) - kf_params_.lr_cx_) * z * (1.f / kf_params_.lr_fx_);
            auto y = (valid_v.to(torch::kFloat32) - kf_params_.lr_cy_) * z * (1.f / kf_params_.lr_fy_);

            auto Pc = torch::stack({x, y, z}, 1); // N x 3
            auto Pc_homo = torch::cat({Pc, torch::ones({Pc.size(0), 1}, Pc.options())}, 1); // N x 4
            auto Pw = pkf_ref->getBasePose().inverse().mm(Pc_homo.transpose(0, 1)); // 4 x N
            auto Pc_back = pkf_ref->getBasePose().mm(Pw); // 4 x N
            auto x_back = Pc_back.index({torch::indexing::Slice(0, 1), torch::indexing::Slice()}).squeeze(); // N
            auto y_back = Pc_back.index({torch::indexing::Slice(1, 2), torch::indexing::Slice()}).squeeze(); // N
            auto z_back = Pc_back.index({torch::indexing::Slice(2, 3), torch::indexing::Slice()}).squeeze(); // N
            std::cout<<"[debug] x_back sizes: "<<x_back.sizes()<<std::endl;

            auto u_back = (x_back / z_back * kf_params_.lr_fx_ + kf_params_.lr_cx_).round().to(torch::kInt32);
            auto v_back = (y_back / z_back * kf_params_.lr_fy_ + kf_params_.lr_cy_).round().to(torch::kInt32);

            u_back = torch::clamp(u_back, 0, kf_params_.lr_width_ - 1);
            v_back = torch::clamp(v_back, 0, kf_params_.lr_height_ - 1);
            auto depth_back = torch::zeros({kf_params_.lr_height_, kf_params_.lr_width_}, z_back.options());
            depth_back.index_put_({v_back, u_back}, z_back);
            std::cout<<"[debug] depth back mean "<<depth_back.mean().item<float>()<<" gt depth mean "<<gt_depth.mean().item<float>()<<std::endl;
            auto depth_diff = torch::abs(depth_back - gt_depth);
            depth_diff = depth_diff * opacity_mask;
            auto depth_diff_cv = tensor_utils::torchTensor2CvMat_Float32(depth_diff);
            depth_diff_cv = depth_diff_cv * this->rendered_depthmap_factor_;
            depth_diff_cv.convertTo(depth_diff_cv, CV_8UC1, 255.0f/2000.f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(max_pixel_fid) + "_back_depth_diff.jpg"), depth_diff_cv); 
            auto depth_back_cv = tensor_utils::torchTensor2CvMat_Float32(depth_back);
            depth_back_cv = depth_back_cv * this->rendered_depthmap_factor_;
            depth_back_cv.convertTo(depth_back_cv, CV_8UC1, 255.0f/2000.f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(max_pixel_fid) + "_back_depth.jpg"), depth_back_cv);
        }
        */
    }

    torch::Tensor accumulated_valid_depth = valid_depth_mask_ref.flatten();
    for(const auto& [kfid, count] : fid_pixel_counts){   
        sorted_valid_fids.push_back(kfid);
        if(kfid == max_pixel_fid) continue;

        // auto& pkf = this->scene_->keyframes_.at(kfid);
        auto pkf = this->scene_->getKeyframe(kfid); // [debug 20260222]

        auto gt_depth = pkf->getGTLRDpt(true);
        auto gt_image = pkf->getGTLRImg(true);

        // Use the correct mask index for this kfid
        int mask_idx = fid_to_mask_idx[kfid];
        auto valid_depth = gt_depth * valid_depth_masks[mask_idx];

        cv::Mat valid_depth_to_ref = this->getDepthRelated(pkf_ref, pkf, pose_ref, pkf->getBasePose(), valid_depth);
        auto valid_depth_to_ref_tensor = torch::from_blob(
            valid_depth_to_ref.data, 
            {valid_depth_to_ref.rows, valid_depth_to_ref.cols}, 
            torch::kFloat32
        ).clone();
        valid_depth_to_ref_tensor = valid_depth_to_ref_tensor.to(device_type_);
        auto valid_depth_to_ref_flatten = valid_depth_to_ref_tensor.flatten();

        auto and_mask = (valid_depth_to_ref_flatten > 1e-2f) & (accumulated_valid_depth > 0.f);
        auto xor_mask = (valid_depth_to_ref_flatten > 1e-2f) & (and_mask == 0);

        auto xor_mask_reshaped = xor_mask.reshape({valid_depth_to_ref.rows, valid_depth_to_ref.cols});
        auto labeled_mask = cc_torch::connected_components_labeling_2d(xor_mask_reshaped.to(torch::kUInt8));
        auto filtered_mask = general_utils::merge_large_components(labeled_mask, this->local_align_batch_conn_comp_min_size_/2);

        if(filtered_mask.sum().item<int>() < 1){
            std::cout<<"[GaussianMapper::insertLocalLRValidDpts] fid "<<kfid<<" has no new valid depth pixels after filtering, skip."<<std::endl;
            continue;
        }
        else{
            std::cout<<"[GaussianMapper::insertLocalLRValidDpts] fid "<<kfid<<" has "<<filtered_mask.sum().item<int>()<<" new valid depth pixels after filtering, insert."<<std::endl;
            
            // auto filtered_depth_at_ref = filtered_mask * valid_depth_to_ref_tensor;
            cv::Mat back_project_depth = this->getDepthRelated(pkf, pkf_ref, pkf->getBasePose(), pose_ref, filtered_mask);
            auto back_project_depth_tensor = torch::from_blob(
                back_project_depth.data, 
                {back_project_depth.rows, back_project_depth.cols}, 
                torch::kFloat32
            ).clone();
            back_project_depth_tensor = back_project_depth_tensor.to(device_type_);
            auto back_project_depth_mask = (back_project_depth_tensor > 1e-2f).to(torch::kFloat32);
            auto insert_depth = gt_depth * back_project_depth_mask;
            // auto insert_image = gt_image * back_project_depth_mask;
            this->gaussians_->increaseKeyframeInitPcd(pkf, insert_depth, gt_image, this->kf_params_);

            if(kf_params_.debug_){
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff"))
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(back_project_depth_mask);
                image_cv.convertTo(image_cv, CV_8UC1, 255.0f, 0.f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_backproj.jpg"), image_cv);
                cv::Mat insert_depth_cv = tensor_utils::torchTensor2CvMat_Float32(insert_depth) * this->rendered_depthmap_factor_;
                insert_depth_cv.convertTo(insert_depth_cv, CV_8UC1, 255.0f/3000.f, 0.f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_insert_depth.jpg"), insert_depth_cv);
            }
        }

        accumulated_valid_depth = (accumulated_valid_depth > 0.f).to(torch::kFloat32) + filtered_mask.flatten().to(torch::kFloat32);
        accumulated_valid_depth = torch::clamp(accumulated_valid_depth, 0.f, 1.f);

        if(kf_params_.debug_){
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff"))

            auto valid_depth_cv = tensor_utils::torchTensor2CvMat_Float32(valid_depth);
            valid_depth_cv.convertTo(valid_depth_cv, CV_8UC1, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_valid.jpg"), valid_depth_cv);
            auto diff_depth_mask_reshaped = and_mask.reshape({valid_depth_to_ref.rows, valid_depth_to_ref.cols}).to(torch::kFloat32);
            auto diff_depth_mask_cv = tensor_utils::torchTensor2CvMat_Float32(diff_depth_mask_reshaped);
            diff_depth_mask_cv.convertTo(diff_depth_mask_cv, CV_8UC1, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_inmask.jpg"), diff_depth_mask_cv);
            auto diff_depth_mask_reshaped2 = xor_mask.reshape({valid_depth_to_ref.rows, valid_depth_to_ref.cols}).to(torch::kFloat32);
            auto diff_depth_mask_cv2 = tensor_utils::torchTensor2CvMat_Float32(diff_depth_mask_reshaped2);
            diff_depth_mask_cv2.convertTo(diff_depth_mask_cv2, CV_8UC1, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_addmask.jpg"), diff_depth_mask_cv2);
            auto diff_depth_mask_reshaped3 = filtered_mask.to(torch::kFloat32);
            auto diff_depth_mask_cv3 = tensor_utils::torchTensor2CvMat_Float32(diff_depth_mask_reshaped3);
            diff_depth_mask_cv3.convertTo(diff_depth_mask_cv3, CV_8UC1, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_finalmask.jpg"), diff_depth_mask_cv3);
            auto accumulated_valid_depth_mask_reshaped = accumulated_valid_depth.reshape({valid_depth_to_ref.rows, valid_depth_to_ref.cols}).to(torch::kFloat32);
            auto acc_cv = tensor_utils::torchTensor2CvMat_Float32(accumulated_valid_depth_mask_reshaped);
            acc_cv.convertTo(acc_cv, CV_8UC1, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(kfid) + "_acc_mask.jpg"), acc_cv);
        }

    }

    valid_fids = sorted_valid_fids;
    for (int i=0; i<valid_fids.size(); ++i) {
        sorted_valid_depth_masks.push_back(valid_depth_masks[fid_to_mask_idx[valid_fids[i]]]);
    }
    valid_depth_masks = sorted_valid_depth_masks;

    if(kf_params_.debug_){
        torch::NoGradGuard no_grad;
        for (int i=0; i<valid_fids.size(); ++i) {
            // auto& pkf = this->scene_->keyframes_.at(valid_fids[i]);
            auto pkf = this->scene_->getKeyframe(valid_fids[i]); // [debug 20260222]
            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, true, true, true
            );

            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);
            auto opacity_mask = (rendered_opacity > 0.01f).to(torch::kFloat32).squeeze();
            auto masked_rendered_image = rendered_image * opacity_mask.unsqueeze(0);
            auto gt_image = pkf->getGTLRImg(true);
            auto masked_gt_image = gt_image * opacity_mask.unsqueeze(0);

            std::cout<<"[GaussianMapper::insertLocalLRValidDpts] final metrics fid "<<pkf->fid_<<std::endl;
            metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);

            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
            cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
            image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_depth_diff" / (std::to_string(pkf->fid_) + "_final.jpg"), image_cv);
        }

    }

    return max_pixel_fid;
}

void GaussianMapper::optimizeInsertedLocalLRDpts(
    std::vector<std::size_t>& valid_fids, std::vector<torch::Tensor>& valid_depth_masks
){
    // std::unique_lock<std::mutex> lock_render(mutex_render_);
    // float main_fid_ratio = 0.4f;
    float add_loss_weight = 0.6f;

    std::vector<torch::Tensor> opacity_masks(valid_fids.size(), torch::Tensor());
    std::vector<torch::Tensor> init_base_poses(valid_fids.size(), torch::Tensor());
    std::vector<torch::Tensor> joint_gs_masks(valid_fids.size(), torch::Tensor());
    bool fix_xyz_geo = false;
    for(int i=0; i<this->local_align_lr_joint_pose_iter_; ++i){
        torch::Tensor batch_loss = torch::zeros({1}, torch::TensorOptions().device(device_type_).dtype(torch::kFloat32));
        for(int idx=0; idx<valid_fids.size(); ++idx){
            // auto& pkf = this->scene_->keyframes_.at(valid_fids[idx]);
            auto pkf = this->scene_->getKeyframe(valid_fids[idx]); // [debug 20260222]

            if(i == 0) pkf->updateOptimizer(0.6f);

            if(i >= this->local_align_lr_joint_pose_iter_/2 && fix_xyz_geo) fix_xyz_geo = false;
            
            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, fix_xyz_geo, fix_xyz_geo, true
            );

            auto rendered_image = std::get<0>(render_pkg);
            auto rendered_depth = std::get<4>(render_pkg);
            auto rendered_opacity = std::get<5>(render_pkg);
            auto gt_image = pkf->getGTLRImg(true);
            auto gt_depth = pkf->getGTLRDpt(true);
            auto gt_depth_mask = pkf->getGTLRDptMsk(true);

            torch::Tensor valid_mask;
            if(i == 0){
                auto opacity_mask = (rendered_opacity > 0.01f).to(torch::kFloat32).squeeze();
                opacity_masks[idx] = opacity_mask;
                valid_mask = gt_depth_mask * opacity_mask;

                init_base_poses[idx] = pkf->getBasePose().clone();
            }
            else valid_mask = gt_depth_mask * opacity_masks[idx];
            // auto opacity_mask = (rendered_opacity > 0.01f).to(torch::kFloat32).squeeze();
            // auto valid_mask = gt_depth_mask * opacity_mask;

            // auto loss = loss_utils::get_loss_depth(
            //     rendered_depth, gt_depth,
            //     valid_mask,
            //     device_type_
            // );

            auto full_loss = loss_utils::get_loss_rgbd(
                rendered_image, gt_image,
                rendered_depth, gt_depth,
                0.8f, 0.6f,
                pkf->exposure_a_, pkf->exposure_b_,
                valid_mask
            );

            auto added_loss = loss_utils::get_loss_rgbd(
                rendered_image, gt_image,
                rendered_depth, gt_depth,
                0.8f, 0.6f,
                pkf->exposure_a_, pkf->exposure_b_,
                valid_mask * valid_depth_masks[idx]
            );

            auto loss = (1.f - add_loss_weight) * full_loss + add_loss_weight * added_loss;
            std::cout<<"[debug 02220043] loss "<<loss.item<float>()<<" full_loss "<<full_loss.item<float>()<<" added_loss "<<added_loss.item<float>()<<" id "<<pkf->fid_<<" iter "<<i<<std::endl;

            // if(idx == 0) loss *= main_fid_ratio;
            // else loss *= (1.f-main_fid_ratio) * (1.f/float(valid_fids.size()-1));
            // loss *= (1.f/float(valid_fids.size()));

            if(i==0) batch_loss += loss;
            else{
                auto updated_delta_pose = pkf->getBasePose() * init_base_poses[idx].inverse();
                auto pose_loss = loss_utils::get_loss_posereg(updated_delta_pose, 0.5f);
                // auto pose_loss = loss_utils::get_loss_posereg(updated_delta_pose, 0.7f) * (1.f/float(valid_fids.size()));
                batch_loss += (loss + 10.f * pose_loss);
            }

            batch_loss += loss;
        }

        batch_loss = batch_loss / float(valid_fids.size());
        batch_loss.backward();

        {
            torch::NoGradGuard no_grad;

            // if(i >= this->local_align_lr_joint_pose_iter_/2)
            // gaussians_->updateBatchGradients(valid_fids, 1.8f);
            
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);

            for(int idx=0; idx<valid_fids.size(); ++idx){
                // auto& pkf = this->scene_->keyframes_.at(valid_fids[idx]);
                auto pkf = this->scene_->getKeyframe(valid_fids[idx]); // [debug 20260222]
                
                pkf->stepOptimizer(true, false);
                pkf->zeroOptimizerGrad(true, true);
                
                auto delta_pose = torch::eye(4, torch::TensorOptions().device(device_type_).dtype(torch::kFloat32));
                bool has_updated = pkf->updateBasePose(delta_pose, true);
                if (!has_updated) continue;
                auto updated_pose = pkf->getBasePose().clone();
                // std::cout<<"[debug] updated getBasePose\n"<<updated_pose<<std::endl;
                // std::cout<<"[debug] delta getBasePose\n"<<delta_pose<<std::endl;

                this->gaussians_->updateKeyframeJointPcd(pkf, updated_pose, delta_pose, joint_gs_masks[idx]);


            }

            if(kf_params_.debug_){
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_depth"))
                for(int idx=0; idx<valid_fids.size(); ++idx){
                    // auto& pkf = this->scene_->keyframes_.at(valid_fids[idx]);
                    auto pkf = this->scene_->getKeyframe(valid_fids[idx]); // [debug 20260222]
                    auto render_pkg = GaussianRenderer::render(pkf,
                        this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                        this->gaussians_, this->pipe_params_,
                        this->background_, this->override_color_,
                        false, false, false, false
                    );
                    auto rendered_depth = std::get<4>(render_pkg);
                    auto rendered_image = std::get<0>(render_pkg);
                    auto rendered_opacity = std::get<5>(render_pkg);
                    auto gt_depth = pkf->getGTLRDpt(true);
                    auto gt_depth_mask = pkf->getGTLRDptMsk(true);
                    auto gt_image = pkf->getGTLRImg(true);

                    torch::Tensor opacity_mask = (rendered_opacity > 0.01f).to(torch::kFloat32).squeeze();

                    auto valid_mask = gt_depth_mask * opacity_mask;

                    rendered_depth = rendered_depth.squeeze();
                    
                    auto masked_rendered_image = rendered_image * valid_mask.unsqueeze(0);
                    auto masked_gt_image = gt_image * valid_mask.unsqueeze(0);

                    // cv::Mat depth_image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_depth) * this->rendered_depthmap_factor_;
                    // depth_image_cv.convertTo(depth_image_cv, CV_8UC1, 255.0f/3000.f, 0.f);
                    // cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_depth" / (std::to_string(pkf->fid_)+"_"+std::to_string(i)+"_depth.jpg"), depth_image_cv);  
                    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                    image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_depth" / (std::to_string(pkf->fid_)+"_"+std::to_string(i)+"_image.jpg"), image_cv);  
                    auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
                    cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
                    gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f, 0.f);
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_depth" / ("a_"+std::to_string(pkf->fid_)+"_gt.jpg"), gt_image_cv);  
                    auto diff_depth = torch::abs(rendered_depth - gt_depth) * this->rendered_depthmap_factor_ * valid_mask;
                    auto diff_depth_cv = tensor_utils::torchTensor2CvMat_Float32(diff_depth);
                    diff_depth_cv.convertTo(diff_depth_cv, CV_8UC1, 255.0f/2000.f, 0.f);
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_depth" / ("z_"+std::to_string(pkf->fid_)+"_"+std::to_string(i)+"_diff.jpg"), diff_depth_cv);  

                    if(idx == 0)
                        std::cout<<"[GaussianMapper::optimizeInsertedLocalLRDpts] rendered LR depth fid "<<pkf->fid_<<" loss after optimization: "<<batch_loss.item<float>()<<" iter "<<i<<std::endl;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }
            }
        }

    }

    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_optim_ply"))
        this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_optim_ply" / (std::to_string(valid_fids[0])+"_optim.ply"));
    }
}

float GaussianMapper::optimizeLocalHRImgs(std::vector<std::size_t>& random_kfids, std::vector<std::size_t>& valid_fids){
    // std::unique_lock<std::mutex> lock_render(mutex_render_);
    
    torch::Tensor loss;

    for(int i=0; i<valid_fids.size(); ++i){
        // auto& pkf = this->scene_->keyframes_.at(valid_fids[i]);
        auto pkf = this->scene_->getKeyframe(valid_fids[i]); // [debug 20260222]
        this->optimizeLocalHRPose(pkf, false);
    }

    int hr_height_resize, hr_width_resize;
    float hr_resize_ratio = this->getRsizedHRScale(1.f, hr_width_resize, hr_height_resize);
    std::unordered_map<std::size_t, torch::Tensor> opacity_masks;
    for(int i=0; i < random_kfids.size(); ++i){
        // auto& pkf = this->scene_->keyframes_.at(random_kfids[i]);
        auto pkf = this->scene_->getKeyframe(random_kfids[i]); // [debug 20260222]

        auto render_hr_pkg = GaussianRenderer::render(
            pkf,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, false, false, false
        );

        auto rendered_image = std::get<0>(render_hr_pkg);
        auto rendered_opacity = std::get<5>(render_hr_pkg);

        torch::Tensor opacity_mask;
        if(opacity_masks.find(pkf->fid_) == opacity_masks.end()){
            opacity_mask = (rendered_opacity > 0.01f).to(torch::kFloat32).squeeze();
            opacity_masks[pkf->fid_] = opacity_mask;
        }
        else opacity_mask = opacity_masks[pkf->fid_];
        
        auto gt_image = pkf->getGTHRImg(hr_resize_ratio, true);

        loss = loss_utils::get_loss_rgb(
            rendered_image, gt_image,
            this->lambdaDssim(),
            pkf->exposure_a_, pkf->exposure_b_,
            opacity_mask,
            device_type_
        );

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            // gaussians_->updateBatchGradients(valid_fids, 1.5f, 0.8f, true);
            
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);

            // pkf->stepOptimizer(false, true);
            pkf->zeroOptimizerGrad(true, true);

            if(kf_params_.debug_){
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color"))
                auto masked_rendered_image = rendered_image * opacity_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color" / (std::to_string(pkf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                auto masked_gt_image = gt_image * opacity_mask.unsqueeze(0);
                auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
                cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
                gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f, 0.f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_image_cv);
 
                std::cout<<"[GaussianMapper::optimizeLocalHRImgs] rendered HR img fid "<<pkf->fid_<<" loss after optimization: "<<loss.item<float>()<<" iter "<<i<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_); 
            }
        }

    }

    return loss.item<float>();
}

void GaussianMapper::insertBatchKeyframes(std::vector<std::size_t>& kfids){
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    this->local_mapping_batch_ids_[this->local_mapping_batch_ids_.size()] = kfids;
    // debug all batch ids
    // for (const auto& [batch_id, ids] : this->local_mapping_batch_ids_) {
    //     std::cout << "[GaussianMapper::insertBatchKeyframes] Batch ID: " << batch_id << ", Keyframe IDs: ";
    //     for (const auto& id : ids) {
    //         std::cout << id << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // throw std::runtime_error("[GaussianMapper::insertBatchKeyframes] debug stop after printing batch ids.");

    std::vector<torch::Tensor> valid_depth_masks;
    std::vector<std::size_t> valid_fids;

    for (int i = 0; i < kfids.size(); ++i){
        std::size_t kfid = kfids[i];
        this->handleKeyframeFrontend(kfid);

        // auto& new_kf = this->scene_->keyframes_.at(kfid);
        auto new_kf = this->scene_->getKeyframe(kfid); // [debug 20260222]
        new_kf->setGTHRImg(this->global_align_time_, this->vstrHRImagePaths_);
    }

    this->optimizeLocalLRPoses(kfids);

    for (int i = 0; i < kfids.size(); ++i){
        std::size_t kfid = kfids[i];
        auto new_kf = this->scene_->getKeyframe(kfid); // [debug 20260222]

        if(!new_kf->has_hr_fid_){
            std::cout<<"[GaussianMapper::insertBatchKeyframes] warning: no GT HR image for kf id "<<new_kf->fid_<<std::endl;
            continue;
        }
        else std::cout<<"[GaussianMapper::insertBatchKeyframes] set GT HR image for kf id "<<new_kf->fid_<<" hr fid "<<new_kf->hr_fid_<<std::endl;

        // this->optimizeLocalLRPose(new_kf, false); // if use ba, disable this // [test20260213] disable LR pnp

        valid_depth_masks.push_back(this->getLocalLRValidDptMsk(new_kf));
        valid_fids.push_back(kfid);

        if(valid_fids.size() >= this->local_align_batch_size_) break;
    }
    

    if(valid_fids.size() == 0){
        throw std::runtime_error("[GaussianMapper::insertBatchKeyframes] no valid keyframe to insert after checking GT HR images.");
    }

    int max_pixel_fid = this->insertLocalLRValidDpts(valid_depth_masks, valid_fids);

    // if (max_pixel_fid < 0) return;

    if(kf_params_.debug_){
        torch::NoGradGuard no_grad;
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd"))
        for(int i=0; i<valid_fids.size(); ++i){
            // auto& pkf = this->scene_->keyframes_.at(valid_fids[i]);
            auto pkf = this->scene_->getKeyframe(valid_fids[i]); // [debug 20260222]

            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.hr_height_/2, this->kf_params_.hr_width_/2,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, false, false, false
            );
            auto rendered_image = std::get<0>(render_pkg);
            auto gt_image = pkf->getGTHRImg(0.5f);
            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
            cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
            image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+".jpg"), image_cv); 
            auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
            cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
            gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+"_gt.jpg"), gt_image_cv);
        
            render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, false, false, false
            );
            auto rendered_image_lr = std::get<0>(render_pkg);
            auto gt_image_lr = pkf->getGTLRImg();
            auto image_cv_lr = tensor_utils::torchTensor2CvMat_Float32(rendered_image_lr);
            cv::cvtColor(image_cv_lr, image_cv_lr, CV_RGB2BGR);
            image_cv_lr.convertTo(image_cv_lr, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+"_lr.jpg"), image_cv_lr); 
            auto gt_image_cv_lr = tensor_utils::torchTensor2CvMat_Float32(gt_image_lr);
            cv::cvtColor(gt_image_cv_lr, gt_image_cv_lr, CV_RGB2BGR);
            gt_image_cv_lr.convertTo(gt_image_cv_lr, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+"_lr_gt.jpg"), gt_image_cv_lr);
        }

        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply"))
        if (max_pixel_fid >= 0)
            this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply" / (std::to_string(max_pixel_fid)+"_before.ply"));
        else 
            this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply" / (std::to_string(valid_fids[0])+"_invalid_before.ply"));
    }

    // return; // [test20260213] disable HR

    // throw std::runtime_error("[GaussianMapper::insertBatchKeyframes] debug stop after inserting valid depth points, check inserted points and metrics, then continue.");

    if (max_pixel_fid > 0) this->optimizeInsertedLocalLRDpts(valid_fids, valid_depth_masks);

    // return; // [test20260213] disable HR

    int random_iters = max_pixel_fid > 0 ? this->local_align_batch_size_ * this->local_align_batch_color_periter_ : this->local_align_batch_color_periter_;
    std::vector<std::size_t> batch_hr_color_fids;
    this->getBatchShuffledFrameIds(valid_fids, batch_hr_color_fids, random_iters);

    this->optimizeLocalHRImgs(batch_hr_color_fids, valid_fids);

    std::cout<<"[debug] finish check"<<std::endl;
    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply"))
        this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply" / (std::to_string(max_pixel_fid)+"_after.ply"));
    }

    // throw std::runtime_error("[GaussianMapper::insertBatchKeyframes] disable local HR optimization for now.");
}


void GaussianMapper::insertBatchKeyframes(
    std::vector<std::shared_ptr<KeyframeFrontend>>& kfs, 
    std::vector<double>& timestamps)
{
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    std::vector<torch::Tensor> valid_depth_masks;
    std::vector<std::size_t> valid_fids;
    for(int i = 0; i < kfs.size(); ++i){
        auto& kf = *kfs[i];
        std::size_t kfid = std::get<0>(kf);
    
        std::shared_ptr<GaussianKeyframe> new_kf = 
            std::make_shared<GaussianKeyframe>(kfid, getIteration(), &this->kf_params_);

        this->handleKeyframeFrontend(kf, new_kf, timestamps[i]);

        new_kf->setGTHRImg(this->global_align_time_, this->vstrHRImagePaths_);
        
        if(!new_kf->has_hr_fid_){
            std::cout<<"[GaussianMapper::insertBatchKeyframes] warning: no GT HR image for kf id "<<new_kf->fid_<<std::endl;
            continue;
        }
        else std::cout<<"[GaussianMapper::insertBatchKeyframes] set GT HR image for kf id "<<new_kf->fid_<<" hr fid "<<new_kf->hr_fid_<<std::endl;

        this->optimizeLocalLRPose(new_kf, false);

        valid_depth_masks.push_back(this->getLocalLRValidDptMsk(new_kf));
        valid_fids.push_back(kfid);
    }

    int max_pixel_fid = this->insertLocalLRValidDpts(valid_depth_masks, valid_fids);

    if(kf_params_.debug_){
        torch::NoGradGuard no_grad;
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd"))
        for(int i=0; i<valid_fids.size(); ++i){
            // auto& pkf = this->scene_->keyframes_.at(valid_fids[i]);
            auto pkf = this->scene_->getKeyframe(valid_fids[i]); // [debug 20260222]

            auto render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.hr_height_/2, this->kf_params_.hr_width_/2,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                true, false, false, false
            );
            auto rendered_image = std::get<0>(render_pkg);
            auto gt_image = pkf->getGTHRImg(0.5f);
            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
            cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
            image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+".jpg"), image_cv); 
            auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
            cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
            gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+"_gt.jpg"), gt_image_cv);
        
            render_pkg = GaussianRenderer::render(pkf,
                this->kf_params_.lr_height_, this->kf_params_.lr_width_,
                this->gaussians_, this->pipe_params_,
                this->background_, this->override_color_,
                false, false, false, false
            );
            auto rendered_image_lr = std::get<0>(render_pkg);
            auto gt_image_lr = pkf->getGTLRImg();
            auto image_cv_lr = tensor_utils::torchTensor2CvMat_Float32(rendered_image_lr);
            cv::cvtColor(image_cv_lr, image_cv_lr, CV_RGB2BGR);
            image_cv_lr.convertTo(image_cv_lr, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+"_lr.jpg"), image_cv_lr); 
            auto gt_image_cv_lr = tensor_utils::torchTensor2CvMat_Float32(gt_image_lr);
            cv::cvtColor(gt_image_cv_lr, gt_image_cv_lr, CV_RGB2BGR);
            gt_image_cv_lr.convertTo(gt_image_cv_lr, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lrhr_add_pcd" / (std::to_string(valid_fids[i])+"_lr_gt.jpg"), gt_image_cv_lr);
        }

        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply"))
        this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply" / (std::to_string(max_pixel_fid)+"_before.ply"));
    }

    this->optimizeInsertedLocalLRDpts(valid_fids, valid_depth_masks);

    throw std::runtime_error("[GaussianMapper::insertBatchKeyframes] disable local HR optimization for now.");

    std::vector<std::size_t> batch_hr_color_fids;
    this->getBatchShuffledFrameIds(valid_fids, batch_hr_color_fids, this->local_align_batch_size_ * this->local_align_batch_color_periter_);

    this->optimizeLocalHRImgs(batch_hr_color_fids, valid_fids);

    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply"))
        this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply" / (std::to_string(max_pixel_fid)+"_after.ply"));
    }

}


void GaussianMapper::insertLocalLRValidDpt(std::shared_ptr<GaussianKeyframe> pkf){
    auto valid_depth_mask = this->getLocalLRValidDptMsk(pkf);
    auto valid_depth = valid_depth_mask * pkf->getGTLRDpt(true);
    auto gt_image = pkf->getGTLRImg(true);
    this->gaussians_->increaseKeyframeInitPcd(pkf, valid_depth, gt_image, this->kf_params_);
}

float GaussianMapper::optimizeLocalHRImg(std::shared_ptr<GaussianKeyframe> pkf){
    torch::Tensor loss;

    int hr_height_resize, hr_width_resize;
    float hr_resize_ratio = this->getRsizedHRScale(1.f, hr_width_resize, hr_height_resize);

    for(int iter = 0; iter < this->local_align_hr_color_iter_; ++iter){
        auto render_hr_pkg = GaussianRenderer::render(
            pkf,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, false, false, false
        );

        auto rendered_image = std::get<0>(render_hr_pkg);
        auto rendered_opacity = std::get<5>(render_hr_pkg);

        auto opacity_mask = (rendered_opacity > 0.01f).to(torch::kFloat32).squeeze();
        
        auto gt_image = pkf->getGTHRImg(hr_resize_ratio, true);

        loss = loss_utils::get_loss_rgb(
            rendered_image, gt_image,
            this->lambdaDssim(),
            pkf->exposure_a_, pkf->exposure_b_,
            opacity_mask,
            device_type_
        );

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            gaussians_->updateBatchGradients({pkf->fid_}, 1.5f, 0.8f, false, false, false);
            
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);

            // pkf->stepOptimizer(false, true);
            pkf->zeroOptimizerGrad(true, true);

            if(kf_params_.debug_){
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color"))
                auto masked_rendered_image = rendered_image * opacity_mask.unsqueeze(0);
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f, 0.f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color" / (std::to_string(pkf->fid_)+"-"+std::to_string(iter)+".jpg"), image_cv);
                auto masked_gt_image = gt_image * opacity_mask.unsqueeze(0);
                auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
                cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
                gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f, 0.f);
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color" / (std::to_string(pkf->fid_)+"_gt.jpg"), gt_image_cv);
                std::cout<<"[GaussianMapper::optimizeLocalHRImg] rendered HR img fid "<<pkf->fid_<<" loss after optimization: "<<loss.item<float>()<<std::endl;
                metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
            }
        }
    }

    return loss.item<float>();
}

void GaussianMapper::insertOneKeyframe(
    KeyframeFrontend& kf,
    double timestamp
){
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    
    std::size_t kfid = std::get<0>(kf);
    std::shared_ptr<GaussianKeyframe> new_kf = 
        std::make_shared<GaussianKeyframe>(kfid, getIteration(), &this->kf_params_);

    this->handleKeyframeFrontend(kf, new_kf, timestamp);

    new_kf->setGTHRImg(this->global_align_time_, this->vstrHRImagePaths_);
        
    if(!new_kf->has_hr_fid_){
        std::cout<<"[GaussianMapper::insertBatchKeyframes] warning: no GT HR image for kf id "<<new_kf->fid_<<std::endl;
        return;
    }
    else std::cout<<"[GaussianMapper::insertBatchKeyframes] set GT HR image for kf id "<<new_kf->fid_<<" hr fid "<<new_kf->hr_fid_<<std::endl;

    this->optimizeLocalLRPose(new_kf, false);

    new_kf->setGlobalDeltaPose(this->global_align_pose_);

    this->optimizeLocalHRPose(new_kf, false);

    this->insertLocalLRValidDpt(new_kf);

    this->optimizeLocalHRImg(new_kf);

    if(kf_params_.debug_){
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply"))
        this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_add_ply" / (std::to_string(new_kf->fid_)+"_after.ply"));
    }

}


void GaussianMapper::insertOneKeyframe_old(
    // std::tuple<unsigned long,    // pKF->mnId,
    // unsigned long,              // pKF->mpCamera->GetId(),
    // Sophus::SE3f,               // pKF->GetPose(),
    // cv::Mat,                    // pKF->imgLeftRGB.clone(),
    // bool,                       // isLoopClosureKF,
    // cv::Mat,                    // pKF->imgAuxiliary,
    // std::vector<float>,         // pixels,
    // std::vector<float>,         // pointsLocal,
    // std::string> &kf,           // pKF->mNameFile
    KeyframeFrontend& kf,
    double timestamp
){
    // step 0. create new keyframe
    std::cout<<"[GaussianMapper::insertOneKeyframe] insert keyframe id "<<std::get<0>(kf)<<std::endl;
    std::cout<<"[GaussianMapper::insertOneKeyframe] keyframe path "<<std::get<8>(kf)<<std::endl;
    std::size_t kfid = std::get<0>(kf);
    
    std::shared_ptr<GaussianKeyframe> new_kf = 
        std::make_shared<GaussianKeyframe>(kfid, getIteration(), &this->kf_params_);

    {
        auto camera_id = std::get<1>(kf);
        auto& pose = std::get<2>(kf);
        auto& img = std::get<3>(kf);
        auto& dpt = std::get<5>(kf);
        new_kf->img_filename_ = std::get<8>(kf);
        new_kf->lr_timestamp_ = float(timestamp);

        // undistort and set camera params
        Camera& camera = scene_->cameras_.at(camera_id);
        new_kf->setCameraParams(camera);

        cv::Mat img_undistorted, dpt_undistorted;
        camera.undistortImage(img, img_undistorted);
        camera.undistortImage(dpt, dpt_undistorted);
        new_kf->img_undist_ = img_undistorted;
        new_kf->setGTLRDpt(dpt_undistorted);
        new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(img_undistorted, device_type_);
        
        this->generatePyramidSizes(new_kf, camera);
        increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

        // set pose
        new_kf->setPose(
            pose.unit_quaternion().cast<double>(),
            pose.translation().cast<double>());
        // new_kf->computeTransformTensors();

        if(kf_params_.debug_){
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist"))
            cv::Mat imgRGB_undistorted_vis, imgRGB_vis;
            img_undistorted.convertTo(imgRGB_undistorted_vis, CV_8UC3, 255.0f, 0.f);
            cv::cvtColor(imgRGB_undistorted_vis, imgRGB_undistorted_vis, CV_RGB2BGR);
            img.convertTo(imgRGB_vis, CV_8UC3, 255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB_undistorted.jpg"), imgRGB_undistorted_vis);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgRGB.jpg"), imgRGB_vis);

            cv::Mat imgAux_undistorted_vis, imgAux_vis;
            dpt_undistorted.convertTo(imgAux_undistorted_vis, CV_8UC1, 4000.0f/6000.0f*255.0f, 0.f);
            dpt.convertTo(imgAux_vis, CV_8UC1, 4000.0f/6000.0f*255.0f, 0.f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux_undistorted.jpg"), imgAux_undistorted_vis);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size()) + kf_params_.debug_dir_) / "lr_undist" / (std::to_string(new_kf->fid_) + "_imgAux.jpg"), imgAux_vis);
        }
    }

    // resize hr size for fast optimization
    new_kf->setGTHRImg(this->global_align_time_, this->vstrHRImagePaths_);
    std::cout<<"[GaussianMapper::insertOneKeyframe] set GT HR image for kf id "<<new_kf->fid_<<" hr fid "<<new_kf->hr_fid_<<std::endl;
    
    int hr_height_resize, hr_width_resize;
    float hr_resize_ratio;
    if(this->local_align_hr_resize_ratio_>=1.f-1e-5f){
        hr_height_resize = this->kf_params_.hr_height_;
        hr_width_resize = this->kf_params_.hr_width_;
        hr_resize_ratio = 1.f;
    }
    else{
        hr_height_resize = int(floor(float(this->kf_params_.hr_height_) * this->local_align_hr_resize_ratio_));
        hr_width_resize = int(floor(float(this->kf_params_.hr_width_) * this->local_align_hr_resize_ratio_));
        hr_resize_ratio = this->local_align_hr_resize_ratio_;
    }

    // add to scene
    scene_->addKeyframe(new_kf, &kfid_shuffled_);

    std::unique_lock<std::mutex> lock_render(mutex_render_);
    
    // step 1. optimize lr init pose
    std::cout<<"[GaussianMapper::insertOneKeyframe] step1: optimize lr init pose."<<std::endl;
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    torch::Tensor gt_image;
    torch::Tensor raw_base_pose = new_kf->getBasePose().clone();
    for(int i=0; i<this->local_align_lr_pose_iter_; ++i){
        render_pkg = GaussianRenderer::render(
            new_kf,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, true, true, true
        );

        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<4>(render_pkg);

        gt_image = new_kf->getGTLRImg().cuda();
        auto gt_depth = new_kf->getGTLRDpt().cuda();
        auto gt_depth_mask = new_kf->getGTLRDptMsk().cuda(); 

        auto loss = loss_utils::get_loss_rgbd(
            rendered_image, gt_image,
            rendered_depth, gt_depth,
            this->lambdaDssim(),
            this->global_align_lr_depth_lambda_,
            new_kf->exposure_a_, new_kf->exposure_b_,
            gt_depth_mask,
            device_type_
        );

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            gaussians_->optimizer_->zero_grad(true); 
            new_kf->stepOptimizer(true);
            new_kf->zeroOptimizerGrad(true, true);

            try{
                // std::cout<<"[debug] fid "<<new_kf->fid_<<" iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
                new_kf->updateBasePose();
                // std::cout<<"[debug] updateBasePose "<<std::endl;
            }
            catch(std::exception& e){
                std::cout<<"[error] theta "<<new_kf->theta_<<" rho "<<new_kf->rho_<<std::endl;
                std::cout<<"[error] fid "<<new_kf->fid_<<" iter "<<i<<std::endl;
                std::cout<<e.what()<<std::endl;
            }

            if(i % 10 == 0)
                new_kf->updateOptimizer(0.5f);
                
            
            if(kf_params_.debug_){
                // auto masked_gt_image = gt_image * gt_depth_mask;
                // auto masked_gt_depth = gt_depth * gt_depth_mask;
                // auto masked_rendered_image = rendered_image * gt_depth_mask;
                // auto masked_rendered_depth = rendered_depth * gt_depth_mask;

                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(new_kf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
            
                if(i==0){
                    auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
                    cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                    gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_train_pose" / (std::to_string(new_kf->fid_)+"_gt.jpg"), gt_cv);
                }
            }
        }
    }

    /*
    // step 2. get depth holes
    std::cout<<"[GaussianMapper::insertOneKeyframe] step2: get depth holes."<<std::endl;
    torch::Tensor valid_depth;
    {
        torch::NoGradGuard no_grad;

        // render_pkg = GaussianRenderer::render(
        //     new_kf,
        //     this->kf_params_.lr_height_, this->kf_params_.lr_width_,
        //     this->gaussians_, this->pipe_params_,
        //     this->background_, this->override_color_,
        //     false, true, true, true
        // );
        auto rendered_depth = std::get<4>(render_pkg);
        auto rendered_opacity = std::get<5>(render_pkg);
        auto opacity_mask = (rendered_opacity < this->local_align_lr_opcacity_thr_).to(torch::kFloat32);
        std::cout<<"[debug] opacity mask sum "<<opacity_mask.sum().item<float>()<<std::endl;

        auto gt_depth = new_kf->getGTLRDpt().cuda();
        auto gt_depth_mask = new_kf->getGTLRDptMsk().cuda();

        valid_depth = gt_depth * gt_depth_mask * opacity_mask;

        if(kf_params_.debug_){
            auto valid_depth_cpu = valid_depth.squeeze().cpu();
            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(valid_depth_cpu);
            image_cv.convertTo(image_cv, CV_8UC1, 4000.0f/6000.0f*255.0f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+".jpg"), image_cv);
            auto opacity_mask_cpu = opacity_mask.squeeze().cpu();
            auto image_cv2 = tensor_utils::torchTensor2CvMat_Float32(opacity_mask_cpu);
            image_cv2.convertTo(image_cv2, CV_8UC1, 255.0f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_mask.jpg"), image_cv2);
            auto opacity_cpu = rendered_opacity.squeeze().cpu();
            auto image_cv3 = tensor_utils::torchTensor2CvMat_Float32(opacity_cpu);
            image_cv3.convertTo(image_cv3, CV_8UC1, 255.0f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_opacity.jpg"), image_cv3);
        }
    }

    // step 3. insert new gaussians from depth holes
    std::cout<<"[GaussianMapper::insertOneKeyframe] step3: insert new gaussians from depth holes."<<std::endl;
    this->gaussians_->increaseKeyframeInitPcd(new_kf, valid_depth, gt_image, this->kf_params_);
    */

    /*
    // step 4. optimize lr pose with new gs
    std::cout<<"[GaussianMapper::insertOneKeyframe] step4: optimize lr pose with new gaussians."<<std::endl;
    // new_kf->setDenseInitJointOptimization(true);
    // new_kf->updateOptimizer(0.5f);
    for(int i=0; i<this->local_align_lr_joint_pose_iter_; ++i){
        render_pkg = GaussianRenderer::render(
            new_kf,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, false, false, true
        );

        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<4>(render_pkg);
        auto rednered_opacity = std::get<5>(render_pkg);

        gt_image = new_kf->getGTLRImg(true);
        auto gt_depth = new_kf->getGTLRDpt(true);
        auto gt_depth_mask = new_kf->getGTLRDptMsk(true); 

        auto loss = loss_utils::get_loss_rgbd(
            rendered_image, gt_image,
            rendered_depth, gt_depth,
            this->lambdaDssim(),
            this->global_align_lr_depth_lambda_,
            new_kf->exposure_a_, new_kf->exposure_b_,
            gt_depth_mask,
            device_type_
        );

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true); 
            new_kf->stepOptimizer(true);
            new_kf->zeroOptimizerGrad(true, true);

            try{
                // std::cout<<"[debug] fid "<<new_kf->fid_<<" iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
                std::cout<<"[debug] fid "<<new_kf->fid_<<" iter "<<i<<" theta norm "<<new_kf->theta_.cpu().norm().item<float>()<<" rho norm "<<new_kf->rho_.norm().item<float>()<<std::endl;
                new_kf->updateBasePose();
                // std::cout<<"[debug] updateBasePose "<<std::endl;
                new_kf->updateDenseInitJointGaussians(this->gaussians_->xyz_, this->gaussians_->exist_since_iter_);
            }
            catch(std::exception& e){
                std::cout<<"[error] theta "<<new_kf->theta_<<" rho "<<new_kf->rho_<<std::endl;
                std::cout<<"[error] fid "<<new_kf->fid_<<" iter "<<i<<std::endl;
                std::cout<<e.what()<<std::endl;
            }

            if(kf_params_.debug_){
                auto masked_rendered_image = (rendered_image * gt_depth_mask).cpu();
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose" / (std::to_string(new_kf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                
                auto masked_rendered_depth = rendered_depth.cpu().squeeze();
                auto depth_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_depth);
                depth_cv.convertTo(depth_cv, CV_8UC1, 4000.0f/6000.0f*255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_depth"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_depth" / (std::to_string(new_kf->fid_)+"-"+std::to_string(i)+"_depth.jpg"), depth_cv);

                auto opacity_cv = rednered_opacity.squeeze().cpu();
                auto opacity_image_cv = tensor_utils::torchTensor2CvMat_Float32(opacity_cv);
                opacity_image_cv.convertTo(opacity_image_cv, CV_8UC1, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_depth"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_depth" / (std::to_string(new_kf->fid_)+"-"+std::to_string(i)+"_opacity.jpg"), opacity_image_cv);
            }
        }
    }
    */

    // step 5. optimize hr pose
    std::cout<<"[GaussianMapper::insertOneKeyframe] step5: optimize hr pose."<<std::endl;
    // this->gaussians_->increaseKeyframeFinalPcd(new_kf);
    // new_kf->setDenseInitJointOptimization(false);
    new_kf->setGlobalDeltaPose(this->global_align_pose_);

    /*
    std::sort(scene_->keyframes_ids_.begin(), scene_->keyframes_ids_.end());
    std::vector<torch::Tensor> hr_pred_poses = {
        scene_->keyframes_.at(scene_->keyframes_ids_.at(scene_->keyframes_ids_.size()-3))->getBasePose(),
        scene_->keyframes_.at(scene_->keyframes_ids_.at(scene_->keyframes_ids_.size()-2))->getBasePose(),
        scene_->keyframes_.at(scene_->keyframes_ids_.at(scene_->keyframes_ids_.size()-1))->getBasePose(),
    };
    std::vector<float> hr_pred_times = {
        scene_->keyframes_.at(scene_->keyframes_ids_.at(scene_->keyframes_ids_.size()-3))->lr_timestamp_,
        scene_->keyframes_.at(scene_->keyframes_ids_.at(scene_->keyframes_ids_.size()-2))->lr_timestamp_,
        scene_->keyframes_.at(scene_->keyframes_ids_.at(scene_->keyframes_ids_.size()-1))->lr_timestamp_,
    };
    float hr_pred_time = new_kf->hr_fid_ / kf_params_.hr_fps_ - this->global_align_time_;
    auto hr_pred_delta_pose = tensor_utils::predict_pose_cubic_hermite(hr_pred_poses, hr_pred_times, hr_pred_time);
    std::cout<<"[debug] hr pred delta pose \n"<<hr_pred_delta_pose<<std::endl;
    // std::cout<<"[debug] hr pred time "<<hr_pred_time<<" hr pred pose \n"<<hr_pred_pose<<std::endl;
    // std::cout<<"[debug] lr time "<<new_kf->lr_timestamp_<<" lr base pose \n"<<new_kf->getBasePose()<<std::endl;
    // std::cout<<"[debug] lr time -1 "<<hr_pred_times[1]<<" lr base pose \n"<<hr_pred_poses[1]<<std::endl;
    // std::cout<<"[debug] lr time 0 "<<hr_pred_times[2]<<" lr base pose \n"<<hr_pred_poses[2]<<std::endl;
    
    new_kf->setLocalDeltaPoseInit(hr_pred_delta_pose);
    */

    new_kf->resetOptimizer(true, kf_params_.theta_lr_, kf_params_.rho_lr_);
    // torch::Tensor opacity0;
    torch::Tensor opacity_mask;
    for(int i=0; i<this->local_align_hr_pose_iter_; i++){
        auto render_hr_pkg = GaussianRenderer::render(
            new_kf,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, true, true, true
        );

        auto rendered_image = std::get<0>(render_hr_pkg);
        auto rendered_depth = std::get<4>(render_hr_pkg);
        auto rendered_opacity = std::get<5>(render_hr_pkg);

        if(i == 0)
            opacity_mask = (rendered_opacity > 0.1f).to(torch::kFloat32);

        torch::Tensor linear_sampling_map;
        gt_image = new_kf->getGTHRImg(linear_sampling_map, hr_resize_ratio, true);
        // gt_image = new_kf->getGTHRImg(hr_resize_ratio, true);

        // std::cout<<"[debug] rendered_image size "<<rendered_image.sizes()<<std::endl;
        // std::cout<<"[debug] gt_image size "<<gt_image.sizes()<<std::endl;
        // std::cout<<"[debug] opacity_mask sum "<<opacity_mask.sum().item<float>()<<std::endl;

        // auto loss_rgb = loss_utils::get_loss_rgb(
        //     linear_sampling_map,
        //     rendered_image, gt_image,
        //     0.1f,
        //     new_kf->exposure_a_, new_kf->exposure_b_,
        //     opacity_mask,
        //     device_type_
        // );
        auto loss_rgb = loss_utils::get_loss_rgb(linear_sampling_map,
            rendered_image, gt_image,
            lambdaDssim(),
            new_kf->exposure_a_, new_kf->exposure_b_,
            opacity_mask
        );

        // auto pose_reg = loss_utils::get_loss_posereg(
        //     new_kf->getLocalDeltaPose(), 0.4f
        // );

        auto loss_depth = loss_utils::get_loss_hr2lr(
            linear_sampling_map, rendered_depth.squeeze(), opacity_mask.squeeze(),
            new_kf->getGTLRDpt(true), new_kf->getGTLRDptMsk(true),
            kf_params_.hr_fx_*hr_resize_ratio, kf_params_.hr_fy_*hr_resize_ratio, 
            kf_params_.hr_cx_*hr_resize_ratio, kf_params_.hr_cy_*hr_resize_ratio,
            kf_params_.lr_fx_, kf_params_.lr_fy_, 
            kf_params_.lr_cx_, kf_params_.lr_cy_,
            new_kf->getFullDeltaPose()
        );

        // auto loss_opc = i == 0 ? torch::zeros({1}, rendered_opacity.options()) 
        //     : loss_utils::get_loss_opacity_delta(opacity0, rendered_opacity);

        // // auto loss = loss_rgb + pose_reg * this->local_align_hr_pose_reg_lambda_ + loss_opc * this->local_align_hr_pose_opacity_lambda_;
        // auto loss = loss_rgb * (1.f - this->local_align_hr_pose_depth_lambda_) + 
        //     loss_depth * this->local_align_hr_pose_depth_lambda_ * 1.f + 
        //     loss_opc * this->local_align_hr_pose_opacity_lambda_ +
        //     pose_reg * this->local_align_hr_pose_reg_lambda_;

        auto loss = loss_rgb * (1.f-this->local_align_hr_pose_depth_lambda_) + 
            loss_depth * this->local_align_hr_pose_depth_lambda_;

        // auto loss = loss_rgb;

        // std::cout<<"[debug] fid "<<new_kf->fid_<<" hr iter "<<i<<" loss "<<loss.item<float>()<<" rgb loss "<<loss_rgb.item<float>()<<" pose reg "<<pose_reg.item<float>()<<" opacity loss "<<loss_opc.item<float>()<<std::endl;
        // std::cout<<"[debug] fid "<<new_kf->fid_<<" hr iter "<<i<<" loss "<<loss.item<float>()<<" rgb loss "<<loss_rgb.item<float>()<<" pose reg "<<pose_reg.item<float>()<<" depth loss "<<loss_depth.item<float>()<<" opacity loss "<<loss_opc.item<float>()<<std::endl;

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            // gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);

            // if(i < this->local_align_hr_pose_iter_/2) new_kf->stepOptimizer(true, false);
            // else new_kf->stepOptimizer(true, true);
            new_kf->stepOptimizer(true, false);
            new_kf->zeroOptimizerGrad(true, true);
            new_kf->updateLocalDeltaPose(true);

            // if(i == this->local_align_hr_pose_iter_/2)
            //     new_kf->updateOptimizer(0.6f);

            // if(i == 0)
            //     opacity0 = rendered_opacity.detach().clone();

            if(kf_params_.debug_){
                std::cout<<"[debug] fid "<<new_kf->fid_<<" hr iter "<<i<<" loss "<<loss.item<float>()<<" rgb loss "<<loss_rgb.item<float>()<<" depth loss "<<loss_depth.item<float>()<<std::endl;
                // std::cout<<"[debug] fid "<<new_kf->fid_<<" hr iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
                auto rendered_image_ab = rendered_image * torch::exp(new_kf->exposure_a_) + new_kf->exposure_b_;
                auto masked_rendered_image = rendered_image_ab * opacity_mask;
                auto masked_gt_image = gt_image * opacity_mask;
                // auto masked_rendered_image_ab = masked_rendered_image * torch::exp(new_kf->exposure_a_) + new_kf->exposure_b_;

                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(new_kf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
            
                if((i+1) % 5 == 0){
                    std::cout<<"[debug] metrics at hr pose iter "<<i<<std::endl;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }

                if(i == 0){
                    std::cout<<"[debug] metrics at hr pose iter "<<i<<std::endl;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                    auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(masked_gt_image);
                    cv::cvtColor(gt_cv, gt_cv, CV_RGB2BGR);
                    gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
                    cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_pose" / (std::to_string(new_kf->fid_)+"_gt.jpg"), gt_cv);
                }
                // if(i == this->local_align_hr_pose_iter_-1){
                //     std::cout<<"[debug] metrics at hr pose final iter "<<i<<std::endl;
                //     metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                // }
            }
        }
    }

    // step 2. get depth holes
    std::cout<<"[GaussianMapper::insertOneKeyframe] step2: get depth holes."<<std::endl;
    torch::Tensor valid_depth;
    {
        torch::NoGradGuard no_grad;

        // auto base_pose_temp = new_kf->getBasePose().clone();
        auto local_pose_temp = new_kf->getLocalDeltaPose().clone();
        auto eye_pose = torch::eye(4).to(device_type_);
        // new_kf->setBasePose(raw_base_pose);
        new_kf->setGlobalDeltaPose(eye_pose);
        new_kf->setLocalDeltaPose(eye_pose);

        render_pkg = GaussianRenderer::render(
            new_kf,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, true, true, true
        );                                                                                                  

        auto rendered_depth = std::get<4>(render_pkg);
        auto rendered_opacity = std::get<5>(render_pkg);
        auto opacity_mask_depth = (rendered_opacity < this->local_align_lr_opcacity_thr_).to(torch::kFloat32);
        std::cout<<"[debug] opacity mask sum "<<opacity_mask_depth.sum().item<float>()<<std::endl;

        auto gt_depth = new_kf->getGTLRDpt().cuda();
        auto gt_depth_mask = new_kf->getGTLRDptMsk().cuda();

        gt_image = new_kf->getGTLRImg().cuda();

        valid_depth = gt_depth * gt_depth_mask * opacity_mask_depth.squeeze();

        // new_kf->setBasePose(base_pose_temp);
        new_kf->setGlobalDeltaPose(this->global_align_pose_);
        new_kf->setLocalDeltaPose(local_pose_temp);

        if(kf_params_.debug_){
            auto valid_depth_cpu = valid_depth.squeeze().cpu();
            auto image_cv = tensor_utils::torchTensor2CvMat_Float32(valid_depth_cpu);
            image_cv.convertTo(image_cv, CV_8UC1, 4000.0f/6000.0f*255.0f);
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes"))
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+".jpg"), image_cv);
            auto opacity_mask_depth_cpu = opacity_mask_depth.squeeze().cpu();
            auto image_cv2 = tensor_utils::torchTensor2CvMat_Float32(opacity_mask_depth_cpu);
            image_cv2.convertTo(image_cv2, CV_8UC1, 255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_mask.jpg"), image_cv2);
            auto opacity_cpu = rendered_opacity.squeeze().cpu();
            auto image_cv3 = tensor_utils::torchTensor2CvMat_Float32(opacity_cpu);
            image_cv3.convertTo(image_cv3, CV_8UC1, 255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_opacity.jpg"), image_cv3);
            auto rendered_depth_cpu = rendered_depth.squeeze().cpu();
            auto image_cv4 = tensor_utils::torchTensor2CvMat_Float32(rendered_depth_cpu);
            image_cv4.convertTo(image_cv4, CV_8UC1, 4000.0f/6000.0f*255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_depth.jpg"), image_cv4);
            auto valid_gt_image = gt_image * gt_depth_mask.unsqueeze(0) * opacity_mask_depth;
            auto gt_image_cpu = valid_gt_image.cpu();
            auto image_cv5 = tensor_utils::torchTensor2CvMat_Float32(gt_image_cpu);
            cv::cvtColor(image_cv5, image_cv5, CV_RGB2BGR);
            image_cv5.convertTo(image_cv5, CV_8UC3, 255.0f);
            cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_rgb.jpg"), image_cv5);
            // std::cout<<"[debug] gt_depth_mask size "<<gt_depth_mask.sizes()<<std::endl;
            // std::cout<<"[debug] opacity_mask_depth size "<<opacity_mask_depth.sizes()<<std::endl;
            
        }
    }

    // step 3. insert new gaussians from depth holes
    std::cout<<"[GaussianMapper::insertOneKeyframe] step3: insert new gaussians from depth holes."<<std::endl;
    this->gaussians_->increaseKeyframeInitPcd(new_kf, valid_depth, gt_image, this->kf_params_);

    if(this->kf_params_.debug_){
        torch::NoGradGuard no_grad;

        auto eye_pose = torch::eye(4).to(device_type_);
        new_kf->setGlobalDeltaPose(eye_pose);

        auto local_pose_temp = new_kf->getLocalDeltaPose().clone();
        new_kf->setLocalDeltaPose(eye_pose);

        auto render_debug_pkg = GaussianRenderer::render(
            new_kf,
            this->kf_params_.lr_height_, this->kf_params_.lr_width_,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            false, true, true, true
        );

        auto rendered_image = std::get<0>(render_debug_pkg);
        // auto rendered_depth = std::get<4>(render_debug_pkg);
        auto rendered_opacity = std::get<5>(render_debug_pkg);

        gt_image = new_kf->getGTLRImg().cuda();

        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_lr_rgb_after.jpg"), image_cv);
        auto opacity_cpu = rendered_opacity.squeeze().cpu();
        auto opacity_image_cv = tensor_utils::torchTensor2CvMat_Float32(opacity_cpu);
        opacity_image_cv.convertTo(opacity_image_cv, CV_8UC1, 255.0f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_lr_opacity_after.jpg"), opacity_image_cv);
        auto gt_image_cpu = gt_image.cpu();
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image_cpu);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_depth_holes" / (std::to_string(new_kf->fid_)+"_lr_gt.jpg"), gt_image_cv);
    
        new_kf->setGlobalDeltaPose(this->global_align_pose_);
        new_kf->setLocalDeltaPose(local_pose_temp);
    }

    // step 6. optimize hr color
    std::cout<<"[GaussianMapper::insertOneKeyframe] step6: optimize hr color."<<std::endl;
    new_kf->updateOptimizer(0.8f);
    for(int i=0; i<this->local_align_hr_color_iter_; i++){
        auto render_hr_pkg = GaussianRenderer::render(
            new_kf,
            hr_height_resize, hr_width_resize,
            this->gaussians_, this->pipe_params_,
            this->background_, this->override_color_,
            true, true, false, false // !!! fix gs xyz, only color the gs
        );

        auto rendered_image = std::get<0>(render_hr_pkg);
        auto rendered_depth = std::get<4>(render_hr_pkg);
        auto rendered_opacity = std::get<5>(render_hr_pkg);

        if(i == 0)
            opacity_mask = (rendered_opacity > 0.1f).to(torch::kFloat32);

        torch::Tensor linear_sampling_map;
        gt_image = new_kf->getGTHRImg(linear_sampling_map, hr_resize_ratio, true);

        auto loss_rgb = loss_utils::get_loss_rgb(
            linear_sampling_map,
            rendered_image, gt_image,
            lambdaDssim(),
            new_kf->exposure_a_, new_kf->exposure_b_,
            opacity_mask
        );

        auto loss_depth = loss_utils::get_loss_hr2lr(
            linear_sampling_map, rendered_depth.squeeze(), opacity_mask.squeeze(),
            new_kf->getGTLRDpt(true), new_kf->getGTLRDptMsk(true),
            kf_params_.hr_fx_*hr_resize_ratio, kf_params_.hr_fy_*hr_resize_ratio, 
            kf_params_.hr_cx_*hr_resize_ratio, kf_params_.hr_cy_*hr_resize_ratio,
            kf_params_.lr_fx_, kf_params_.lr_fy_, 
            kf_params_.lr_cx_, kf_params_.lr_cy_,
            new_kf->getFullDeltaPose()
        ); 

        auto loss = loss_rgb * (1.f - this->local_align_hr_pose_depth_lambda_) + 
            loss_depth * this->local_align_hr_pose_depth_lambda_;

        // if(loss.item<float>() < this->local_align_hr_color_loss_thr_){
        //     std::cout<<"[GaussianMapper::insertOneKeyframe] fid "<<new_kf->fid_<<" early stop hr color optimization at iter "<<i<<" loss "<<loss.item<float>()<<std::endl;
        //     break;
        // }

        loss.backward();

        {
            torch::NoGradGuard no_grad;

            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true); 

            new_kf->stepOptimizer(true, false);
            new_kf->zeroOptimizerGrad(true, true);

            new_kf->updateLocalDeltaPose(true);

            if(kf_params_.debug_){
                std::cout<<"[debug] fid "<<new_kf->fid_<<" hr color iter "<<i<<" loss "<<loss.item<float>()<<" rgb loss "<<loss_rgb.item<float>()<<" depth loss "<<loss_depth.item<float>()<<std::endl;
                auto rendered_image_ab = rendered_image * torch::exp(new_kf->exposure_a_) + new_kf->exposure_b_;
                auto masked_rendered_image = rendered_image_ab * opacity_mask;
                auto masked_gt_image = gt_image * opacity_mask;
                auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_rendered_image);
                cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
                image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
                CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color"))
                cv::imwrite(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "hr_train_color" / (std::to_string(new_kf->fid_)+"-"+std::to_string(i)+".jpg"), image_cv);
                if((i+1) % 5 == 0 || i == 0){
                    std::cout<<"[debug] metrics at hr color iter "<<i<<std::endl;
                    metrics_utils::report_metrics(masked_rendered_image, masked_gt_image, this->lpips_model_);
                }
            }
        }
    }

    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS((result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_ply"))
    // this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_ply" / ("kf_"+std::to_string(new_kf->fid_)+".ply"), new_kf, true);
    // this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_ply" / ("combined_kf_"+std::to_string(new_kf->fid_)+".ply"), new_kf, false);
    this->gaussians_->savePly(result_dir_ / (std::to_string(getIteration()) + "-" + std::to_string(this->local_mapping_operations_.size())  + kf_params_.debug_dir_) / "lr_joint_train_pose_ply" / ("original.ply"));
    // throw std::runtime_error("[GaussianMapper::insertOneKeyframe] debug throw!");

}


/*
void GaussianMapper::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}
*/

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam || !viewpoint_cam->has_hr_fid_) {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
            image_height = kf_params_.hr_height_;
            image_width = kf_params_.hr_width_;
            gt_image = viewpoint_cam->getGTHRImg().cuda();
            mask = kf_params_.lr_undistort_mask_tensor_;
        }
        else{
            image_height = viewpoint_cam->image_height_;
            image_width = viewpoint_cam->image_width_;
            gt_image = viewpoint_cam->original_image_.cuda();
            mask = undistort_mask_[viewpoint_cam->camera_id_];
        }
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_)
            mask = kf_params_.gaus_pyramid_hr_undistort_mask_[training_level];
        else mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    std::cout<<"[debug] train level "<<training_level<<" image_height "<<image_height<<" image_width "<<image_width<<std::endl;   

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_ ,
        true, true
    );
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;
    // gt_image = viewpoint_cam->hr_image_undist_.cuda();
    // torch::Tensor masked_image = rendered_image;

    // Loss
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    loss.backward();

    torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        // if (keyframe_record_interval_ &&
        //     getIteration() % keyframe_record_interval_ == 0)
        //     recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_, result_dir_, result_dir_);

        // Densification
        // if (getIteration() < opt_params_.densify_until_iter_) {
        //     // Keep track of max radii in image-space for pruning
        //     gaussians_->max_radii2D_.index_put_(
        //         {visibility_filter},
        //         torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
        //                     radii.index({visibility_filter})));
        //     // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
        //         gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

        //     if ((getIteration() > opt_params_.densify_from_iter_) &&
        //         (getIteration() % densifyInterval()== 0)) {
        //         int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
        //         gaussians_->densifyAndPrune(
        //             densifyGradThreshold(),
        //             densify_min_opacity_,//0.005,//
        //             scene_->cameras_extent_,
        //             size_threshold
        //         );
        //     }

        //     if (opacityResetInterval()
        //         && (getIteration() % opacityResetInterval() == 0
        //             ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
        //         gaussians_->resetOpacity();
        // }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);

            // viewpoint_cam->optimizer_->step(); // !!! need to change zeroOptimizerGrad
            // viewpoint_cam->optimizer_->zero_grad(true);

            viewpoint_cam->updateLocalDeltaPose();
        }
    }
}

void GaussianMapper::waitUntilFinished() {
    std::cout << "[GaussianMapper::waitUntilFinished] Waiting for mapper to process all frames..." << std::endl;

    while (!isStopped()) {
        std::vector<std::size_t> tracking_ids, mapping_ids;

        if(this->pCuVSLAM_->IsTrackingFinished()) {
            this->pCuVSLAM_->GetFrameIds(tracking_ids);
            this->scene_->getKeyframeIds(mapping_ids);

            std::cout << "[GaussianMapper::waitUntilFinished] Tracking frames: " << tracking_ids.size() 
                  << ", Mapped frames: " << mapping_ids.size() << std::endl;

            if (mapping_ids.size() >= tracking_ids.size()) {
                std::cout << "[GaussianMapper::waitUntilFinished] All frames mapped. Mapper finished." << std::endl;
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    std::cout << "[GaussianMapper::waitUntilFinished] Wait complete." << std::endl;
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

void GaussianMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            std::cout << "[Gaussian Mapper]Local BA Detected." << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    handleNewKeyframe(kf);
                }
            }

            // Get new points
            // auto& associated_points = opr.associatedMapPoints();
            // auto& points = std::get<0>(associated_points);
            // auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && !this->dense_map_points_) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                
                auto& associated_points = opr.associatedMapPoints();
                auto& points = std::get<0>(associated_points);
                auto& colors = std::get<1>(associated_points);
                if(points.size() >= 30)
                    gaussians_->increasePcd(points, colors, getIteration());
            }
            else if(initial_mapped_){
                
            }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
            // Mark the transformed points to avoid transforming more than once
            torch::Tensor point_not_transformed_flags =
                torch::full(
                    {gaussians_->xyz_.size(0)},
                    true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            int num_transformed = 0;
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                // In case new points are added in handleNewKeyframe()
                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        /*dim=*/0);
                // If kf is already in the scene, evaluate the change in pose,
                // if too large we perform loop correction on its visible model points.
                // If not in the scene, create a new one.
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    // If is loop closure kf
// if (std::get<4>(kf)) {
// renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                        Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                        Sophus::SE3f inv_pose = pose.inverse();
                        Sophus::SE3f diff_pose = inv_pose * original_pose;
                        bool large_rot = !diff_pose.rotationMatrix().isApprox(
                            Eigen::Matrix3f::Identity(), large_rot_th_);
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                            1.0, large_trans_th_);
                        if (large_rot || large_trans) {
                            std::cout << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
                                    << kfid << std::endl;
                            diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                            diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                            diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                            {
                                std::unique_lock<std::mutex> lock_render(mutex_render_);
                                gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                    point_not_transformed_flags,
                                    diff_pose_tensor,
                                    pkf->world_view_transform_,
                                    pkf->full_proj_transform_,
                                    pkf->creation_iter_,
                                    stableNumIterExistence(),
                                    num_transformed,
                                    loop_kf_scale); // selected xyz *= s
                            }
                            // Give loop keyframes times of use
                            increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
// renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
// std::cout<<num_transformed<<std::endl;
                        }
// }
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();
// if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                }
                else {
                    handleNewKeyframe(kf);
                }
            }
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
// keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));

            // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
            // auto& associated_points = opr.associatedMapPoints();
            // auto& points = std::get<0>(associated_points);
            // auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && !this->dense_map_points_) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);

                auto& associated_points = opr.associatedMapPoints();
                auto& points = std::get<0>(associated_points);
                auto& colors = std::get<1>(associated_points);
                if(points.size() >= 30)
                    gaussians_->increasePcd(points, colors, getIteration());
            }

            // Mark this iteration
            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      << std::endl;

            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            if (initial_mapped_) {
                // Apply the scaled transformation on gaussian model points
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                // Apply the scaled transformation to the scene
                scene_->applyScaledTransformation(s, T);
            }
            else { // TODO: the workflow should not come here, delete this branch
                // Apply the scaled transformation to the cached points
                for (auto& pt : scene_->cached_point_cloud_) {
                    // pt <- (s * Ryw * pt + tyw)
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }

                // Apply the scaled transformation on gaussian keyframes
                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}

void GaussianMapper::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(
            std::get<0>(kf), getIteration(), &this->kf_params_);
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);

        this->generatePyramidSizes(pkf, camera);

        // pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
        // if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
        //     pkf->gaus_pyramid_height_ = kf_params_.gaus_pyramid_hr_height_;
        //     pkf->gaus_pyramid_width_ = kf_params_.gaus_pyramid_hr_width_;
        // }
        // else{
        //     pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        //     pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        // }
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    // pkf->img_undist_ = imgRGB_undistorted;
    pkf->setGTLRImg(imgRGB_undistorted);
    pkf->img_auxiliary_undist_ = imgAux_undistorted; // !!! setGTLRDpt
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    if(this->kf_params_.align_pose_ && this->kf_params_.render_aligned_){
        pkf->setGTHRImg(this->global_align_time_, this->vstrHRImagePaths_);
        pkf->setGlobalDeltaPose(this->global_align_pose_);
    }

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        this->generatePyramidFrames(pkf);
        // cv::cuda::GpuMat img_gpu;
        // img_gpu.upload(pkf->img_undist_);
        // pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        // for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        //     cv::cuda::GpuMat img_resized;
        //     cv::cuda::resize(img_gpu, img_resized,
        //                         cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
        //     pkf->gaus_pyramid_original_image_[l] =
        //         tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        // }
    }
    else {
        throw std::runtime_error("GaussianMapper::handleNewKeyframe CPU gaus pyramid training not implemented!");
        // pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        // for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        //     cv::Mat img_resized;
        //     cv::resize(pkf->img_undist_, img_resized,
        //                 cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
        //     pkf->gaus_pyramid_original_image_[l] =
        //         tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        // }
    }
}

void GaussianMapper::generateKfidRandomShuffle()
{
// if (viewpoint_sliding_window_.empty())
//     return;

// std::size_t sliding_window_size = viewpoint_sliding_window_.size();
// kfid_shuffle_.resize(sliding_window_size);
// std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
// std::mt19937 g(rd_());
// std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

// auto t2 = std::chrono::steady_clock::now();
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
// std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
{
    std::unordered_set<unsigned long> kfids =
        pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    std::vector<unsigned long> kfids_to_erase;
    std::size_t nkfs = scene_->keyframes().size();
    kfids_to_erase.reserve(nkfs);
    for (auto& kfit : scene_->keyframes()) {
        unsigned long kfid = kfit.first;
        if (kfids.find(kfid) == kfids.end()) {
            kfids_to_erase.emplace_back(kfid);
        }
    }

    for (auto& kfid : kfids_to_erase) {
        scene_->keyframes().erase(kfid);
    }
}

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREO:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBD:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // To clear undisired and unreliable depth
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false/*true*/, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to get 3D points
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        // Add new points to the model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered_image,
        torch::Tensor &rendered_opacity,
        torch::Tensor &rendered_depth,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_opc_dir,
        std::filesystem::path result_dpt_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_image);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_rendered_opacity_) {
        // std::cout<<"[debug] opc min "<<rendered_opacity.min()<<std::endl;
        // std::cout<<"[debug] opc max "<<rendered_opacity.max()<<std::endl;
        auto opacity_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_opacity);
        opacity_cv.convertTo(opacity_cv, CV_8UC1, 255.0f);
        cv::imwrite(result_opc_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), opacity_cv);
    }

    if (record_rendered_depth_) {
        // std::cout<<"[debug] dpt min "<<rendered_depth.min()<<std::endl;
        // std::cout<<"[debug] dpt max "<<rendered_depth.max()<<std::endl;
        auto depth_cv = tensor_utils::torchTensor2CvMat_Float32(rendered_depth);
        depth_cv = depth_cv * this->rendered_depthmap_factor_;
        if(record_rendered_depth_vis_){
            depth_cv.convertTo(depth_cv, CV_8UC1, 255.0f/65536.0f, 0.f);
            cv::imwrite(result_dpt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), depth_cv);
        }
        else{
            depth_cv.convertTo(depth_cv, CV_16UC1);
            cv::imwrite(result_dpt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), depth_cv);
        }
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered_image - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_opc_dir,
    std::filesystem::path result_dpt_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix)
{
    int image_height, image_width;
    bool render_hr;
    torch::Tensor undistort_mask, gt_image;
    if(kf_params_.align_pose_ && kf_params_.render_aligned_){
        image_height = kf_params_.hr_height_;
        image_width = kf_params_.hr_width_;
        render_hr = true;
        undistort_mask = kf_params_.lr_undistort_mask_tensor_;
        gt_image = pkf->getGTHRImg().cuda();
    }
    else{
        image_height = pkf->image_height_;
        image_width = pkf->image_width_;
        render_hr = false;
        undistort_mask = undistort_mask_[pkf->camera_id_];
        gt_image = pkf->original_image_;
    }

    auto start_timing = std::chrono::steady_clock::now();
    auto render_pkg = GaussianRenderer::render(
        pkf,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_,
        render_hr
    );
    auto rendered_image = std::get<0>(render_pkg);
    auto rendered_opacity = std::get<5>(render_pkg);
    auto rendered_depth = std::get<4>(render_pkg);
    torch::Tensor masked_image = rendered_image * undistort_mask;
    torch::Tensor masked_opacity = rendered_opacity * undistort_mask;
    torch::Tensor masked_depth = rendered_depth * undistort_mask;
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    // auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, masked_opacity, masked_depth, gt_image, pkf->fid_, result_img_dir, result_opc_dir, result_dpt_dir, result_gt_dir, result_loss_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path opacity_dir = result_dir / "opacity";
    if (record_rendered_opacity_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(opacity_dir);

    std::filesystem::path depth_dir = result_dir / "depth";
    if (record_rendered_depth_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        if(!(*kfit).second->has_hr_fid_){
            std::cout<<"[GaussianMapper::renderAndRecordAllKeyframes]Skip rendering keyframe "<<(*kfit).first<<" due to no HR image available."<<std::endl;
            continue;
        }

        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, opacity_dir, depth_dir, image_gt_dir, image_loss_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapper::savePly(std::filesystem::path result_dir, bool save_sparse)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");
    if(save_sparse) gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}