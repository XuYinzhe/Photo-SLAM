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

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <map>
#include <random>
#include <mutex>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

#include <jsoncpp/json/json.h>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/ORBVocabulary.h"
#include "ORB-SLAM3/include/ORBextractor.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "third_party/Connected_components_PyTorch/cpp/buf.h"

// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/approximate_voxel_grid.h>
// #include <fast_gicp/gicp/fast_vgicp.hpp>
// #include <fast_gicp/gicp/fast_vgicp_cuda.hpp>

#include "operate_points.h"
#include "stereo_vision.h"
#include "tensor_utils.h"
#include "metrics_utils.h"
#include "gaussian_keyframe.h"
#include "gaussian_scene.h"
#include "gaussian_trainer.h"

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());

using KeyframeFrontend = std::tuple<
    unsigned long,    // pKF->mnId
    unsigned long,    // pKF->mpCamera->GetId()
    Sophus::SE3f,     // pKF->GetPose()
    cv::Mat,          // pKF->imgLeftRGB.clone()
    bool,             // isLoopClosureKF
    cv::Mat,          // pKF->imgAuxiliary
    std::vector<float>, // pixels
    std::vector<float>, // pointsLocal
    std::string         // pKF->mNameFile
>;

struct UndistortParams
{
    UndistortParams(
        const cv::Size& old_size,
        cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))
        : old_size_(old_size)
    {
        dist_coeff.copyTo(dist_coeff_);
    }

    cv::Size old_size_;
    cv::Mat dist_coeff_;
};


enum SystemSensorType
{
    INVALID = 0,
    MONOCULAR = 1,
    STEREO = 2,
    RGBD = 3
};

struct VariableParameters
{
    float position_lr_init;
    float feature_lr;
    float opacity_lr;
    float scaling_lr;
    float rotation_lr;
    float percent_dense;
    float lambda_dssim;
    int opacity_reset_interval;
    float densify_grad_th;
    int densify_interval;
    int new_kf_times_of_use;
    int stable_num_iter_existence; ///< loop closure correction

    bool keep_training;
    bool do_gaus_pyramid_training;
    bool do_inactive_geo_densify;
};

class GaussianMapper
{
public:
    GaussianMapper(
        std::shared_ptr<ORB_SLAM3::System> pSLAM,
        std::filesystem::path gaussian_config_file_path,
        std::filesystem::path result_dir = "",
        int seed = 0,
        torch::DeviceType device_type = torch::kCUDA);

    void readConfigFromFile(std::filesystem::path cfg_path);

    void run();
    void trainColmap();
    void trainForOneIteration();

    bool isStopped();
    void signalStop(const bool going_to_stop = true);

    cv::Mat renderFromPose(
        const Sophus::SE3f &Tcw,
        const int width,
        const int height,
        const bool main_vision = false);

    int getIteration();
    void increaseIteration(const int inc = 1);

    float positionLearningRateInit();
    float featureLearningRate();
    float opacityLearningRate();
    float scalingLearningRate();
    float rotationLearningRate();
    float percentDense();
    float lambdaDssim();
    int opacityResetInterval();
    float densifyGradThreshold();
    int densifyInterval();
    int newKeyframeTimesOfUse();
    int stableNumIterExistence();
    bool isKeepingTraining();
    bool isdoingGausPyramidTraining();
    bool isdoingInactiveGeoDensify();

    void setPositionLearningRateInit(const float lr);
    void setFeatureLearningRate(const float lr);
    void setOpacityLearningRate(const float lr);
    void setScalingLearningRate(const float lr);
    void setRotationLearningRate(const float lr);
    void setPercentDense(const float percent_dense);
    void setLambdaDssim(const float lambda_dssim);
    void setOpacityResetInterval(const int interval);
    void setDensifyGradThreshold(const float th);
    void setDensifyInterval(const int interval);
    void setNewKeyframeTimesOfUse(const int times);
    void setStableNumIterExistence(const int niter);
    void setKeepTraining(const bool keep);
    void setDoGausPyramidTraining(const bool gaus_pyramid);
    void setDoInactiveGeoDensify(const bool inactive_geo_densify);

    VariableParameters getVaribleParameters();
    void setVaribleParameters(const VariableParameters &params);

    GaussianModelParams& getGaussianModelParams() { return this->model_params_; }
    void setColmapDataPath(std::filesystem::path colmap_path) { this->model_params_.source_path_ = colmap_path; }
    void setSensorType(SystemSensorType sensor_type) { this->sensor_type_ = sensor_type; }

    void setOtherData(const std::vector<std::string>& paths, 
        const std::vector<double>& timestamps = {},
        const std::vector<std::vector<double>>& lr_gt_poses = {},
        const std::vector<std::vector<double>>& hr_gt_poses = {});

    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");

protected:
    bool hasMetInitialMappingConditions();
    bool hasMetIncrementalMappingConditions();

    void combineMappingOperations();

    void handleNewKeyframe(std::tuple<unsigned long,
                                      unsigned long,
                                      Sophus::SE3f,
                                      cv::Mat,
                                      bool,
                                      cv::Mat,
                                      std::vector<float>,
                                      std::vector<float>,
                                      std::string> &kf);
    void generateKfidRandomShuffle();
    std::shared_ptr<GaussianKeyframe> useOneRandomSlidingWindowKeyframe();
    std::shared_ptr<GaussianKeyframe> useOneRandomKeyframe();
    void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> pkf, int times);
    void cullKeyframes();

    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<GaussianKeyframe> pkf);

    // bool needInterruptTraining();
    // void setInterruptTraining(const bool interrupt_training);

    // cv::Mat sampleDepthMap(const cv::Mat& depth);
    // void cacheSampledDepthMap(std::shared_ptr<GaussianKeyframe> pkf, Sophus::SE3<float>& pose);
    void cacheKeyframeDepthMap();
    cv::Mat getDepthRelated(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
        Sophus::SE3f pose1, Sophus::SE3f pose2);
    cv::Mat getDepthRelated(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
        torch::Tensor pose1, torch::Tensor pose2, torch::Tensor give_depth);
    cv::Mat getDepthDiff(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
        bool give_poses = false, Sophus::SE3f pose1 = Sophus::SE3f(), Sophus::SE3f pose2 = Sophus::SE3f());
    cv::Mat getDepthDiff(std::shared_ptr<GaussianKeyframe> pkf1, std::shared_ptr<GaussianKeyframe> pkf2, 
        torch::Tensor pose1, torch::Tensor pose2);

    void optimizeGlobalAlign();
    void getAvgGlobalPose(std::vector<std::size_t>& kfids, bool soften = false, float soften_ratio = 0.1f);
    void getAvgGlobalPose(std::vector<std::size_t>& kfids, torch::Tensor& avg_pose, bool soften = false, float soften_ratio = 0.1f);

    float getRelatedPoseGMS(std::shared_ptr<GaussianKeyframe> pkf, 
        const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2,
        const std::vector<int>& match12,
        const cv::Mat& depth1,
        const cv::Mat& K2,
        cv::Mat& rvec, cv::Mat& tvec, 
        std::vector<int>& inliers,
        bool is_lr = true,
        float hr_resize_ratio = 1.0f,
        bool use_lr2hr_depth = false
    );

    // torch::Tensor refinePoseFastVGICP(std::shared_ptr<GaussianKeyframe> pkf);

    std::size_t handleKeyframeFrontend(KeyframeFrontend& kf, std::shared_ptr<GaussianKeyframe> new_kf, float timestamp);
    float getRsizedHRScale(float ratio, int& out_width, int& out_height);
    // current methods only use diff or classic, ours hyper robust to more situations, for tracking contribution
    void getBatchShuffledFrameIds(const std::vector<std::size_t>& in_fids, std::vector<std::size_t>& out_fids, int required_iters);
    float optimizeGlobalLRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose = true);
    float optimizeGlobalLRImg(std::shared_ptr<GaussianKeyframe> pkf, int i);
    float optimizeGlobalHRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose = true);
    float optimizeLocalLRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose = true);
    float optimizeLocalHRPose(std::shared_ptr<GaussianKeyframe> pkf, bool use_differential_pose = true);
    torch::Tensor getLocalLRValidDptMsk(std::shared_ptr<GaussianKeyframe> pkf);
    int insertLocalLRValidDpts(std::vector<torch::Tensor>& valid_depth_masks, std::vector<std::size_t>& valid_fids);
    void optimizeInsertedLocalLRDpts(std::vector<std::size_t>& valid_fids);
    void insertLocalLRValidDpt(std::shared_ptr<GaussianKeyframe> pkf);
    float optimizeLocalHRImgs(std::vector<std::size_t>& random_kfids, std::vector<std::size_t>& valid_fids);
    float optimizeLocalHRImg(std::shared_ptr<GaussianKeyframe> pkf);

    void insertNewKeyframesFromSLAM();
    // void insertOneKeyframe(std::tuple<
    //     unsigned long,
    //     unsigned long,
    //     Sophus::SE3f,
    //     cv::Mat,
    //     bool,
    //     cv::Mat,
    //     std::vector<float>,
    //     std::vector<float>,
    //     std::string> &kf, double timestamp);
    void insertOneKeyframe_old(KeyframeFrontend& kf, double timestamp);
    void insertOneKeyframe(KeyframeFrontend& kf, double timestamp);
    void insertBatchKeyframes(
        std::vector<std::shared_ptr<KeyframeFrontend>>& kfs, 
        std::vector<double>& timestamps);

    // void undistortKeyframe(std::shared_ptr<GaussianKeyframe> pkf, std::size_t camera_id);
    void generatePyramidSizes(std::shared_ptr<GaussianKeyframe> pkf, const Camera& camera);
    void generatePyramidFrames(std::shared_ptr<GaussianKeyframe> pkf);

    void recordKeyframeRendered(
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
        std::string name_suffix = "");
    void renderAndRecordKeyframe(
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
        std::string name_suffix = "");
    void renderAndRecordAllKeyframes(
        std::string name_suffix = "");

    void savePly(std::filesystem::path result_dir, bool save_sparse = true);
    void keyframesToJson(std::filesystem::path result_dir);
    void saveModelParams(std::filesystem::path result_dir);
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = "");

public:
    // Parameters
    std::filesystem::path config_file_path_;

    // Model
    std::shared_ptr<GaussianModel> gaussians_;
    std::shared_ptr<GaussianScene> scene_;

    // SLAM system
    std::shared_ptr<ORB_SLAM3::System> pSLAM_;

    // Settings
    torch::DeviceType device_type_;
    int num_gaus_pyramid_sub_levels_ = 0;
    std::vector<int> kf_gaus_pyramid_times_of_use_;
    std::vector<float> kf_gaus_pyramid_factors_;

    bool viewer_camera_id_set_ = false;
    std::uint32_t viewer_camera_id_ = 0;
    float rendered_image_viewer_scale_ = 1.0f;
    float rendered_image_viewer_scale_main_ = 1.0f;

    float z_near_ = 0.01f;
    float z_far_ = 100.0f;

    // Data
    bool kfid_shuffled_ = false;
    std::map<camera_id_t, torch::Tensor> undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;
    
    torch::Tensor hr_undistort_mask_;

protected:
    // Parameters
    GaussianModelParams model_params_;
    GaussianOptimizationParams opt_params_;
    GaussianPipelineParams pipe_params_;
    KeyframeOptimizationParams kf_params_;

    // Data
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> viewpoint_sliding_window_;
    std::vector<std::size_t> kfid_shuffle_;
    std::size_t kfid_shuffle_idx_ = 0;
    std::map<std::size_t, int> kfs_used_times_;

    // Status
    bool initial_mapped_;
    bool interrupt_training_;
    bool stopped_;
    int iteration_;
    float ema_loss_for_log_;
    bool SLAM_ended_;
    bool loop_closure_iteration_;
    bool keep_training_ = false;
    int default_sh_ = 0;

    // Settings
    SystemSensorType sensor_type_;

    float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0;
    float stereo_baseline_length_ = 0.0f;
    int stereo_min_disparity_ = 0;
    int stereo_num_disparity_ = 128;
    cv::Mat stereo_Q_;
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;
    float RGBD_min_depth_ = 0.0f;
    float RGBD_max_depth_ = 100.0f;

    bool inactive_geo_densify_ = true;
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;

    unsigned long min_num_initial_map_kfs_;
    torch::Tensor background_;
    float large_rot_th_;
    float large_trans_th_;
    torch::Tensor override_color_;

    int new_keyframe_times_of_use_;
    int local_BA_increased_times_of_use_;
    int loop_closure_increased_times_of_use_;

    bool cull_keyframes_;
    int stable_num_iter_existence_;

    bool do_gaus_pyramid_training_;

    // dense mapping initialization
    torch::Tensor dense_init_pcd_xyz_, dense_init_pcd_rgb_, dense_init_pcd_idx_;
    bool dense_map_points_;
    float dense_sample_num_; // deprecated
    int dense_height_num_, dense_width_num_; // deprecated
    float dense_diff_thld_;
    float dense_diff_thld_ratio_;
    int dense_fid_offset_ = 2;
    std::vector<std::size_t> dense_init_fids_;
    // std::vector<std::unordered_map<int, int>> dense_init_fids_valid_pixels_;
    std::unordered_map<int, int> dense_init_fid_valid_pixels_;
    Eigen::ArrayXf dense_height_map_, dense_width_map_;
    torch::Tensor dense_height_map_tensor_, dense_width_map_tensor_;

    // global alignment
    int init_train_iter_;
    int max_pnp_render_attempts_ = 3;
    int global_align_lr_pose_iter_;
    int global_align_lr_iter_;
    float global_align_lr_depth_lambda_;
    float global_align_lr_fix_mean3d_thld_ = 0.1f;
    int global_align_hr_iter_;
    int global_align_hr_warmup_iter_;
    float global_align_hr_warmup_lr_dump_;
    float global_align_hr_resize_ratio_;
    float global_align_time_window_ratio_;
    float global_align_opacity_thr_;
    float global_align_hr_opacity_thr_;
    float global_align_hr_pose_lambda_;
    float global_align_hr_pose_reg_lambda_;
    float global_align_hr_pose_depth_lambda_;
    int global_align_hr_pose_iter_;
    std::vector<int> global_align_select_fids_;
    int global_align_hr_color_iter_;
    float global_align_hr_color_lambda_;

    // local alignment
    std::vector<std::map<ORB_SLAM3::MappingOperation::OprType, std::vector<std::size_t>>> local_mapping_operations_;
    int local_align_lr_pose_iter_;
    int local_align_lr_iter_;
    float local_align_lr_opcacity_thr_;
    float local_align_hr_resize_ratio_;
    int local_align_lr_joint_pose_iter_;
    float local_align_lr_joint_pose_dump_;
    int local_align_hr_pose_iter_;
    float local_align_hr_opacity_thr_;
    float local_align_hr_pose_reg_lambda_;
    float local_align_hr_pose_opacity_lambda_;
    float local_align_hr_pose_depth_lambda_;
    int local_align_hr_color_iter_;
    float local_align_hr_color_loss_thr_;
    int local_align_batch_size_;
    std::unordered_map<std::size_t, double> local_align_batch_fids_timestamps_map_;
    float local_align_batch_fillholes_opacity_thr_;
    int local_align_batch_conn_comp_min_size_;
    int local_align_batch_color_periter_;

    // gt related
    std::vector<std::string> vstrHRImagePaths_;
    std::vector<double> vHRTimestamps_;
    std::vector<std::vector<double>> vvLRGTPose_;
    std::vector<std::vector<double>> vvHRGTPose_;
    bool hr_timestamps_exist_ = false;
    bool gt_pose_exist_ = false;
    std::vector<std::size_t> global_algin_fids_;
    float global_align_time_ = 0.f;
    torch::Tensor global_align_pose_;

    // recording
    std::filesystem::path result_dir_;
    int keyframe_record_interval_;
    int all_keyframes_record_interval_;
    bool record_rendered_image_;
    bool record_rendered_opacity_;
    bool record_rendered_depth_;
    bool record_rendered_depth_vis_;
    bool record_ground_truth_image_;
    bool record_loss_image_;
    float rendered_depthmap_factor_ = 1000;

    int training_report_interval_;
    bool record_loop_ply_;

    int prune_big_point_after_iter_;
    float densify_min_opacity_ = 20;

    // Tools
    std::random_device rd_;
    std::shared_ptr<torch::jit::script::Module> lpips_model_;

    // Mutex
    std::mutex mutex_status_;
    std::mutex mutex_settings_;
    std::mutex mutex_render_; ///< the model is suppose to be read-only from outside
};