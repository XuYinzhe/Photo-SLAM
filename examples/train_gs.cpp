// test deblur

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>
#include <map>
#include <random>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

#include <torch/torch.h>

#include <jsoncpp/json/json.h>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "include/gaussian_mapper.h"
#include "include/operate_points.h"
#include "include/stereo_vision.h"
#include "include/tensor_utils.h"
#include "include/gaussian_keyframe.h"
#include "include/gaussian_scene.h"
#include "include/gaussian_trainer.h"

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);
void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);
void LoadPathConfig(std::filesystem::path cfg_path, 
    std::filesystem::path& strMapperPath, 
    std::filesystem::path& strDataPath, 
    std::filesystem::path& strGTPath,
    std::filesystem::path& strSavingPath, 
    bool& bViewer);
void readGSConfig(std::filesystem::path cfg_path,
    GaussianModelParams& model_params_,
    GaussianOptimizationParams& opt_params_,
    GaussianPipelineParams& pipe_params_);
void LoadDataset(const std::filesystem::path& strDataPath,const std::filesystem::path& strGTPath,
    std::vector<std::string>& vstrImageNamesRGB, std::vector<std::string>& vstrImageNamesRGBGT,
    std::vector<std::string>& vstrImagePathsRGB, std::vector<std::string>& vstrImagePathsRGBGT,
    std::vector<std::vector<float>>& vvfloatVelLin, std::vector<std::vector<float>>& vvfloatVelAng,
    std::vector<std::vector<float>>& vvfloatPose, std::vector<std::vector<float>>& vvfloatPLY,
    Camera& camera);
void train(
    std::shared_ptr<GaussianScene> sceneGS,
    std::shared_ptr<GaussianModel> modelGS,
    GaussianModelParams& model_params,
    GaussianOptimizationParams& opt_params,
    GaussianPipelineParams& pipe_params);

int main(int argc, char* argv[]) {

    if (argc != 2){
        std::cerr << "[Error]Wrong arguments input." << std::endl;
    }

    std::cout<< "Load cfg file." <<std::endl;
    std::filesystem::path strMapperPath; 
    std::filesystem::path strDataPath;
    std::filesystem::path strGTPath;
    std::filesystem::path strSavingPath;
    bool bViewer;
    LoadPathConfig(std::string(argv[1]), strMapperPath, strDataPath, strGTPath, strSavingPath, bViewer);

    std::cout<< "Load dataset." <<std::endl;
    // Retrieve paths to images
    std::vector<std::string> vstrImageNamesRGB;
    std::vector<std::string> vstrImageNamesRGBGT;
    std::vector<std::string> vstrImagePathsRGB;
    std::vector<std::string> vstrImagePathsRGBGT;
    std::vector<std::vector<float>> vvfloatVelLin;
    std::vector<std::vector<float>> vvfloatVelAng;
    std::vector<std::vector<float>> vvfloatPose;
    std::vector<std::vector<float>> vvfloatPLY;
    Camera camera; camera.setModelId(Camera::CameraModelType::PINHOLE);
    LoadDataset(strDataPath, strGTPath, 
        vstrImageNamesRGB, vstrImageNamesRGBGT, 
        vstrImagePathsRGB, vstrImagePathsRGBGT,
        vvfloatVelLin, vvfloatVelAng, vvfloatPose, 
        vvfloatPLY, camera);

    std::cout<< "Load params." <<std::endl;
    GaussianModelParams model_params;
    GaussianOptimizationParams opt_params;
    GaussianPipelineParams pipe_params;
    readGSConfig(strMapperPath, model_params, opt_params, pipe_params);

    auto modelGS = std::make_shared<GaussianModel>(model_params);
    auto sceneGS = std::make_shared<GaussianScene>(model_params);

    std::cout<< "Load mapping points." <<std::endl;
    std::cout<< "Points: " << vvfloatPLY.size() <<std::endl;
    std::cout<< "Cams: " << vstrImageNamesRGB.size() <<std::endl;
    bool kfid_shuffled = false;
    for(size_t i=0; i<vvfloatPLY.size(); i++){
        auto& mp = vvfloatPLY[i];
        Point3D point3D;
        point3D.xyz_(0) = mp[0];
        point3D.xyz_(1) = mp[1];
        point3D.xyz_(2) = mp[2];
        point3D.color_(0) = mp[3]/255.f;
        point3D.color_(1) = mp[4]/255.f;
        point3D.color_(2) = mp[5]/255.f;
        sceneGS->cachePoint3D(i, point3D);
    }
    std::cout<< "Load keyframes." <<std::endl;
    for(size_t i=0; i<vstrImageNamesRGB.size(); i++){
        std::shared_ptr<GaussianKeyframe> kf = std::make_shared<GaussianKeyframe>(i, 0);
        kf->setCameraParams(camera);
        kf->zfar_ = 100.f; kf->znear_ = 0.01f;
        kf->setPose(vvfloatPose[i], true);
        kf->img_filename_ = vstrImageNamesRGB[i];

        auto imgRGB = cv::imread(vstrImagePathsRGB[i], cv::IMREAD_UNCHANGED);
        cv::cvtColor(imgRGB, imgRGB, CV_BGR2RGB);
        imgRGB.convertTo(imgRGB, CV_32FC3, 1.0 / 255.0);
        kf->original_image_ = 
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB, torch::kCUDA);
        kf->computeTransformTensors();
        kf->enable_optim_exposure = false;
        kf->enable_optim_pose = false;
        kf->enable_optim_velocity = false;
        sceneGS->addKeyframe(kf, &kfid_shuffled);
    }

    std::cout<< "Init scene and model." <<std::endl;
    sceneGS->cameras_extent_ = std::get<1>(sceneGS->getNerfppNorm());
    modelGS->createFromPcd(sceneGS->cached_point_cloud_, sceneGS->cameras_extent_);

    std::cout<< "Start training." <<std::endl;
    train(sceneGS, modelGS, model_params, opt_params, pipe_params);

    return 0;
}

void train(
    std::shared_ptr<GaussianScene> sceneGS,
    std::shared_ptr<GaussianModel> modelGS,
    GaussianModelParams& model_params,
    GaussianOptimizationParams& opt_params,
    GaussianPipelineParams& pipe_params)
{
    float ema_loss_for_log = 0.0f;

    modelGS->trainingSetup(opt_params);
    sceneGS->setupKeyframeOptimization(opt_params);
    // sceneGS->setupKeyframeOptimization(opt_params);

    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};
    torch::Tensor background_color = torch::tensor(bg_color, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto override_color = torch::empty(0, torch::TensorOptions().device(torch::kCUDA));

    for(int iter = 1; iter<=opt_params.iterations_; iter++)
    {
        auto iter_start_timing = std::chrono::steady_clock::now();
        
        auto viewpoint_stack = sceneGS->keyframes();
        modelGS->updateLearningRate(iter);

        if(iter%1000 == 0) modelGS->oneUpShDegree();
    
        // random kf
        int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / static_cast<int>(viewpoint_stack.size()));
        auto random_cam_it = viewpoint_stack.begin();
        for(int i=0; i<random_cam_idx; i++) random_cam_it++;
        std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;
    
        auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        viewpoint_cam->image_height_,
        viewpoint_cam->image_width_,
        modelGS,
        pipe_params,
        background_color,
        override_color);

        auto rendered_image = std::get<0>(render_pkg);
        auto viewspace_point_tensor = std::get<1>(render_pkg);
        auto visibility_filter = std::get<2>(render_pkg);
        auto radii = std::get<3>(render_pkg);
        // auto depth = std::get<4>(render_pkg);
        // auto opacity = std::get<5>(render_pkg);
        // auto n_touched = std::get<6>(render_pkg);

        // std::cout<<"render size"<<std::endl;
        // std::cout<<visibility_filter.sizes()<<std::endl;
        // std::cout<<visibility_filter.sum()<<std::endl;
        // std::cout<<radii.max()<<std::endl;
        // std::cout<<radii.min()<<std::endl;
        // std::cout<<radii.median()<<std::endl;
        // std::cout<<rendered_image.sizes()<<std::endl;
        // std::cout<<viewspace_point_tensor.sizes()<<std::endl;
        // std::cout<<radii.sizes()<<std::endl;
        auto Ll1 = loss_utils::l1_loss(rendered_image, viewpoint_cam->original_image_);
        float lambda_dssim = opt_params.lambda_dssim_;
        auto loss = (1.0 - lambda_dssim) * Ll1;
        loss += lambda_dssim * (1.0 - loss_utils::ssim(rendered_image, viewpoint_cam->original_image_, torch::kCUDA));
        
        // std::cout<<"bw1 "<<loss.data()<<std::endl;
        loss.backward();
        // std::cout<<"bw2 "<<loss.data()<<std::endl;

        torch::cuda::synchronize();
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end_timing - iter_start_timing).count();
        {
            torch::NoGradGuard no_grad;

            ema_loss_for_log = 0.4f * loss.item().toFloat() + 
                0.6 * ema_loss_for_log;
            GaussianTrainer::trainingReport(
                iter,
                opt_params.iterations_,
                Ll1,
                loss,
                ema_loss_for_log,
                loss_utils::l1_loss,
                iter_time,
                *modelGS,
                *sceneGS,
                pipe_params,
                background_color,
                false
            );

            // Densification
            if (iter < opt_params.densify_until_iter_) {
                // Keep track of max radii in image-space for pruning
                modelGS->max_radii2D_.index_put_(
                    {visibility_filter},
                    torch::max(modelGS->max_radii2D_.index({visibility_filter}),
                               radii.index({visibility_filter})));
                // std::cout<< "[debug output]addDensificationStats" << std::endl;
                modelGS->addDensificationStats(viewspace_point_tensor, visibility_filter);

                if ((iter > opt_params.densify_from_iter_) && (iter % opt_params.densification_interval_ == 0)) {
                    int size_threshold = (iter > opt_params.opacity_reset_interval_) ? 20 : 0;
                    // std::cout<< "[debug output]densifyAndPrune" << std::endl;
                    modelGS->densifyAndPrune(opt_params.densify_grad_threshold_, 0.005, sceneGS->cameras_extent_, size_threshold);
                }

                if (opt_params.opacity_reset_interval_ && iter % opt_params.opacity_reset_interval_ == 0 || 
                    (model_params.white_background_ && iter == opt_params.densify_from_iter_)
                ){
                    // std::cout<< "[debug output]modelGS->resetOpacity()" << std::endl;
                    modelGS->resetOpacity();
                }
            }

            // std::cout<< "[debug output]optim" << std::endl;

            if(iter < opt_params.iterations_){
                modelGS->optimizer_->step();
                modelGS->optimizer_->zero_grad();
                sceneGS->optimizer_->step();
                sceneGS->optimizer_->zero_grad();
            }
        }
        // std::cout<<"iter "<< iter<<std::endl;
    }
}

void LoadDataset(const std::filesystem::path& strDataPath,const std::filesystem::path& strGTPath,
    std::vector<std::string>& vstrImageNamesRGB, std::vector<std::string>& vstrImageNamesRGBGT,
    std::vector<std::string>& vstrImagePathsRGB, std::vector<std::string>& vstrImagePathsRGBGT,
    std::vector<std::vector<float>>& vvfloatVelLin, std::vector<std::vector<float>>& vvfloatVelAng,
    std::vector<std::vector<float>>& vvfloatPose, std::vector<std::vector<float>>& vvfloatPLY,
    Camera& camera)
{
    auto strImgPath = strDataPath / "images";
    auto strImgGTPath = strGTPath / "images";
    auto strPLYPath = strDataPath / "sparse_pc.ply";
    auto strJsonPath = strDataPath / "transforms.json";

    // images
    auto _sort = [](const std::string& s1, const std::string& s2) {
            return std::stoi(s1.substr(0,3)) < std::stoi(s2.substr(0,3));};

    std::vector<std::string> filenames = {};
    for(const auto& file : std::filesystem::directory_iterator(strImgPath))
        filenames.push_back(file.path().filename().string());
    std::sort(filenames.begin(), filenames.end(), _sort);
    for(const auto& file : filenames){
        vstrImageNamesRGB.push_back(file);
        vstrImagePathsRGB.push_back(strImgPath / file);
    }
    
    filenames = {};
    for(const auto& file : std::filesystem::directory_iterator(strImgGTPath))
        filenames.push_back(file.path().filename().string());
    std::sort(filenames.begin(), filenames.end(), _sort);
    for(const auto& file : filenames){
        vstrImageNamesRGBGT.push_back(file);
        vstrImagePathsRGBGT.push_back(strImgGTPath / file);
    }

    // json
    Json::Value json;
    Json::CharReaderBuilder reader;
    std::string errs;
    std::ifstream jsonfile(strJsonPath);
    if (!jsonfile.is_open()) throw std::runtime_error("Error opening file: " + strJsonPath.string());
    if (!Json::parseFromStream(reader, jsonfile, &json, &errs)) throw std::runtime_error("Error parsing JSON: " + errs);

    camera.width_ = json["w"].asInt();
    camera.height_ = json["h"].asInt();
    camera.params_[0] = json["fl_x"].asDouble();
    camera.params_[1] = json["fl_y"].asDouble();
    camera.params_[2] = json["cx"].asDouble();
    camera.params_[3] = json["cy"].asDouble();
    camera.exposure_time = json["exposure_time"].asFloat();
    camera.rolling_shutter_time = json["rolling_shutter_time"].asFloat();

    vvfloatVelLin.resize(json["frames"].size());
    vvfloatVelAng.resize(json["frames"].size());
    vvfloatPose.resize(json["frames"].size());
    for(const auto& frame : json["frames"]){
        std::vector<float> vfloatVelLin;
        for(const auto& e : frame["camera_linear_velocity"])
            vfloatVelLin.push_back(e.asFloat());
        std::vector<float> vfloatVelAng;
        for(const auto& e : frame["camera_angular_velocity"])
            vfloatVelAng.push_back(e.asFloat());
        std::vector<float> vfloatPose;
        for(const auto& row : frame["transform_matrix"])
            for(const auto& e: row)
                vfloatPose.push_back(e.asFloat());

        std::string filename = frame["file_path"].asString();
        size_t index = std::stoi(filename.substr(9,3)); //"./images/000.png"
        vvfloatVelLin[index] = vfloatVelLin;
        vvfloatVelAng[index] = vfloatVelAng;
        vvfloatPose[index] = vfloatPose;
    }

    // ply
    std::ifstream plyfile(strPLYPath);
    std::vector<float> vfloatPLY;
    std::string line;
    if (!plyfile.is_open()) throw std::runtime_error("Failed to open file: " + strPLYPath.string());

    int vertex_count = 0;
    bool header_end = false;
    while (std::getline(plyfile, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string dummy;
            iss >> dummy >> dummy >> vertex_count;
        }
        if (line == "end_header") {
            header_end = true;
            break;
        }
    }

    if (!header_end) throw std::runtime_error("Invalid PLY file - missing end_header");

    for (int i = 0; i < vertex_count; ++i) {
        std::getline(plyfile, line);
        std::istringstream iss(line);
        
        float x,y,z,r,g,b;
        if (!(iss >> x >> y >> z >> r >> g >> b))
            throw std::runtime_error("Error parsing vertex data at line " + std::to_string(i+1));
        
        std::vector<float> vfloatPLY = {x,y,z,r,g,b};
        vvfloatPLY.push_back(vfloatPLY);
    }

}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }

    // std::sort(vTimesTrack.begin(), vTimesTrack.end());
    // out << "-------" << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "mean tracking time: " << totaltime / nImages << std::endl;

    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}

void LoadPathConfig(std::filesystem::path cfg_path, 
    std::filesystem::path& strMapperPath, 
    std::filesystem::path& strDataPath, 
    std::filesystem::path& strGTPath,
    std::filesystem::path& strSavingPath, 
    bool& bViewer)
{

    cv::FileStorage cfg(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!cfg.isOpened()) {
       std::cerr << "[Error]Failed to open config file at: " << cfg_path << std::endl;
       exit(-1);
    }
    else std::cout << "Reading parameters from " << cfg_path << std::endl;

    strMapperPath = cfg["Mapper.Parameters"];
    strDataPath = cfg["Dataset"];
    strGTPath = cfg["GT"];
    strSavingPath = cfg["Results.Saving"];
    bViewer = (cfg["Results.Viewer"].operator int()) != 0;

    strSavingPath = strSavingPath / "";
    std::filesystem::create_directories(strSavingPath);

    bool allExist = std::filesystem::exists(strMapperPath) && 
        std::filesystem::exists(strDataPath) && 
        std::filesystem::exists(strGTPath);

    if (!allExist) {
        std::cerr << "[Error]Parameters error in config file at: " << cfg_path << std::endl;
        exit(-1);
    }
}

void readGSConfig(std::filesystem::path cfg_path,
    GaussianModelParams& model_params_,
    GaussianOptimizationParams& opt_params_,
    GaussianPipelineParams& pipe_params_)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }


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
    // z_near_ =
    //     settings_file["Camera.z_near"].operator float();
    // z_far_ =
    //     settings_file["Camera.z_far"].operator float();

    // monocular_inactive_geo_densify_max_pixel_dist_ =
    //     settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    // stereo_min_disparity_ =
    //     settings_file["Stereo.min_disparity"].operator int();
    // stereo_num_disparity_ =
    //     settings_file["Stereo.num_disparity"].operator int();
    // RGBD_min_depth_ =
    //     settings_file["RGBD.min_depth"].operator float();
    // RGBD_max_depth_ =
    //     settings_file["RGBD.max_depth"].operator float();

    // inactive_geo_densify_ =
    //     (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    // max_depth_cached_ =
    //     settings_file["Mapper.depth_cache"].operator int();
    // min_num_initial_map_kfs_ = 
    //     static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    // new_keyframe_times_of_use_ = 
    //     settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    // local_BA_increased_times_of_use_ = 
    //     settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    // loop_closure_increased_times_of_use_ = 
    //     settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    // cull_keyframes_ =
    //     (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    // large_rot_th_ =
    //     settings_file["Mapper.large_rotation_threshold"].operator float();
    // large_trans_th_ =
    //     settings_file["Mapper.large_translation_threshold"].operator float();
    // stable_num_iter_existence_ =
    //     settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    // do_gaus_pyramid_training_ =
    //     (settings_file["GausPyramid.do"].operator int()) != 0;
    // num_gaus_pyramid_sub_levels_ =
    //     settings_file["GausPyramid.num_sub_levels"].operator int();
    // int sub_level_times_of_use =
    //     settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    // kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    // kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    // for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    //     kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
    //     kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    // }

    // keyframe_record_interval_ = 
    //     settings_file["Record.keyframe_record_interval"].operator int();
    // all_keyframes_record_interval_ = 
    //     settings_file["Record.all_keyframes_record_interval"].operator int();
    // record_rendered_image_ = 
    //     (settings_file["Record.record_rendered_image"].operator int()) != 0;
    // record_ground_truth_image_ = 
    //     (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    // record_loss_image_ = 
    //     (settings_file["Record.record_loss_image"].operator int()) != 0;
    // training_report_interval_ = 
    //     settings_file["Record.training_report_interval"].operator int();
    // record_loop_ply_ =
    //     (settings_file["Record.record_loop_ply"].operator int()) != 0;

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

    // opt_params_.cam_rotation_lr_;
    // opt_params_.cam_translation_lr_;
    // opt_params_.cam_linear_velocity_lr_;
    // opt_params_.cam_angular_velocity_lr_;

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    // std::cout<<settings_file["Optimization.densification_interval"].operator int()<<std::endl;
    // std::cout<<opt_params_.densification_interval_<<std::endl;
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    // std::cout<<settings_file["Optimization.opacity_reset_interval"].operator int()<<std::endl;
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    // prune_big_point_after_iter_ =
    //     settings_file["Optimization.prune_big_point_after_iter"].operator int();
    // densify_min_opacity_ =
    //     settings_file["Optimization.densify_min_opacity"].operator float();

    // // Viewer Parameters
    // rendered_image_viewer_scale_ =
    //     settings_file["GaussianViewer.image_scale"].operator float();
    // rendered_image_viewer_scale_main_ =
    //     settings_file["GaussianViewer.image_scale_main"].operator float();
}

void readOrbConfig(std::filesystem::path cfg_path,
    Camera& camera){
    
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    camera.width_ = settings_file["Camera.width"].operator int();
    camera.height_ = settings_file["Camera.height"].operator int();

    camera.params_[0] = settings_file["Camera1.fx"].operator double();
    camera.params_[1] = settings_file["Camera1.fy"].operator double();
    camera.params_[2] = settings_file["Camera1.cx"].operator double();
    camera.params_[3] = settings_file["Camera1.cy"].operator double();

    camera.dist_coeff_ = (cv::Mat_<float>(1, 5) << 
        settings_file["Camera1.k1"].operator float(), 
        settings_file["Camera1.k2"].operator float(), 
        settings_file["Camera1.p1"].operator float(),  
        settings_file["Camera1.p2"].operator float(), 
        settings_file["Camera1.k3"].operator float());

}

