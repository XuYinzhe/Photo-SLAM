#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>
#include <iostream>
#include <memory>

#include "loss_utils.h"

namespace metrics_utils
{

inline double compute_psnr(const torch::Tensor& img1, const torch::Tensor& img2, double max_val = 1.0) {
    TORCH_CHECK(img1.sizes() == img2.sizes(), "Images must have same shape for PSNR");
    auto mse = torch::mean(torch::pow(img1 - img2, 2));
    auto psnr = -10.0 * torch::log(mse) / torch::log(torch::tensor(10.0, img1.device()));
    return psnr.item<double>();
}

// consistent to loss_utils::ssim
inline double compute_ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true
){
    return loss_utils::ssim(img1, img2, device_type, window_size, size_average).item<double>();
}

inline std::shared_ptr<torch::jit::script::Module> load_lpips_model(const std::string& path, torch::Device device = torch::kCUDA) {
    try {
        auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(path, device));
        module->eval();
        return module;
    } catch (const c10::Error& e) {
        std::cerr << "[metrics_utils::load_lpips_model] Error loading LPIPS TorchScript model: " << e.msg() << std::endl;
        throw;
    }
}

inline double compute_lpips(std::shared_ptr<torch::jit::script::Module> lpips_model,
                            const torch::Tensor& img1, const torch::Tensor& img2) {
    TORCH_CHECK(lpips_model != nullptr, "[metrics_utils::compute_lpips] LPIPS model must be loaded");
    auto input1 = img1 * 2.0 - 1.0;
    auto input2 = img2 * 2.0 - 1.0;
    auto output = lpips_model->forward({input1, input2}).toTensor();
    return output.item<double>();
}

inline void report_metrics(
    torch::Tensor &rendered_img,
    torch::Tensor &gt_img,
    std::shared_ptr<torch::jit::script::Module> lpips_model,
    double &psnr,
    double &ssim,
    double &lpips
){
    // Ensure images are in the same device
    if (rendered_img.device().type() != gt_img.device().type()){
        gt_img = gt_img.to(rendered_img.device());
    }

    psnr = compute_psnr(rendered_img, gt_img, 1.0);
    ssim = compute_ssim(rendered_img, gt_img, rendered_img.device().type(), 11, true);
    // lpips = compute_lpips(lpips_model, rendered_img.unsqueeze(0), gt_img.unsqueeze(0));
    lpips = 0;

    std::cout<<"[metrics_utils::report_metrics] PSNR: "<<psnr
        <<", SSIM: "<<ssim
        <<", LPIPS: "<<lpips<<std::endl;
}

inline void report_metrics(
    torch::Tensor &rendered_img,
    torch::Tensor &gt_img,
    std::shared_ptr<torch::jit::script::Module> lpips_model
){
    double psnr, ssim, lpips;
    report_metrics(rendered_img, gt_img, lpips_model, psnr, ssim, lpips);
}


}