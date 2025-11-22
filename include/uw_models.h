#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <tuple>

namespace uw {

// Constants (same values as in the Python snippet)
static const std::vector<double> WATER_BETA_D = {2.6, 2.4, 1.8};
static const std::vector<double> WATER_BETA_B = {1.9, 1.7, 1.4};
static const std::vector<double> WATER_B_INF = {0.07, 0.2, 0.39};
static constexpr double FOG_BETA_B = 2.4;

// Utility: estimate atmospheric light (DCP-based), dark channel, and renderers.
// All functions are device-aware: they use the device/dtype of the input tensors.
torch::Tensor dark_channel_estimate(const torch::Tensor& rgb, int64_t patch_size = 41);
torch::Tensor estimate_atmospheric_light(const torch::Tensor& rgb);
torch::Tensor render_uw(const torch::Tensor& rgb, const torch::Tensor& depth);

torch::Tensor normalize_depth(torch::Tensor& depth_raw, torch::Tensor& opc, float normalize_depth = 1.f, bool norm_depth_max = true);

class BackscatterNet : public torch::nn::Module {
public:
    // Constructor args mimic Python:
    // - use_residual: include the J_prime * exp(-b*z) term
    // - scale: scales conv filters when do_sigmoid is true
    // - do_sigmoid: apply sigmoid to conv parameters (keeping them in [0,1])
    // - init_vals: initialize backscatter_conv_params to [0.95, 0.8, 0.8]
    BackscatterNet(bool use_residual = false,
                    double scale = 5.0,
                    bool do_sigmoid = false,
                    bool init_vals = false);

    // Forward on depth: depth shape [N,1,H,W], returns backscatter [N,3,H,W]
    torch::Tensor forward(const torch::Tensor& depth);

    // forward_rgb: MSE between estimated atmospheric light and B_inf
    // Expects rgb shape [N,3,H,W]
    torch::Tensor forward_rgb(const torch::Tensor& rgb);

    // Parameters
    torch::Tensor backscatter_conv_params; // [3,1,1,1]
    torch::Tensor residual_conv_params;    // [3,1,1,1] (if use_residual)
    torch::Tensor J_prime;                 // [3,1,1]    (if use_residual)
    torch::Tensor B_inf;                   // [3,1,1]

    // Options
    bool use_residual_;
    double scale_;
    bool do_sigmoid_;

};

class AttenuateNet : public torch::nn::Module {
public:
    // Constructor args mimic Python:
    // - scale: scales conv filters when do_sigmoid is true
    // - do_sigmoid: apply sigmoid to conv parameters
    // - init_vals: initialize attenuation_conv_params to [1.1, 0.95, 0.95]
    AttenuateNet(double scale = 5.0,
                    bool do_sigmoid = false,
                    bool init_vals = true);

    // Forward on depth: depth shape [N,1,H,W], returns attenuation map [N,3,H,W]
    torch::Tensor forward(const torch::Tensor& depth);

    // Parameters
    torch::Tensor attenuation_conv_params; // [3,1,1,1]

    // Options
    double scale_;
    bool do_sigmoid_;

};


} // namespace uw