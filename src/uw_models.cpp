#include "include/uw_models.h"
#include <iostream>

namespace uw {

BackscatterNet::BackscatterNet(
    bool use_residual,
    double scale,
    bool do_sigmoid,
    bool init_vals
) : use_residual_(use_residual), scale_(scale), do_sigmoid_(do_sigmoid)
{
    // backscatter_conv_params
    if (init_vals) {
        auto init_tensor = torch::tensor({0.95, 0.8, 0.8}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
                               .view({3, 1, 1, 1});
        this->backscatter_conv_params = register_parameter("backscatter_conv_params", init_tensor);
    } else {
        auto rand_tensor = torch::rand({3, 1, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        this->backscatter_conv_params = register_parameter("backscatter_conv_params", rand_tensor);
    }

    // residual-related params
    if (use_residual) {
        auto res_conv = torch::rand({3, 1, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        this->residual_conv_params = register_parameter("residual_conv_params", res_conv);

        auto Jp = torch::rand({3, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        this->J_prime = register_parameter("J_prime", Jp);

    }

    auto Binf = torch::rand({3, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    this->B_inf = register_parameter("B_inf", Binf);

}

torch::Tensor BackscatterNet::forward(const torch::Tensor& depth) {
    torch::Tensor b_conv_params = this->do_sigmoid_ ? 
        torch::sigmoid(this->backscatter_conv_params) * scale_ : 
        this->backscatter_conv_params;
        
    torch::Tensor beta_b_conv = torch::nn::functional::conv2d(
        depth, b_conv_params,
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0)
    );

    torch::Tensor beta_b_conv_activated = this->do_sigmoid_ ? 
        torch::relu(beta_b_conv) : 
        torch::clamp_min(beta_b_conv, 0.0);

    auto backscatter = torch::sigmoid(this->B_inf) * (1.0 - torch::exp(-beta_b_conv_activated));

    if(this->use_residual_){
        torch::Tensor r_conv_params = this->do_sigmoid_ ? 
            torch::sigmoid(this->residual_conv_params) * scale_ : 
            this->residual_conv_params;

        torch::Tensor beta_r_conv = torch::nn::functional::conv2d(
            depth, r_conv_params,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0)
        );

        torch::Tensor beta_r_conv_activated = this->do_sigmoid_ ? 
            torch::relu(beta_r_conv) : 
            torch::clamp_min(beta_r_conv, 0.0);

        auto residual = torch::sigmoid(this->J_prime) * torch::exp(-beta_r_conv_activated);
        backscatter += residual;
    }

    return backscatter;
}

AttenuateNet::AttenuateNet(
    double scale,
    bool do_sigmoid,
    bool init_vals
) : scale_(scale), do_sigmoid_(do_sigmoid)
{
    // attenuation_conv_params
    if (init_vals) {
        auto init_tensor = torch::tensor({1.1, 0.95, 0.95}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
                               .view({3, 1, 1, 1});
        this->attenuation_conv_params = register_parameter("attenuation_conv_params", init_tensor);
    } else {
        auto rand_tensor = torch::rand({3, 1, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        this->attenuation_conv_params = register_parameter("attenuation_conv_params", rand_tensor);
    }
}

torch::Tensor AttenuateNet::forward(const torch::Tensor& depth) {
    torch::Tensor a_conv_params = this->do_sigmoid_ ? 
        torch::sigmoid(this->attenuation_conv_params) * scale_ : 
        this->attenuation_conv_params;
        
    torch::Tensor beta_d_conv = torch::nn::functional::conv2d(
        depth, a_conv_params,
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(0)
    );

    torch::Tensor beta_d_conv_activated = this->do_sigmoid_ ? 
        torch::relu(beta_d_conv) : 
        torch::clamp_min(beta_d_conv, 0.0);

    auto attenuation = torch::exp(-beta_d_conv_activated);

    return attenuation;
}

torch::Tensor normalize_depth(torch::Tensor& depth_raw, torch::Tensor& opc, float normalize_depth, bool norm_depth_max){
    auto depth = depth_raw / opc;

    if (torch::any(torch::logical_or(torch::isnan(depth), torch::isinf(depth))).item<bool>()) {
        auto valid_depth_vals = depth.masked_select(torch::logical_not(torch::logical_or(torch::isnan(depth), torch::isinf(depth))));
        float not_nan_max;

        if (valid_depth_vals.size(0) == 0) {
            std::cout << "[uw::normalize_depth] everything is nan" << std::endl;
            not_nan_max = 100.f;
        } else {
            not_nan_max = valid_depth_vals.max().item<float>();
        }

        depth = torch::nan_to_num(depth, not_nan_max, not_nan_max);
    }

    depth = depth / normalize_depth;

    if (norm_depth_max) {
        if (depth.min().item<float>() != depth.max().item<float>()) {
            depth = (depth - depth.min()) / (depth.max() - depth.min());
        } else {
            depth = depth / depth.max();
        }
    }

    return depth;
}

} // namespace uw