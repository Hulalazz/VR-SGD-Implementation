#pragma once

#include "lib/utils.hpp"

#include <Eigen/Dense>

#include <vector>
#include <functional>
#include <random>

namespace VRSGD {

/*
 * @param w_tidle_opt
 * 0: w_tidle = last w
 * 1: w_tidle = one of the w in the last inner iteration
 * // 2: w_tidle = average of w in the last inner iteration
 */

template <typename ProblemT>
void katyusha_train(ProblemT& problem, double sigma, double lambda, double L, int batch_size, int num_iter, int num_inner_iter, int w_feature_num, int y_update_opt, bool is_strongly_convex, int sample_period) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, problem.size() - 1);
    std::uniform_int_distribution<> dis_num_inner_iter(0, num_inner_iter - 1);

    VectorXd x_tidle[2];
    x_tidle[0].resize(w_feature_num);
    x_tidle[1].resize(w_feature_num);
    VectorXd x(w_feature_num);
    VectorXd y(w_feature_num);
    VectorXd z[2];
    z[0].resize(w_feature_num);
    z[1].resize(w_feature_num);
    VectorXd mu_tidle(w_feature_num);
    VectorXd batch_w_change(w_feature_num);

    int data_num = problem.size();
    int num_effective_pass = 0;
    int num_inner_iter_ = num_inner_iter;

    double tau_2 = 0.5;
    double tau_1 = std::min(std::sqrt(num_inner_iter * sigma / 3. / L), 0.5);
    double alpha = 1. / (3. * tau_1 * L);

    double one_plus_alpha_sigma_pow;
    double one_plus_alpha_sigma_pow_sum;

    for (int i = 0; i < num_iter; i++) {
        if (!is_strongly_convex) {
            tau_1 = 2. / (i + 4.);
            alpha = 1. / (3. * tau_1 * L);
        }

        mu_tidle.setZero();
        for (int i = 0; i < data_num; i++) {
            mu_tidle += problem.grad_func(x_tidle[i % 2], i);
        }
        mu_tidle /= data_num;

        x_tidle[(i + 1) % 2].setZero();

        one_plus_alpha_sigma_pow = 1;
        one_plus_alpha_sigma_pow_sum = 0;

        for (int j = 0; j < num_inner_iter_; j++) {
            if (num_effective_pass % sample_period == 0) {
                printf("%d %.15f\n", num_effective_pass, problem.cost_func(x));
            }

            x = tau_1 * z[j % 2] + tau_2 * x_tidle[i % 2] + (1. - tau_1 - tau_2) * y;

            batch_w_change.setZero();
            for (int k = 0; k < batch_size; k++) {
                int rand_row = dis_num_sample(gen);

                batch_w_change -= (  problem.grad_func(x, rand_row)
                                   - problem.grad_func(x_tidle[i % 2], rand_row)
                                   + mu_tidle);
            }
            batch_w_change /= batch_size;

            z[(j + 1) % 2] = problem.prox_func(z[j % 2] + alpha * batch_w_change, alpha, lambda);
            if (y_update_opt == 1) {
                y = problem.prox_func(x + 1. / 3. / L * batch_w_change, 1. / 3. / L, lambda);
            } else {
                y = x + tau_1 * (z[(j + 1) % 2] - z[j % 2]);
            }

            if (is_strongly_convex) {
                x_tidle[(i + 1) % 2] += one_plus_alpha_sigma_pow * y;
                one_plus_alpha_sigma_pow_sum += one_plus_alpha_sigma_pow;
                one_plus_alpha_sigma_pow *= (1 + alpha * sigma);
            } else {
                x_tidle[(i + 1) % 2] += y;
            }

            num_effective_pass++;
        }

        if (is_strongly_convex) {
            x_tidle[(i + 1) % 2] /= one_plus_alpha_sigma_pow_sum;
        } else {
            x_tidle[(i + 1) % 2] /= num_inner_iter;
        }
    }

    printf("%d %.15lf\n", num_effective_pass, problem.cost_func(x));
}

}

