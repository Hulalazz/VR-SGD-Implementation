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
void svrg_train(ProblemT& problem, double alpha, double lambda, int batch_size, int num_iter, int num_inner_iter, int w_feature_num, int w_tidle_opt, int sample_period) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, problem.size() - 1);
    std::uniform_int_distribution<> dis_num_inner_iter(0, num_inner_iter - 1);

    VectorXd w_tidle(w_feature_num);
    VectorXd w(w_feature_num);
    VectorXd mu_tidle(w_feature_num);
    VectorXd batch_w_change(w_feature_num);

    int data_num = problem.size();
    int num_effective_pass = 0;
    int num_inner_iter_ = num_inner_iter;

    for (int i = 0; i < num_iter; i++) {
        w_tidle = w;
        
        mu_tidle.setZero();
        for (int i = 0; i < data_num; i++) {
            mu_tidle += problem.grad_func(w_tidle, i);
        }
        mu_tidle /= data_num;

        if (w_tidle_opt == 1) {
            num_inner_iter_ = dis_num_inner_iter(gen);
        }

        for (int j = 0; j < num_inner_iter_; j++) {
            if (num_effective_pass % sample_period == 0) {
                printf("%d %.15f\n", num_effective_pass, problem.cost_func(w));
            }

            batch_w_change.setZero();
            for (int k = 0; k < batch_size; k++) {
                int rand_row = dis_num_sample(gen);

                batch_w_change -= alpha * (  problem.grad_func(w, rand_row)
                                           - problem.grad_func(w_tidle, rand_row)
                                           + mu_tidle);
            }

            // TODO: may hurt performance by not using +=?
            w = problem.prox_func(w + batch_w_change / batch_size, alpha, lambda);
            //w += batch_w_change / (double)batch_size;

            num_effective_pass++;
        }
    }

    printf("%d %.15lf\n", num_effective_pass, problem.cost_func(w));
}

}

