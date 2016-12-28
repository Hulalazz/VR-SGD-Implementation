#pragma once

#include "lib/utils.hpp"

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
void svrg_lazy_update_train(ProblemT& problem, double alpha, double lambda, int batch_size, int num_iter, int num_inner_iter, int w_feature_num, int w_tidle_opt, int sample_period) {
    typedef typename ProblemT::VectorGradT VectorGradT;
    typedef typename std::remove_reference<decltype(problem.data(0))>::type VectorDataT;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, problem.size() - 1);
    std::uniform_int_distribution<> dis_num_inner_iter(0, num_inner_iter - 1);

    VectorXd w_tidle(w_feature_num);
    VectorXd w(w_feature_num);
    VectorXd mu_tidle(w_feature_num);
    ChangeStore<double> batch_w_change(batch_size * w_feature_num);

    std::vector<int> last_update(w_feature_num);

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

        std::fill(last_update.begin(), last_update.end(), 0);

        if (w_tidle_opt == 1) {
            num_inner_iter_ = dis_num_inner_iter(gen);
        }

        for (int j = 0; j < num_inner_iter_; j++) {
            if (num_effective_pass % sample_period == 0) {
                for (int k = 0; k < w_feature_num; ++k) {
                    w[k] = problem.prox_func(w[k], mu_tidle[k], alpha, lambda, j - last_update[k]);
                }
                printf("%d %.15f\n", num_effective_pass, problem.cost_func(w));
            }

            batch_w_change.clear();
            batch_w_change.reserve(batch_size * w_feature_num);
            for (int k = 0; k < batch_size; k++) {
                int rand_row = dis_num_sample(gen);

                for (typename VectorDataT::InnerIterator it(problem.data(rand_row)); it; ++it) {
                    w[it.index()] = problem.prox_func(w[it.index()], mu_tidle[it.index()], alpha, lambda, j - last_update[it.index()]);
                    last_update[it.index()] = j;
                }

                batch_w_change -= (  problem.grad_func(w, rand_row)
                                   - problem.grad_func(w_tidle, rand_row)).eval();
            }

            batch_w_change *= alpha / batch_size;
            w += batch_w_change;

            num_effective_pass++;
        }

        for (int j = 0; j < w_feature_num; ++j) {
            w[j] = problem.prox_func(w[j], mu_tidle[j], alpha, lambda, num_inner_iter_ - last_update[j]);
        }
    }

    printf("%d %.15lf\n", num_effective_pass, problem.cost_func(w));
}

}

