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

template<typename T, typename U, bool is_sparse, typename ProblemT>
void svrg_train(ProblemT& problem, double alpha, double lambda, int batch_size, int num_iter, int num_inner_iter, int w_feature_num, int w_tidle_opt, int sample_period) {
    typedef LabeledPoint<Vector<T, is_sparse>, U> LabeledPoint_;
    typedef Vector<T, is_sparse> Vector_data;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, problem.size() - 1);
    std::uniform_int_distribution<> dis_num_inner_iter(0, num_inner_iter - 1);

    DenseVector<T> w_tidle(w_feature_num);
    DenseVector<T> w(w_feature_num);
    DenseVector<T> mu_tidle(w_feature_num);
    DenseVector<T> batch_w_change(w_feature_num);

    int data_num = problem.size();
    int num_effective_pass = 0;
    int num_inner_iter_ = num_inner_iter;

    for (int i = 0; i < num_iter; i++) {
        w_tidle = w;
        
        mu_tidle.set_zero();
        for (int i = 0; i < data_num; i++) {
            mu_tidle += problem.grad_func(w_tidle, i) / data_num;
        }

        if (w_tidle_opt == 1) {
            num_inner_iter_ = dis_num_inner_iter(gen);
        }

        for (int j = 0; j < num_inner_iter_; j++) {
            if (num_effective_pass % sample_period == 0) {
                printf("%d %.15f\n", num_effective_pass, problem.cost_func(w));
            }

            batch_w_change.set_zero();
            for (int k = 0; k < batch_size; k++) {
                int rand_row = dis_num_sample(gen);

                auto grad = problem.grad_func(w, rand_row);
                auto grad_snapshot = problem.grad_func(w_tidle, rand_row);

                batch_w_change -= alpha * (grad - (grad_snapshot - mu_tidle));
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

