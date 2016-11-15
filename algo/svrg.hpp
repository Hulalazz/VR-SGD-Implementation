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

template<typename T, typename U, bool is_sparse>
void svrg_train(std::vector<LabeledPoint<Vector<T, is_sparse>, U>>& data_points, double alpha, double lambda, int batch_size, int num_iter, int num_inner_iter, int w_feature_num, int w_tidle_opt, int sample_period, std::function<T(const DenseVector<T>&, const std::vector<LabeledPoint<Vector<T, is_sparse>, U>>&)> cost_func, std::function<Vector<T, is_sparse>(const DenseVector<T>&, const LabeledPoint<Vector<T, is_sparse>, U>&)> grad_func, std::function<DenseVector<T>(const DenseVector<T>&, T, T)> prox_func) {
    typedef LabeledPoint<Vector<T, is_sparse>, U> LabeledPoint_;
    typedef Vector<T, is_sparse> Vector_data;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, data_points.size() - 1);
    std::uniform_int_distribution<> dis_num_inner_iter(0, num_inner_iter - 1);

    DenseVector<T> w_tidle(w_feature_num);
    DenseVector<T> w(w_feature_num);
    DenseVector<T> mu_tidle(w_feature_num);
    DenseVector<T> batch_w_change(w_feature_num);

    int data_num = data_points.size();
    int num_effective_pass = 0;
    int num_inner_iter_ = num_inner_iter;

    for (int i = 0; i < num_iter; i++) {
        w_tidle = w;
        
        mu_tidle.set_zero();
        for (int i = 0; i < data_num; i++) {
            mu_tidle += grad_func(w_tidle, data_points[i]);
        }
        mu_tidle /= (double)data_num;

        if (w_tidle_opt == 1) {
            num_inner_iter_ = dis_num_inner_iter(gen);
        }

        for (int j = 0; j < num_inner_iter_; j++) {
            if (num_effective_pass % sample_period == 0) {
                printf("%d %.15f\n", num_effective_pass, cost_func(w, data_points));
            }

            batch_w_change.set_zero();
            for (int k = 0; k < batch_size; k++) {
                int rand_row = dis_num_sample(gen);
                LabeledPoint_& data_point = data_points[rand_row];

                auto grad = grad_func(w, data_point);
                auto grad_snapshot = grad_func(w_tidle, data_point);

                batch_w_change -= alpha * (grad - (grad_snapshot - mu_tidle));
            }

            // TODO: may hurt performance by not using +=?
            w = prox_func(w + batch_w_change / batch_size, alpha, lambda);
            //w += batch_w_change / (double)batch_size;

            num_effective_pass++;
        }
    }

    printf("%d %.15lf\n", num_effective_pass, cost_func(w, data_points));
}

}

