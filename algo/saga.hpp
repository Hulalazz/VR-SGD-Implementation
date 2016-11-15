#pragma once

#include "lib/utils.hpp"

#include <vector>
#include <functional>
#include <random>

namespace VRSGD {

template<typename T, typename U, bool is_sparse>
void saga_train(std::vector<LabeledPoint<Vector<T, is_sparse>, U>>& data_points, double alpha, double lambda, int batch_size, int num_iter, int w_feature_num, int sample_period, std::function<T(const DenseVector<T>&, const std::vector<LabeledPoint<Vector<T, is_sparse>, U>>&)> cost_func, std::function<Vector<T, is_sparse>(const DenseVector<T>&, const LabeledPoint<Vector<T, is_sparse>, U>&)> grad_func, std::function<DenseVector<T>(const DenseVector<T>&, T, T)> prox_func) {
    typedef LabeledPoint<Vector<T, is_sparse>, U> LabeledPoint_;
    typedef Vector<T, is_sparse> Vector_data;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, data_points.size() - 1);

    DenseVector<T> table_avg(w_feature_num);
    std::vector<Vector_data> table;
    DenseVector<T> w(w_feature_num);

    DenseVector<T> batch_w_change(w_feature_num);
    std::vector<std::pair<int, Vector_data>> batch_table;
    DenseVector<T> table_sum_change(w_feature_num);

    int data_num = data_points.size();
    for (int i = 0; i < data_num; i++) {
        table.emplace_back(grad_func(w, data_points[i]));
        table_avg += table[i];
    }
    table_avg /= (double)data_num;

    for (int i = 0; i < num_iter; i++) {
        if (i % sample_period == 0) {
            printf("%d %.15f\n", i, cost_func(w, data_points));
        }

        batch_w_change.set_zero();
        batch_table.clear();
        table_sum_change.set_zero();

        for (int j = 0; j < batch_size; j++) {
            int rand_row = dis_num_sample(gen);
            LabeledPoint_& data_point = data_points[rand_row];

            auto grad = grad_func(w, data_point);
            batch_table.emplace_back(rand_row, grad);

            batch_w_change -= alpha * (grad - (table[rand_row] - table_avg));

            table_sum_change += grad - table[rand_row];
        }

        // TODO: may hurt performance by not using +=?
        w = prox_func(w + batch_w_change / batch_size, alpha, lambda);
        //w += batch_w_change / (double)batch_size;
        table_avg += table_sum_change / (double)data_num;
        for (auto& batch_item : batch_table) {
            table[batch_item.first] = batch_item.second;
        }
    }

    printf("%d %.15lf\n", num_iter, cost_func(w, data_points));
}

}

