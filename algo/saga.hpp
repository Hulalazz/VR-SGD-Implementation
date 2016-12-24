#pragma once

#include "lib/utils.hpp"

#include <vector>
#include <functional>
#include <random>

namespace VRSGD {

template<typename T, typename U, bool is_sparse, typename ProblemT>
void saga_train(ProblemT problem, double alpha, double lambda, int batch_size, int num_iter, int w_feature_num, int sample_period) {
    typedef LabeledPoint<Vector<T, is_sparse>, U> LabeledPoint_;
    typedef Vector<T, is_sparse> Vector_data;
    typedef decltype(std::declval<ProblemT>().grad_func(DenseVector<T>(), 0)) Vector_grad;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis_num_sample(0, problem.size() - 1);

    DenseVector<T> table_avg(w_feature_num);
    DenseVector<T> w(w_feature_num);
    std::vector<Vector_grad> table;

    DenseVector<T> batch_w_change(w_feature_num);
    std::vector<std::pair<int, Vector_grad>> batch_table;
    DenseVector<T> table_sum_change(w_feature_num);

    int data_num = problem.size();
    for (int i = 0; i < data_num; i++) {
        table.emplace_back(problem.grad_func(w, i));
        table_avg += table[i];
    }
    table_avg /= (double)data_num;

    for (int i = 0; i < num_iter; i++) {
        if (i % sample_period == 0) {
            printf("%d %.15f\n", i, problem.cost_func(w));
        }

        batch_w_change.set_zero();
        batch_table.clear();
        table_sum_change.set_zero();

        for (int j = 0; j < batch_size; j++) {
            int rand_row = dis_num_sample(gen);

            auto grad = problem.grad_func(w, rand_row);
            batch_table.emplace_back(rand_row, grad);

            batch_w_change -= alpha * (grad - (table[rand_row] - table_avg));

            table_sum_change += grad - table[rand_row];
        }

        // TODO: may hurt performance by not using +=?
        w = problem.prox_func(w + batch_w_change / batch_size, alpha, lambda);
        table_avg += table_sum_change / (double)data_num;
        for (auto& batch_item : batch_table) {
            table[batch_item.first] = batch_item.second;
        }
    }

    printf("%d %.15lf\n", num_iter, problem.cost_func(w));
}

}

