#include <lib/vector.hpp>
#include <lib/utils.hpp>
#include <lib/prox.hpp>
#include <algo/saga.hpp>

#include <cmath>

template<bool is_sparse>
double logistic_regression_func(const VRSGD::DenseVector<double>& w, const VRSGD::Vector<double, is_sparse>& x) {
    return 1./(1.+std::exp(-(w.dot_with_intcpt(x))));
}

template<bool is_sparse>
double logistic_regression_cost_func(const VRSGD::DenseVector<double>& w, const std::vector<VRSGD::LabeledPoint<VRSGD::Vector<double, is_sparse>, double>>& data_points) {
    double res = 0;
    for (auto& data_point : data_points) {
        double tmp = logistic_regression_func(w, data_point.x) - data_point.y;
        res += tmp * tmp;
    }
    return res / data_points.size();
}

template<bool is_sparse>
VRSGD::Vector<double, is_sparse> logistic_regression_grad(const VRSGD::DenseVector<double>& w, const VRSGD::LabeledPoint<VRSGD::Vector<double, is_sparse>, double>& data_point) {
    return data_point.x.scalar_multiple_with_intcpt(logistic_regression_func(w, data_point.x) - data_point.y);
}

template<bool is_sparse>
double calc_L(const std::vector<VRSGD::LabeledPoint<VRSGD::Vector<double, is_sparse>, double>>& data_points) {
    double max_L = std::sqrt(data_points[0].x.dot(data_points[0].x) + 1);

    for (auto& data_point : data_points) {
        double L = std::sqrt(data_point.x.dot(data_point.x) + 1);
        if (L > max_L) {
            max_L = L;
        }
    }

    return max_L;
}

int main() {
    const bool is_sparse = true;

    const int feature_num = 123;
    const double alpha = 0.085;
    const double lambda = 0.001/123.;

    std::vector<VRSGD::LabeledPoint<VRSGD::Vector<double, is_sparse>, double>> data_points;
    VRSGD::read_libsvm(data_points, "./datasets/a9a", 123);

    for (auto& entry : data_points) {
        if (entry.y < 0) {
            entry.y = 0;
        }
        //entry.x /= std::sqrt(entry.x.dot(entry.x));
    }

    printf("L: %.15lf\n", calc_L(data_points));

    VRSGD::saga_train<double, double, is_sparse>(
            data_points,
            alpha,
            lambda,
            1,
            1000000,
            feature_num + 1,
            100,
            logistic_regression_cost_func<is_sparse>,
            logistic_regression_grad<is_sparse>,
            VRSGD::prox_l1<double, false>);
}

