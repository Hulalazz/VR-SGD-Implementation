#include <lib/utils.hpp>
#include <algo/svrg.hpp>
#include <problem/lasso_regression.hpp>

#include <cmath>

template <typename VectorDataT>
double calc_L(const std::vector<VRSGD::LabeledPoint<VectorDataT, double>>& data_points) {
    //double max_L = data_points[0].x.squaredNorm() + 1.;
    double max_L = data_points[0].x.squaredNorm();

    for (auto& data_point : data_points) {
        //double L = data_point.x.squaredNorm() + 1.;
        double L = data_point.x.squaredNorm();
        if (L > max_L) {
            max_L = L;
        }
    }

    return max_L * max_L * 4.;
}

int main() {
    typedef VRSGD::VectorXd VectorDataT;
    //typedef VRSGD::SparseVectorXd VectorDataT;

    std::vector<VRSGD::LabeledPoint<VectorDataT, double>> data_points;

    const int feature_num = 54;
    //const double alpha = 0.000961;
    const double alpha = 0.4;
    //const double lambda = 1. / feature_num;
    const double lambda = 1e-4;

    VRSGD::read_libsvm(data_points, "./datasets/covtype.binary", feature_num);
    for (auto& data_point : data_points) {
        data_point.x /= data_point.x.norm();
        if (data_point.y < 1.5) {
            data_point.y = -1;
        } else {
            data_point.y = 1;
        }
    }

    /*const int feature_num = 47236;
    //const double alpha = 0.000961;
    const double alpha = 0.4;
    //const double lambda = 1. / feature_num;
    const double lambda = 1e-4;

    VRSGD::read_libsvm(data_points, "./datasets/rcv1_train.binary", feature_num);
    for (auto& data_point : data_points) {
        data_point.x /= data_point.x.norm();
    }*/

    /*const int feature_num = 14;
    const double alpha = 0.000642;
    const double lambda = 1. / feature_num;

    std::vector<VRSGD::LabeledPoint<VectorDataT, double>> data_points;
    VRSGD::read_libsvm(data_points, "./datasets/housing_scale", feature_num);
    double max_y = 0;
    for (auto& data_point : data_points) {
        if (std::abs(data_point.y) > max_y) {
            max_y = std::abs(data_point.y);
        }
    }
    for (auto& data_point : data_points) {
        data_point.y /= max_y;
    }*/

    VRSGD::LassoRegression<VectorDataT> lasso_regrssion(data_points, lambda);

    //double L = calc_L(data_points);
    //printf("L: %.15lf\n", L);

    VRSGD::svrg_train(
            lasso_regrssion,
            alpha,
            lambda,
            1,
            10,
            2 * data_points.size(),
            //feature_num + 1,
            feature_num,
            0,
            data_points.size());
}

