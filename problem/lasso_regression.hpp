#include <lib/vector.hpp>
#include <lib/prox.hpp>

namespace VRSGD {

template <bool is_sparse>
class LassoRegression {
 public:
    LassoRegression(const std::vector<VRSGD::LabeledPoint<VRSGD::Vector<double, is_sparse>, double>>& data_points, double lambda)
        : data_points(data_points),
          lambda(lambda) {
        data_num = data_points.size();
    }

    double cost_func(const VRSGD::DenseVector<double>& w) {
        double res = 0;
        for (auto& data_point : data_points) {
            //double tmp = w.dot_with_intcpt(data_point.x) - data_point.y;
            double tmp = w.dot(data_point.x) - data_point.y;
            res += tmp * tmp / (2 * data_points.size());
        }

        res += lambda * w.norm();

        return res;
    }

    VRSGD::Vector<double, is_sparse> grad_func(const VRSGD::DenseVector<double>& w) {
        Vector<double, false> res;

        for (const auto& data_point : data_points) {
            //res += data_point.x.scalar_multiple_with_intcpt(w.dot_with_intcpt(data_point.x) - data_point.y);
            res += data_point.x * (w.dot(data_point.x) - data_point.y) / data_num;
        }

        return res;
    }

    inline VRSGD::Vector<double, is_sparse> grad_func(const VRSGD::DenseVector<double>& w, int idx) {
        auto& data_point = data_points[idx];
        //return data_point.x.scalar_multiple_with_intcpt(w.dot_with_intcpt(data_point.x) - data_point.y);
        return data_point.x * (w.dot(data_point.x) - data_point.y);
    }

    inline DenseVector<double> prox_func(DenseVector<double> y, double alpha, double lambda) {
        return prox_l1(y, alpha, lambda);
    }

    int size() {
        return data_num;
    }

 protected:
    const std::vector<LabeledPoint<VRSGD::Vector<double, is_sparse>, double>>& data_points;
    int data_num;
    double lambda;
};

}

