#include <lib/vector.hpp>
#include <lib/prox.hpp>

namespace VRSGD {

template <typename VectorDataT>
class LassoRegression {
 public:
    typedef VectorDataT VectorGradT;

    LassoRegression(const std::vector<VRSGD::LabeledPoint<VectorDataT, double>>& data_points, double lambda)
        : data_points(data_points),
          lambda(lambda) {
        data_num = data_points.size();
    }

    double cost_func(const VectorXd& w) const {
        double res = 0;
        for (auto& data_point : data_points) {
            //double tmp = w.dot_with_intcpt(data_point.x) - data_point.y;
            double tmp = w.dot(data_point.x) - data_point.y;
            res += tmp * tmp / (2 * data_points.size());
        }

        res += lambda * w.norm();

        return res;
    }

    VectorDataT grad_func(const VectorXd& w) const {
        VectorXd res;

        for (const auto& data_point : data_points) {
            //res += data_point.x.scalar_multiple_with_intcpt(w.dot_with_intcpt(data_point.x) - data_point.y);
            res += data_point.x * (w.dot(data_point.x) - data_point.y) / data_num;
        }

        return res;
    }

    template <typename Derived>
    inline auto grad_func(const MatrixBase<Derived>& w, int idx) const {
        auto& data_point = data_points[idx];
        //return data_point.x.scalar_multiple_with_intcpt(w.dot_with_intcpt(data_point.x) - data_point.y);
        return data_point.x * (w.dot(data_point.x) - data_point.y);
    }

    template <typename Derived>
    inline auto prox_func(const MatrixBase<Derived>& y, double alpha, double lambda) const {
        return prox_l1(y, alpha, lambda);
    }

    int size() const {
        return data_num;
    }

 protected:
    const std::vector<LabeledPoint<VectorDataT, double>>& data_points;
    int data_num;
    double lambda;
};

}

