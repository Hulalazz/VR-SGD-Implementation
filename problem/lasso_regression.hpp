#include <lib/vector.hpp>
#include <lib/prox.hpp>

#include <algorithm>

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

    inline double prox_func(double w, const double& mu_tidle, double alpha, double lambda, int tau) const {
        w -= alpha * tau * mu_tidle;

        int positive;
        if (w < 0) {
            positive = -1;
            w *= -1;
        } else {
            positive = 1;
        }

        if (w > 0) {
            int x = std::min((int)(w / lambda), tau);
            if (x >= tau) {
                return positive * (w - tau * lambda);
            } else {
                double tmp1 = w - x * lambda;
                double tmp2 = tmp1 - lambda;
                if (std::abs(tmp1) < std::abs(tmp2)) {
                    return positive * tmp1;
                } else {
                    return positive * tmp2;
                }
            }
        }
    }

    inline const VectorDataT& data(int idx) const {
        return data_points[idx].x;
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

