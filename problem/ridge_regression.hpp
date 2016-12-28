#include <lib/vector.hpp>
#include <lib/prox.hpp>

namespace VRSGD {

template <typename VectorDataT>
class RidgeRegression {
 public:
    typedef VectorXd VectorGradT;

    RidgeRegression(const std::vector<VRSGD::LabeledPoint<VectorDataT, double>>& data_points, double lambda)
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

        res += lambda / 2. * w.squaredNorm();

        return res;
    }

    VectorXd grad_func(const VectorXd& w) const {
        VectorXd res;

        for (const auto& data_point : data_points) {
            //res += data_point.x.scalar_multiple_with_intcpt(w.dot_with_intcpt(data_point.x) - data_point.y) / data_num;
            res += data_point.x * (w.dot(data_point.x) - data_point.y) / data_num;
        }
        
        res += lambda * w;

        return res;
    }

    template <typename Derived>
    inline auto grad_func(const MatrixBase<Derived>& w, int idx) const {
        auto& data_point = data_points[idx];
        //return data_point.x.scalar_multiple_with_intcpt(w.dot_with_intcpt(data_point.x) - data_point.y);
        return data_point.x * (w.dot(data_point.x) - data_point.y) + lambda * w;
    }

    const VectorXd& prox_func(const VectorXd& y, double, double) const {
        return y;
    }

    int size() const {
        return data_num;
    }

 protected:
    const std::vector<LabeledPoint<VectorDataT, double>>& data_points;
    int data_num;
    double lambda;
};

template <typename VectorDataT>
class RidgeRegressionProx {
 public:
    typedef VectorDataT VectorGradT;

    RidgeRegressionProx(const std::vector<VRSGD::LabeledPoint<VectorDataT, double>>& data_points, double lambda)
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

        res += lambda / 2. * w.squaredNorm();

        return res;
    }

    VectorXd grad_func(const VectorXd& w) const {
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
        return prox_l2(y, alpha, lambda);
    }

    inline double prox_func(double w, const double& mu_tidle, double alpha, double lambda, int tau) const {
        double beta = 1. / (1. + alpha * lambda);
        double beta_pow_tau = std::pow(beta, tau);
        return beta_pow_tau * w - alpha * beta / (1 - beta) * (1 - beta_pow_tau) * mu_tidle;
    }

    inline const VectorDataT& data(int idx) const {
        return data_points[idx].x;
    }

    int size() {
        return data_num;
    }

 private:
    const std::vector<LabeledPoint<VectorDataT, double>>& data_points;
    int data_num;
    double lambda;
};

}
