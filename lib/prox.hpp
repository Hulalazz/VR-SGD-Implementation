#pragma once

#include <lib/vector.hpp>

#include <cmath>

namespace VRSGD {

template <typename VectorT, typename T>
inline auto prox_l2(const VectorT& y, T alpha, T lambda) {
    return y / (1 + alpha * lambda);
}

template <typename T>
inline T prox_l1(T y, T lambda) {
    return std::max(0., y - lambda) + std::min(0., y + lambda);
}

template<typename Scalar>
struct CwiseProxL1Op {
    CwiseProxL1Op(const Scalar& lambda) : lambda(lambda) {}
    inline const Scalar operator()(const Scalar& y) const {
        return prox_l1(y, lambda);
    }
    Scalar lambda;
};

template <typename VectorT, typename T>
auto prox_l1(const VectorT& y, T lambda) {
    return y.unaryExpr(CwiseProxL1Op<T>(lambda));
}

template <typename VectorT, typename T>
inline auto prox_l1(const VectorT& y, T alpha, T lambda) {
    return prox_l1(y, alpha * lambda);
}

/*template <typename T, bool is_sparse>
inline Vector<T, is_sparse> prox_identity(const Vector<T, is_sparse>& y, T, T) {
    return y;
}*/

}

