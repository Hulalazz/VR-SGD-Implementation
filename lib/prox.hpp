#pragma once

#include <lib/vector.hpp>

#include <cmath>

namespace VRSGD {

template<typename T, bool is_sparse>
inline Vector<T, is_sparse> prox_l2(const Vector<T, is_sparse>& y, T alpha, T lambda) {
    return y / (1 + alpha * lambda);
}

template <typename T, bool is_sparse>
Vector<T, is_sparse> prox_l1(const Vector<T, is_sparse>& y, T alpha, T lambda) {
    Vector<T, is_sparse> res(y.get_feature_num());
    T lambda_ = alpha * lambda;

    for (auto it = y.begin_feaval(); it != y.end_feaval(); ++it) {
        const auto&& entry = *it;

        if (std::abs(entry.val) <= lambda_) {
            res.set(entry.fea, 0);
        } else if (entry.val > 0){
            res.set(entry.fea, entry.val - lambda_);
        } else {
            res.set(entry.fea, entry.val + lambda_);
        }
    }

    return res;
}

}

