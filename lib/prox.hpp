#pragma once

#include <lib/vector.hpp>

#include <cmath>

namespace VRSGD {

template<typename T, bool is_sparse>
inline Vector<T, is_sparse> prox_l2(const Vector<T, is_sparse>& y, T alpha, T lambda) {
    return y / (1 + alpha * lambda);
}

template <typename T, bool is_sparse>
Vector<T, is_sparse> prox_l1(const Vector<T, is_sparse>& y, T lambda) {
    Vector<T, is_sparse> res(y.get_feature_num());

    for (auto it = y.begin_feaval(); it != y.end_feaval(); ++it) {
        const auto&& entry = *it;

        if (std::abs(entry.val) <= lambda) {
            res.set(entry.fea, 0);
        } else if (entry.val > 0){
            res.set(entry.fea, entry.val - lambda);
        } else {
            res.set(entry.fea, entry.val + lambda);
        }
    }

    return res;
}

template <typename T>
T prox_l1(T y, T lambda) {
    if (std::abs(y) <= lambda) {
        return 0;
    } else if (y > 0) {
        return y - lambda;
    } else {
        return y + lambda;
    }
}

template <typename T, bool is_sparse>
inline Vector<T, is_sparse> prox_l1(const Vector<T, is_sparse>& y, T alpha, T lambda) {
    return prox_l1(y, alpha * lambda);
}

template <typename T, bool is_sparse>
inline Vector<T, is_sparse> prox_identity(const Vector<T, is_sparse>& y, T, T) {
    return y;
}

}

