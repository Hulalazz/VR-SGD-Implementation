#pragma once

#include <lib/Vector.hpp>

#include <cmath>

namespace VRSGD {

template<typename T, bool is_sparse>
inline Vector<T, is_sparse> prox_l2(const Vector<T, is_sparse>& y, T alpha, T lambda) {
    return y / (1 + alpha * lambda);
}

template<typename T>
inline Vector<T, false> prox_l1(const Vector<T, false>& y, T alpha, T lambda) {
    Vector<T, false> res(y.get_feature_num());
    T lambda_ = alpha * lambda;

    for (int i = 0; i < y.get_feature_num(); i++) {
        if (std::abs(y[i]) <= lambda_) {
            res[i] = 0;
        } else if (y[i] > 0){
            res[i] = y[i] - lambda_;
        } else {
            res[i] = y[i] + lambda_;
        }
    }

    return res;
}

template<typename T>
inline Vector<T, true> prox_l1(const Vector<T, true>& y, T alpha, T lambda) {
    Vector<T, true> res(y.get_feature_num());
    T lambda_ = alpha * lambda;

    for (auto& entry : y) {
        if (std::abs(entry.second) <= lambda_) {
            res.set(entry.first, 0);
        } else if (entry.second > 0){
            res.set(entry.first, entry.second - lambda_);
        } else {
            res.set(entry.first, entry.second + lambda_);
        }
    }

    return res;
}

}

