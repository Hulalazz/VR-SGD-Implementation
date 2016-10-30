#include <cassert>

template<typename T>
DenseVector<T> DenseVector<T>::operator-() const {
    DenseVector<T> res(*this);

    for (int i = 0; i < feature_num; i++) {
        res[i] *= -1;
    }

    return res;
}

template<typename T>
DenseVector<T> DenseVector<T>::operator*(T c) const {
    DenseVector<T> res(feature_num);

    for (int i = 0; i < feature_num; i++) {
        res[i] = c * vec[i];
    }

    return res;
}

template<typename T>
DenseVector<T> DenseVector<T>::scalar_multiple_with_intcpt(T c) const {
    DenseVector<T> res(feature_num + 1);

    for (int i = 0; i < feature_num; i++) {
        res[i] = c * vec[i];
    }
    res[feature_num] = c;

    return res;
}

template<typename T>
DenseVector<T>& DenseVector<T>::operator*=(T c) {
    for (int i = 0; i < feature_num; i++) {
        vec[i] *= c;
    }

    return *this;
}

template<typename T>
DenseVector<T> DenseVector<T>::operator/(T c) const {
    DenseVector<T> res(feature_num);

    for (int i = 0; i < feature_num; i++) {
        res[i] = vec[i] / c;
    }

    return res;
}

template<typename T>
DenseVector<T>& DenseVector<T>::operator/=(T c) {
    for (int i = 0; i < feature_num; i++) {
        vec[i] /= c;
    }

    return *this;
}

template<typename T>
DenseVector<T> DenseVector<T>::operator+(const DenseVector<T>& b) const {
    assert(feature_num == b.feature_num);

    DenseVector<T> res(feature_num);

    for (int i = 0; i < feature_num; i++) {
        res[i] = vec[i] + b.vec[i];
    }

    return res;
}

template<typename T>
DenseVector<T>& DenseVector<T>::operator+=(const DenseVector<T>& b) {
    assert(feature_num == b.feature_num);

    for (int i = 0; i < feature_num; i++) {
        vec[i] += b.vec[i];
    }

    return *this;
}

template<typename T>
DenseVector<T> DenseVector<T>::operator+(const SparseVector<T>& b) const {
    DenseVector<T> res(*this);
    res += b;
    return res;
}

template<typename T>
DenseVector<T>& DenseVector<T>::operator+=(const SparseVector<T>& b) {
    assert(feature_num == b.get_feature_num());

    for (const std::pair<int, T>& entry : b) {
        vec[entry.first] += entry.second;
    }

    return *this;
}

template<typename T>
DenseVector<T> DenseVector<T>::operator-(const DenseVector<T>& b) const {
    assert(feature_num == b.feature_num);

    DenseVector<T> res(feature_num);

    for (int i = 0; i < feature_num; i++) {
        res[i] = vec[i] - b.vec[i];
    }

    return res;
}

template<typename T>
DenseVector<T>& DenseVector<T>::operator-=(const DenseVector<T>& b) {
    assert(feature_num == b.feature_num);

    for (int i = 0; i < feature_num; i++) {
        vec[i] -= b.vec[i];
    }

    return *this;
}

template<typename T>
DenseVector<T> DenseVector<T>::operator-(const SparseVector<T>& b) const {
    DenseVector<T> res(*this);
    res -= b;
    return res;
}

template<typename T>
DenseVector<T>& DenseVector<T>::operator-=(const SparseVector<T>& b) {
    assert(feature_num == b.get_feature_num());

    for (const std::pair<int, T>& entry : b.vec) {
        vec[entry.first] -= entry.second;
    }

    return *this;
}

template<typename T>
T DenseVector<T>::dot(const DenseVector<T>& b) const {
    assert(feature_num == b.feature_num);

    T res = 0;

    for (int i = 0; i < feature_num; i++) {
        res += vec[i] * b.vec[i];
    }

    return res;
}

template<typename T>
T DenseVector<T>::dot(const SparseVector<T>& b) const {
    assert(feature_num == b.get_feature_num());

    T res = 0;

    for (const std::pair<int, T>& entry : b) {
        res += vec[entry.first] * entry.second;
    }

    return res;
}

template<typename T>
T DenseVector<T>::dot_with_intcpt(const DenseVector<T>& b) const {
    assert(feature_num == b.feature_num + 1);

    T res = 0;

    for (int i = 0; i < feature_num - 1; i++) {
        res += vec[i] * b.vec[i];
    }
    res += vec[feature_num - 1];

    return res;
}

template<typename T>
T DenseVector<T>::dot_with_intcpt(const SparseVector<T>& b) const {
    assert(feature_num == b.get_feature_num() + 1);

    T res = 0;

    for (const std::pair<int, T>& entry : b) {
        res += vec[entry.first] * entry.second;
    }
    res += vec[feature_num - 1];

    return res;
}

template<typename T>
SparseVector<T> SparseVector<T>::operator-() const {
    SparseVector<T> res(*this);

    for (std::pair<int, T>& entry : res) {
        entry.second *= -1;
    }

    return res;
}

template<typename T>
SparseVector<T> SparseVector<T>::operator*(T c) const {
    SparseVector<T> res(*this);

    for (std::pair<int, T>& entry : res) {
        entry.second *= c;
    }

    return res;
}

template<typename T>
SparseVector<T> SparseVector<T>::scalar_multiple_with_intcpt(T c) const {
    SparseVector<T> res(feature_num + 1);

    for (const auto& entry : vec) {
        res.vec.emplace_back(entry.first, c * entry.second);
    }
    res.vec.emplace_back(feature_num, c);

    return res;
}

template<typename T>
SparseVector<T>& SparseVector<T>::operator*=(T c) {
    for (std::pair<int, T>& entry : vec) {
        entry.second *= c;
    }

    return *this;
}

template<typename T>
SparseVector<T> SparseVector<T>::operator/(T c) const {
    SparseVector<T> res(*this);

    for (std::pair<int, T>& entry : res) {
        entry.second /= c;
    }

    return res;
}

template<typename T>
SparseVector<T>& SparseVector<T>::operator/=(T c) {
    for (std::pair<int, T>& entry : vec) {
        entry.second /= c;
    }

    return *this;
}

template<typename T>
DenseVector<T> SparseVector<T>::operator+(const SparseVector<T>& b) const {
    assert(feature_num == b.feature_num);

    DenseVector<T> res(*this);

    for (const auto& entry : b.vec) {
        res[entry.first] += entry.second;
    }

    return res;
}

template<typename T>
DenseVector<T> SparseVector<T>::operator-(const DenseVector<T>& b) const {
    assert(feature_num == b.get_feature_num());

    DenseVector<T> res(std::move(-b));

    for (const std::pair<int, T>& entry : vec) {
        res[entry.first] += entry.second;
    }

    return res;
}

template<typename T>
DenseVector<T> SparseVector<T>::operator-(const SparseVector<T>& b) const {
    assert(feature_num == b.feature_num);

    DenseVector<T> res(*this);

    for (const auto& entry : b.vec) {
        res[entry.first] -= entry.second;
    }

    return res;
}

template<typename T>
T SparseVector<T>::dot_with_intcpt(const DenseVector<T>& b) const {
    assert(feature_num == b.get_feature_num() + 1);

    T res = 0;

    for (const auto& entry : vec) {
        if (entry.first < feature_num - 1) {
            res += entry.second * vec[entry.first];
        } else {
            res += entry.second;
        }
    }

    return res;
}

