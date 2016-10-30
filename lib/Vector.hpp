#pragma once

#include <type_traits>
#include <vector>
#include <cmath>

namespace VRSGD {

template<typename T, bool is_sparse>
class Vector;

template<typename T>
using SparseVector = Vector<T, true>;

template<typename T>
using DenseVector = Vector<T, false>;

template<typename T>
class Vector<T, false> {
public:
    typedef typename std::vector<T>::iterator Iterator;
    typedef typename std::vector<T>::const_iterator ConstIterator;
    typedef typename std::vector<T>::iterator ValueIterator;
    typedef typename std::vector<T>::const_iterator ConstValueIterator;

    Vector<T, false>(int feature_num)
        : feature_num(feature_num),
          vec(feature_num) {}

    Vector<T, false>(const SparseVector<T>& b)
        : feature_num(b.get_feature_num()),
          vec(b.get_feature_num()) {
        for (auto entry : b) {
            vec[entry.first] = entry.second;
        }

    }

    inline int get_feature_num() const {
        return feature_num;
    }

    inline Iterator begin() {
        return vec.begin();
    }

    inline ConstIterator begin() const {
        return vec.begin();
    }

    inline Iterator end() {
        return vec.end();
    }

    inline ConstIterator end() const {
        return vec.end();
    }

    inline ValueIterator begin_value() {
        return vec.begin();
    }

    inline ConstValueIterator begin_value() const {
        return vec.begin();
    }

    inline ValueIterator end_value() {
        return vec.end();
    }

    inline ConstValueIterator end_value() const {
        return vec.end();
    }

    inline T& operator[](int idx) {
        return vec[idx];
    }

    inline const T& operator[](int idx) const {
        return vec[idx];
    }

    // TODO: how to make the following function available?
    /*inline void set(int idx, const T& val) {
        vec[idx] = val;
    }*/

    inline void set(int idx, T val) {
        vec[idx] = std::move(val);
    }

    DenseVector<T> operator-() const;

    DenseVector<T> operator*(T) const;
    DenseVector<T>& operator*=(T);
    DenseVector<T> scalar_multiple_with_intcpt(T) const;

    DenseVector<T> operator/(T) const;
    DenseVector<T>& operator/=(T);

    DenseVector<T> operator+(const DenseVector<T>&) const;
    DenseVector<T>& operator+=(const DenseVector<T>&);
    DenseVector<T> operator+(const SparseVector<T>&) const;
    DenseVector<T>& operator+=(const SparseVector<T>&);

    DenseVector<T> operator-(const DenseVector<T>&) const;
    DenseVector<T>& operator-=(const DenseVector<T>&);
    DenseVector<T> operator-(const SparseVector<T>&) const;
    DenseVector<T>& operator-=(const SparseVector<T>&);

    T dot(const DenseVector<T>&) const;
    T dot(const SparseVector<T>&) const;

    T dot_with_intcpt(const DenseVector<T>&) const;
    T dot_with_intcpt(const SparseVector<T>&) const;

    template <bool is_sparse>
    inline T euclid_dist(const Vector<T, is_sparse>& b) const {
        DenseVector<T> diff = *this - b;
        return std::sqrt(diff.dot(diff));
    }

private:
    std::vector<T> vec;
    int feature_num;
};

template<typename T>
class Vector<T, true> {
public:
    typedef typename std::vector<std::pair<int, T>>::iterator Iterator;
    typedef typename std::vector<std::pair<int, T>>::const_iterator ConstIterator;

    class ValueIterator {
    public:
        ValueIterator(std::vector<std::pair<int, T>>& vec) : idx(0), vec(vec) {}
        ValueIterator(std::vector<std::pair<int, T>>& vec, int idx) : idx(idx), vec(vec) {}

        T& operator*() {
            return vec[idx].second;
        }

        T* operator->() {
            return &(vec[idx].second);
        }

        ValueIterator& operator++() {
            idx++;
            return *this;
        }

        ValueIterator& operator--() {
            idx--;
            return *this;
        }

        ValueIterator operator++(int) {
            ValueIterator it(*this);
            idx++;
            return it;
        }

        ValueIterator operator--(int) {
            ValueIterator it(*this);
            idx--;
            return it;
        }

        bool operator==(const ValueIterator& b) const {
            return idx == b.idx;
        }

        bool operator!=(const ValueIterator& b) const {
            return idx != b.idx;
        }

    private:
        int idx;
        std::vector<std::pair<int, T>>& vec;
    };

    class ConstValueIterator {
    public:
        ConstValueIterator(const std::vector<std::pair<int, T>>& vec) : idx(0), vec(vec) {}
        ConstValueIterator(const std::vector<std::pair<int, T>>& vec, int idx) : idx(idx), vec(vec) {}

        const T& operator*() const {
            return vec[idx].second;
        }

        const T* operator->() const {
            return &(vec[idx].second);
        }

        ConstValueIterator& operator++() {
            idx++;
            return *this;
        }

        ConstValueIterator& operator--() {
            idx--;
            return *this;
        }

        ConstValueIterator operator++(int) {
            ConstValueIterator it(*this);
            idx++;
            return it;
        }

        ValueIterator operator--(int) {
            ConstValueIterator it(*this);
            idx--;
            return it;
        }

        bool operator==(const ConstValueIterator& b) const {
            return idx == b.idx;
        }

        bool operator!=(const ConstValueIterator& b) const {
            return idx != b.idx;
        }

    private:
        int idx;
        const std::vector<std::pair<int, T>>& vec;
    };

    Vector<T, true>(int feature_num)
        : feature_num(feature_num) {}

    inline int get_feature_num() const {
        return feature_num;
    }

    inline Iterator begin() {
        return vec.begin();
    }

    inline ConstIterator begin() const {
        return vec.begin();
    }

    inline Iterator end() {
        return vec.end();
    }

    inline ConstIterator end() const {
        return vec.end();
    }

    inline ValueIterator begin_value() {
        return ValueIterator(vec, 0);
    }

    inline ConstValueIterator begin_value() const {
        return ConstValueIterator(vec, 0);
    }

    inline ValueIterator end_value() {
        return ValueIterator(vec, vec.size());
    }

    inline ConstValueIterator end_value() const {
        return ConstValueIterator(vec, vec.size());
    }

    // TODO: how to make the following function available?
    /*inline void set(int idx, const T& val) {
        vec.push_back(std::move(std::make_pair(idx, val)));
    }*/

    inline void set(int idx, T val) {
        vec.push_back(std::move(std::make_pair(idx, std::move(val))));
    }

    SparseVector<T> operator-() const;

    SparseVector<T> operator*(T) const;
    SparseVector<T>& operator*=(T);
    SparseVector<T> scalar_multiple_with_intcpt(T) const;

    SparseVector<T> operator/(T) const;
    SparseVector<T>& operator/=(T);

    DenseVector<T> operator+(const SparseVector<T>& b) const;

    inline DenseVector<T> operator+(const DenseVector<T>& b) const {
        return b + *this;
    }

    DenseVector<T> operator-(const SparseVector<T>& b) const;
    DenseVector<T> operator-(const DenseVector<T>&) const;

    inline T dot(const DenseVector<T>& b) const {
        return b.dot(*this);
    }

    T dot_with_intcpt(const DenseVector<T>& b) const;

    template <bool is_sparse>
    inline T euclid_dist(const DenseVector<T>& b) const {
        DenseVector<T> diff = b - *this;
        return std::sqrt(diff.dot(diff));
    }

private:
    std::vector<std::pair<int, T>> vec;
    int feature_num;
};

template<typename T>
inline DenseVector<T> operator*(T c, const DenseVector<T>& a) {
    return a * c;
}

template<typename T>
inline SparseVector<T> operator*(T c, const SparseVector<T>& a) {
    return a * c;
}

template<typename T, typename U, bool is_sparse>
struct DataPoint {
    DataPoint(int feature_num) : x(feature_num) {}
    Vector<T, is_sparse> x;
    U y;
};

#include "Vector.tpp"

}

