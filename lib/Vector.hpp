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
struct FeaValPair {
    FeaValPair(int fea, T val) : fea(fea), val(val) {}
    // TODO: I want fea to be a const, but this will make operator=() unavailable and break the use of STL containers
    int fea;
    T val;
};

template<typename T>
class Vector<T, false> {
public:
    typedef typename std::vector<T>::iterator Iterator;
    typedef typename std::vector<T>::const_iterator ConstIterator;
    typedef typename std::vector<T>::iterator ValueIterator;
    typedef typename std::vector<T>::const_iterator ConstValueIterator;

    class FeaValIterator {
    public:
        FeaValIterator(std::vector<T>& vec) : idx(0), vec(vec) {}
        FeaValIterator(std::vector<T>& vec, int idx) : idx(idx), vec(vec) {}

        FeaValPair<T&> operator*() {
            return FeaValPair<T&>(idx, vec[idx]);
        }

        FeaValIterator& operator++() {
            idx++;
            return *this;
        }

        FeaValIterator& operator--() {
            idx--;
            return *this;
        }

        FeaValIterator operator++(int) {
            FeaValIterator it(*this);
            idx++;
            return it;
        }

        FeaValIterator operator--(int) {
            FeaValIterator it(*this);
            idx--;
            return it;
        }

        bool operator==(const FeaValIterator& b) const {
            return idx == b.idx;
        }

        bool operator!=(const FeaValIterator& b) const {
            return idx != b.idx;
        }

    private:
        int idx;
        std::vector<T>& vec;
    };

    class ConstFeaValIterator {
    public:
        ConstFeaValIterator(const std::vector<T>& vec) : idx(0), vec(vec) {}
        ConstFeaValIterator(const std::vector<T>& vec, int idx) : idx(idx), vec(vec) {}

        FeaValPair<const T&> operator*() const {
            return FeaValPair<const T&>(idx, vec[idx]);
        }

        ConstFeaValIterator& operator++() {
            idx++;
            return *this;
        }

        ConstFeaValIterator& operator--() {
            idx--;
            return *this;
        }

        ConstFeaValIterator operator++(int) {
            ConstFeaValIterator it(*this);
            idx++;
            return it;
        }

        ConstFeaValIterator operator--(int) {
            ConstFeaValIterator it(*this);
            idx--;
            return it;
        }

        bool operator==(const ConstFeaValIterator& b) const {
            return idx == b.idx;
        }

        bool operator!=(const ConstFeaValIterator& b) const {
            return idx != b.idx;
        }

    private:
        int idx;
        const std::vector<T>& vec;
    };

    Vector<T, false>(int feature_num)
        : feature_num(feature_num),
          vec(feature_num) {}

    Vector<T, false>(const SparseVector<T>& b)
        : feature_num(b.get_feature_num()),
          vec(b.get_feature_num()) {
        for (auto entry : b) {
            vec[entry.fea] = entry.val;
        }
    }

    Vector<T, false>(int feature_num, T& val)
        : feature_num(feature_num),
          vec(feature_num, val) {}

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

    inline FeaValIterator begin_feaval() {
        return FeaValIterator(vec, 0);
    }

    inline ConstFeaValIterator begin_feaval() const {
        return ConstFeaValIterator(vec, 0);
    }

    inline FeaValIterator end_feaval() {
        return FeaValIterator(vec, feature_num);
    }

    inline ConstFeaValIterator end_feaval() const {
        return ConstFeaValIterator(vec, feature_num);
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
    typedef typename std::vector<FeaValPair<T>>::iterator Iterator;
    typedef typename std::vector<FeaValPair<T>>::const_iterator ConstIterator;
    typedef typename std::vector<FeaValPair<T>>::iterator FeaValIterator;
    typedef typename std::vector<FeaValPair<T>>::const_iterator ConstFeaValIterator;

    class ValueIterator {
    public:
        ValueIterator(std::vector<FeaValPair<T>>& vec) : idx(0), vec(vec) {}
        ValueIterator(std::vector<FeaValPair<T>>& vec, int idx) : idx(idx), vec(vec) {}

        T& operator*() {
            return vec[idx].val;
        }

        T* operator->() {
            return &(vec[idx].val);
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
        std::vector<FeaValPair<T>>& vec;
    };

    class ConstValueIterator {
    public:
        ConstValueIterator(const std::vector<FeaValPair<T>>& vec) : idx(0), vec(vec) {}
        ConstValueIterator(const std::vector<FeaValPair<T>>& vec, int idx) : idx(idx), vec(vec) {}

        const T& operator*() const {
            return vec[idx].val;
        }

        const T* operator->() const {
            return &(vec[idx].val);
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
        const std::vector<FeaValPair<T>>& vec;
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

    inline FeaValIterator begin_feaval() {
        return vec.begin();
    }

    inline ConstFeaValIterator begin_feaval() const {
        return vec.begin();
    }

    inline FeaValIterator end_feaval() {
        return vec.end();
    }

    inline ConstFeaValIterator end_feaval() const {
        return vec.end();
    }

    // TODO: how to make the following function available, or is it really needed?
    /*inline void set(int idx, const T& val) {
        vec.emplace_back(idx, val);
    }*/

    inline void set(int idx, T val) {
        vec.emplace_back(idx, std::move(val));
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
    std::vector<FeaValPair<T>> vec;
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

