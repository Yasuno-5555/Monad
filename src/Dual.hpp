#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// Simple Dual Number for Forward-Mode AD
// Represents x + eps * dx where eps^2 = 0
template <typename T>
struct Dual {
    T val; // Value
    T der; // Derivative

    HOST_DEVICE Dual(T v = 0.0, T d = 0.0) : val(v), der(d) {}

    // Arithmetic
    HOST_DEVICE Dual operator+(const Dual& other) const { return Dual(val + other.val, der + other.der); }
    HOST_DEVICE Dual operator-(const Dual& other) const { return Dual(val - other.val, der - other.der); }
    HOST_DEVICE Dual operator*(const Dual& other) const { return Dual(val * other.val, val * other.der + der * other.val); }
    HOST_DEVICE Dual operator/(const Dual& other) const { 
        return Dual(val / other.val, (der * other.val - val * other.der) / (other.val * other.val)); 
    }

    // Scalar arithmetic
    HOST_DEVICE Dual operator+(T scalar) const { return Dual(val + scalar, der); }
    HOST_DEVICE Dual operator-(T scalar) const { return Dual(val - scalar, der); }
    HOST_DEVICE Dual operator*(T scalar) const { return Dual(val * scalar, der * scalar); }
    HOST_DEVICE Dual operator/(T scalar) const { return Dual(val / scalar, der / scalar); }
    
    // Friends for scalar on LHS
    friend HOST_DEVICE Dual operator+(T scalar, const Dual& d) { return Dual(scalar + d.val, d.der); }
    friend HOST_DEVICE Dual operator-(T scalar, const Dual& d) { return Dual(scalar - d.val, -d.der); }
    friend HOST_DEVICE Dual operator*(T scalar, const Dual& d) { return Dual(scalar * d.val, scalar * d.der); }
    friend HOST_DEVICE Dual operator/(T scalar, const Dual& d) { return Dual(scalar / d.val, -scalar * d.der / (d.val * d.val)); }

    // Math Functions
    // Use fully qualified calls or specialized helpers to support both host and device
    friend HOST_DEVICE Dual pow(const Dual& base, T exp) {
        // d/dx (x^n) = n * x^(n-1) * x'
        double e = static_cast<double>(exp);
        #ifdef __CUDA_ARCH__
        return Dual(::pow(base.val, e), e * ::pow(base.val, e - 1.0) * base.der);
        #else
        return Dual(std::pow(base.val, e), e * std::pow(base.val, e - 1.0) * base.der);
        #endif
    }
    
    // For general pow(Dual, Dual)
    friend HOST_DEVICE Dual pow(const Dual& base, const Dual& exp) {
        // y = u^v => ln y = v ln u => y'/y = v' ln u + v u'/u
        #ifdef __CUDA_ARCH__
        T res_val = ::pow(base.val, exp.val);
        return Dual(res_val, res_val * (exp.der * ::log(base.val) + exp.val * base.der / base.val));
        #else
        T res_val = std::pow(base.val, exp.val);
        return Dual(res_val, res_val * (exp.der * std::log(base.val) + exp.val * base.der / base.val));
        #endif
    }

    friend HOST_DEVICE Dual exp(const Dual& d) {
        #ifdef __CUDA_ARCH__
        T res = ::exp(d.val);
        #else
        T res = std::exp(d.val);
        #endif
        return Dual(res, res * d.der);
    }

    friend HOST_DEVICE Dual log(const Dual& d) {
        #ifdef __CUDA_ARCH__
        return Dual(::log(d.val), d.der / d.val);
        #else
        return Dual(std::log(d.val), d.der / d.val);
        #endif
    }
    
    friend HOST_DEVICE Dual abs(const Dual& d) {
        // Derivative of abs is sgn(x) * x'
        double sgn = 0.0;
        if (d.val > 0) sgn = 1.0;
        else if (d.val < 0) sgn = -1.0;
        
        #ifdef __CUDA_ARCH__
        return Dual(::fabs(d.val), sgn * d.der);
        #else
        return Dual(std::abs(d.val), sgn * d.der);
        #endif
    }

    // Comparisons (Friend functions to allow T vs Dual)
    friend HOST_DEVICE bool operator<(const Dual& a, const Dual& b) { return a.val < b.val; }
    friend HOST_DEVICE bool operator>(const Dual& a, const Dual& b) { return a.val > b.val; }
    friend HOST_DEVICE bool operator<=(const Dual& a, const Dual& b) { return a.val <= b.val; }
    friend HOST_DEVICE bool operator>=(const Dual& a, const Dual& b) { return a.val >= b.val; }
    
    friend HOST_DEVICE bool operator<(const Dual& a, T b) { return a.val < b; }
    friend HOST_DEVICE bool operator>(const Dual& a, T b) { return a.val > b; }
    friend HOST_DEVICE bool operator<=(const Dual& a, T b) { return a.val <= b; }
    friend HOST_DEVICE bool operator>=(const Dual& a, T b) { return a.val >= b; }
    
    friend HOST_DEVICE bool operator<(T a, const Dual& b) { return a < b.val; }
    friend HOST_DEVICE bool operator>(T a, const Dual& b) { return a > b.val; }
    friend HOST_DEVICE bool operator<=(T a, const Dual& b) { return a <= b.val; }
    friend HOST_DEVICE bool operator>=(T a, const Dual& b) { return a >= b.val; }
    
    // Max/Min
    friend HOST_DEVICE Dual max(const Dual& a, const Dual& b) {
        if (a.val >= b.val) return a;
        else return b;
    }
    
    friend HOST_DEVICE Dual min(const Dual& a, const Dual& b) {
        if (a.val <= b.val) return a;
        else return b;
    }
};

using Duald = Dual<double>;
