#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cmath>
#include <iostream>

// Simple Dual Number for Forward-Mode AD
// Represents x + eps * dx where eps^2 = 0
template <typename T>
struct Dual {
    T val; // Value
    T der; // Derivative

    Dual(T v = 0.0, T d = 0.0) : val(v), der(d) {}

    // Arithmetic
    Dual operator+(const Dual& other) const { return Dual(val + other.val, der + other.der); }
    Dual operator-(const Dual& other) const { return Dual(val - other.val, der - other.der); }
    Dual operator*(const Dual& other) const { return Dual(val * other.val, val * other.der + der * other.val); }
    Dual operator/(const Dual& other) const { 
        return Dual(val / other.val, (der * other.val - val * other.der) / (other.val * other.val)); 
    }

    // Scalar arithmetic
    Dual operator+(T scalar) const { return Dual(val + scalar, der); }
    Dual operator-(T scalar) const { return Dual(val - scalar, der); }
    Dual operator*(T scalar) const { return Dual(val * scalar, der * scalar); }
    Dual operator/(T scalar) const { return Dual(val / scalar, der / scalar); }
    
    // Friends for scalar on LHS
    friend Dual operator+(T scalar, const Dual& d) { return Dual(scalar + d.val, d.der); }
    friend Dual operator-(T scalar, const Dual& d) { return Dual(scalar - d.val, -d.der); }
    friend Dual operator*(T scalar, const Dual& d) { return Dual(scalar * d.val, scalar * d.der); }
    friend Dual operator/(T scalar, const Dual& d) { return Dual(scalar / d.val, -scalar * d.der / (d.val * d.val)); }

    // Math Functions
    friend Dual pow(const Dual& base, T exp) {
        // d/dx (x^n) = n * x^(n-1) * x'
        double e = static_cast<double>(exp);
        return Dual(std::pow(base.val, e), e * std::pow(base.val, e - 1.0) * base.der);
    }
    
    // For general pow(Dual, Dual)
    friend Dual pow(const Dual& base, const Dual& exp) {
        // y = u^v => ln y = v ln u => y'/y = v' ln u + v u'/u
        T res_val = std::pow(base.val, exp.val);
        return Dual(res_val, res_val * (exp.der * std::log(base.val) + exp.val * base.der / base.val));
    }

    friend Dual exp(const Dual& d) {
        T res = std::exp(d.val);
        return Dual(res, res * d.der);
    }

    friend Dual log(const Dual& d) {
        return Dual(std::log(d.val), d.der / d.val);
    }
    
    friend Dual abs(const Dual& d) {
        // Derivative of abs is sgn(x) * x'
        double sgn = 0.0;
        if (d.val > 0) sgn = 1.0;
        else if (d.val < 0) sgn = -1.0;
        return Dual(std::abs(d.val), sgn * d.der);
    }

    // Comparisons (Friend functions to allow T vs Dual)
    friend bool operator<(const Dual& a, const Dual& b) { return a.val < b.val; }
    friend bool operator>(const Dual& a, const Dual& b) { return a.val > b.val; }
    friend bool operator<=(const Dual& a, const Dual& b) { return a.val <= b.val; }
    friend bool operator>=(const Dual& a, const Dual& b) { return a.val >= b.val; }
    
    friend bool operator<(const Dual& a, T b) { return a.val < b; }
    friend bool operator>(const Dual& a, T b) { return a.val > b; }
    friend bool operator<=(const Dual& a, T b) { return a.val <= b; }
    friend bool operator>=(const Dual& a, T b) { return a.val >= b; }
    
    friend bool operator<(T a, const Dual& b) { return a < b.val; }
    friend bool operator>(T a, const Dual& b) { return a > b.val; }
    friend bool operator<=(T a, const Dual& b) { return a <= b.val; }
    friend bool operator>=(T a, const Dual& b) { return a >= b.val; }
    
    // Max/Min
    friend Dual max(const Dual& a, const Dual& b) {
        if (a.val >= b.val) return a;
        else return b;
    }
    
    friend Dual min(const Dual& a, const Dual& b) {
        if (a.val <= b.val) return a;
        else return b;
    }
};

using Duald = Dual<double>;
