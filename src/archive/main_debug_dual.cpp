#define NOMINMAX
#include "Dual.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    // 1. Vector instantiation
    std::vector<Duald> v(10);
    v[0] = Duald(1.0, 0.0);
    
    // 2. Math
    Duald a(2.0, 1.0);
    Duald b = pow(a, 2.0); // pow(Dual, double)
    Duald c = pow(a, a);   // pow(Dual, Dual)
    
    // 3. Comparison
    bool x = (a < b);
    
    // 4. Sort/Lower Bound (causes implicit strict weak ordering checks)
    std::vector<Duald> vec = { Duald(3.0), Duald(1.0), Duald(2.0) };
    std::sort(vec.begin(), vec.end());
    
    auto it = std::lower_bound(vec.begin(), vec.end(), Duald(1.5));
    
    // 5. Lower Bound with double?
    // auto it2 = std::lower_bound(vec.begin(), vec.end(), 1.5); 
    // This requires operator<(Dual, double) AND operator<(double, Dual).
    
    std::cout << "Debug Build Success" << std::endl;
    return 0;
}
