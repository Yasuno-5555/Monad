
#include <iostream>
#include <vector>
#include <cassert>
#include "src/grid/MultiDimGrid.hpp"

// Simple test for MultiDimGrid indexing
int main() {
    std::cout << "Testing MultiDimGrid..." << std::endl;
    
    // Create dummy grids
    std::vector<double> m_nodes = {0.0, 1.0, 2.0};
    std::vector<double> a_nodes = {10.0, 20.0};
    int n_z = 2; // e.g. low, high
    
    UnifiedGrid m(m_nodes);
    UnifiedGrid a(a_nodes);
    
    MultiDimGrid grid(m, a, n_z);
    
    std::cout << "Dimensions: Nm=" << grid.N_m << ", Na=" << grid.N_a << ", Nz=" << grid.N_z << std::endl;
    std::cout << "Total Size: " << grid.total_size << " (Expected: " << 3*2*2 << " = 12)" << std::endl;
    assert(grid.total_size == 12);
    
    // Test Forward Indexing
    // idx = iz * (2*3) + ia * 3 + im
    // idx(2, 1, 1) -> 1*6 + 1*3 + 2 = 11 (Last element)
    int idx_last = grid.idx(2, 1, 1);
    std::cout << "idx(2,1,1) = " << idx_last << std::endl;
    assert(idx_last == 11);
    
    // Test Reverse Indexing
    int im, ia, iz;
    grid.get_coords(11, im, ia, iz);
    std::cout << "Coords of 11: (" << im << ", " << ia << ", " << iz << ")" << std::endl;
    assert(im == 2);
    assert(ia == 1);
    assert(iz == 1);
    
    grid.get_coords(4, im, ia, iz);
    // 4 = 0*6 + 1*3 + 1 -> (1, 1, 0)
    std::cout << "Coords of 4: (" << im << ", " << ia << ", " << iz << ")" << std::endl;
    assert(im == 1);
    assert(ia == 1);
    assert(iz == 0);
    
    std::cout << "SUCCESS: MultiDimGrid indexing verified." << std::endl;
    return 0;
}
