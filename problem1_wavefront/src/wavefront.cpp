#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>

using namespace std;
using waveFrontComputation = vector<vector<int>>(*) (vector<vector<int>>);

# define N 10001 // array size
# define TS 1000 // parallel block size (can be a measure of granularity)

vector<vector<int>> initGrid(int);                              // initialize the first row and column with 1's 

bool validateGrid(vector<vector<int>>);                         // validate that each cell is the sum of its top and left

vector<vector<int>> sequentialWavefront(vector<vector<int>>);   // compute the wavefront sequentially

vector<vector<int>> parallelWavefront(vector<vector<int>>);     // compute the wavefront parallely

vector<vector<int>> measureExecutionTime(vector<vector<int>>, waveFrontComputation); // measure the execution time of the wave front calculation (either sequential or parallel verison)


int main(){
    // Initialize the grid
    vector<vector<int>> grid = initGrid(N);

    // Measure execution time for sequential version
    cout << endl << "Sequential Execution" << endl;
    vector<vector<int>> sequentialOut = measureExecutionTime(grid, sequentialWavefront);
    // validate sequential version
    cout << "Is valid: " << validateGrid(sequentialOut) << endl;

    // Measure execution time for parallel version
    cout << endl << "Parallel Execution" << endl;
    vector<vector<int>> parallelOut = measureExecutionTime(grid, parallelWavefront);
    // validate parallel version
    cout << "Is valid: " << validateGrid(parallelOut) << endl;
    
    return 0;

}

vector<vector<int>> initGrid(int n){
    vector<vector<int>> res (n, vector<int>(n, 0));
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(i==0) res[i][j] = 1;
            if(j==0) res[i][j] = 1;
        }
    }

    return res;
}

bool validateGrid(vector<vector<int>> grid){
    int i, j;
    for(i=1; i<grid.size(); i++){
        for(j=1; j<grid.size(); j++){
            if(grid[i][j] != (grid[i-1][j] + grid[i][j-1])) return false;
        }
    }
    return true;
}  

vector<vector<int>> measureExecutionTime(vector<vector<int>> grid, waveFrontComputation func){
    auto start = chrono::high_resolution_clock::now();
    vector<vector<int>> res = func(grid);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(stop - start);   
    cout << "Execution Time: " << duration.count() << " microseconds" << endl;
    return res;
}

vector<vector<int>> sequentialWavefront(vector<vector<int>> grid){
    int i, j;
    for(i=1; i<grid.size(); i++){
        for(j=1; j<grid.size(); j++){
            grid[i][j] = (grid[i-1][j] + grid[i][j-1]);
        }
    }

    return grid;
}

vector<vector<int>> parallelWavefront(vector<vector<int>> grid){
    #pragma omp parallel
    #pragma omp single
    {   
        // iterate over the parallel blocks
        for (int ii = 1; ii < N; ii += TS){
            for (int jj = 1; jj < N; jj += TS) {

                // assign each task a block 
                #pragma omp task depend(inout: grid[ii:TS][jj:TS])
                {
                  for (int i = ii; i < ii + TS; ++i){
                    for (int j = jj; j < jj + TS; ++j){
                        grid[i][j] = grid[i-1][j] + grid[i][j-1];
                        // cout << "Grid (" << i << ", " << j << ") = " << grid[i][j] << endl;
                    }
                  }
                }
            }
        }
        #pragma omp taskwait
    }
    
    
    

    return grid;
} 