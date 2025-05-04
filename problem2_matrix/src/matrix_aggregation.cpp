// mpiexec -n 5 "./problem2_matrix/src/matrix_aggregation.exe"
#include <iostream>
#include <ctime>
#include <cstddef>
#include <vector>
#include <chrono>
#include <mpi.h>


using namespace std;

# define k 500        // sub-matrix size (k x k)
# define n_workers 5 // number of tasks used in the parallel version (used when executing the file in cmd)

typedef struct { int rows, cols; int data[k*k]; } MatrixMeta;

int sequentialMatrixAggregation();                    // measure the time it takes sequentially
//void parallelMatrixAggregation(int, char**);           // measure the time it takes parallely

int main(int argc, char* argv[]) {
    // sequentialMatrixAggregation();
    MPI_Init(&argc, &argv);

    int rank, num_tasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    // init the timer for parallel
    std::chrono::_V2::system_clock::time_point start_p;
    if(rank == 0) start_p = chrono::high_resolution_clock::now();

    // each task generates a local sub-matrix of size k x k
    srand(time(0) + rank); // change the seed for each task to get different numbets
    MatrixMeta localMatrix;
    localMatrix.rows = k; localMatrix.cols = k;
    for(int i=0; i<k*k; i++) localMatrix.data[i] = rand()%100;

    // defining MPI struct
    MPI_Datatype MatrixMeta_MPI;
    int blockLengths[3] = {1, 1, k*k};                      // size of each parameter
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};    // type of each parameter
    MPI_Aint displacements[3] = {offsetof(MatrixMeta, rows), offsetof(MatrixMeta, cols), offsetof(MatrixMeta, data)}; // displacements of each parameter

    
    // create and commit the MPI struct
    MPI_Type_create_struct(3, blockLengths, displacements, types, &MatrixMeta_MPI);
    MPI_Type_commit(&MatrixMeta_MPI);

    // gather all sub-matricies in the root process
    vector<MatrixMeta> gatheredMatricies;
    if(rank == 0) gatheredMatricies.resize(num_tasks);  // resize to the number of tasks (since each task will be saved in an index)
    MPI_Gather(
        &localMatrix,
        1,
        MatrixMeta_MPI,
        gatheredMatricies.data(),
        1,
        MatrixMeta_MPI,
        0,
        MPI_COMM_WORLD          
    );
    // root task verifies results
    if (rank == 0) {
        auto stop_p = chrono::high_resolution_clock::now();
        auto duration_p = chrono::duration_cast<chrono::microseconds>(stop_p - start_p);   

        cout << "\n--- Root Process Verification ---" << endl;
        cout << "Received " << gatheredMatricies.size() << " matrices." << endl;

        int total_rows_received = 0;
        bool data_consistent = true;

        for (int i = 0; i < num_tasks; ++i) {
            total_rows_received += gatheredMatricies[i].rows;
            if (gatheredMatricies[i].rows != k || gatheredMatricies[i].cols != k) {
                data_consistent = false;
                cerr << "ERROR: Inconsistency found in matrix dimensions from process " << i << endl;
                cerr << "  Expected (" << k << "x" << k << "), Got ("
                            << gatheredMatricies[i].rows << "x" << gatheredMatricies[i].cols << ")" << endl;
            }
        }

        int expected_total_rows = num_tasks * k;
        cout << "Total rows received: " << total_rows_received << endl;
        cout << "Expected total rows: " << expected_total_rows << endl;

        if (total_rows_received == expected_total_rows && data_consistent) {
            cout << "Verification successful: Total rows match and dimensions are consistent." << endl;
        } else {
            cerr << "Verification FAILED!" << endl;
        }
        

        int seq_dur = sequentialMatrixAggregation();
        cout << "Parallel execution Time: " << duration_p.count() << " microseconds" << endl;
        cout << "Sequential execution Time: " << seq_dur << " microseconds" << endl;

        cout << "Speedup: " << seq_dur / duration_p.count() << endl;
    }
    // cout << "Parallel execution Time: " << duration.count() << " microseconds" << endl;
    MPI_Finalize();
    return 0;
}

int sequentialMatrixAggregation(){
    auto start = chrono::high_resolution_clock::now();
    vector<vector<int>> res (k*n_workers);
    for(int i=0; i<k*n_workers; i++){
        vector<int> temp(k*n_workers);
        for(int j=0; j<k*n_workers; j++) temp[j] = rand()%100;
        res.push_back(temp);
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);   
    return duration.count();
}
