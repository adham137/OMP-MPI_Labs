
#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <math.h>
#include <mpi.h>



using namespace std;

# define N 1000000000           // limit (inclusive)
# define n_proc 2       // number of tasks used in the parallel version (used when executing the file in cmd)

void mark_composites_parallel(vector<char>& arr, const vector<long long> prime_nums, const long long start, const long long end);    // given a range and prime numbers,mark composite numbers with zeros parallely
vector<char> sequential_sieve(long long n);                                   // given a range, mark composite numbers with zeros sequentially
bool vectors_identical(const vector<char>& a, const vector<char>& b);   // check if two vectors are identical

// run command: mpirun -np <N> ./hybrid_sieve.exe
int main(int argc, char* argv[]) {
    auto start_p = chrono::high_resolution_clock::now();

    vector<char> arr(N+1, true);

    MPI_Init(&argc, &argv);
    int rank, num_tasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    // process zero will take 0 -> sqrt(N)
    // the rest of the array will be divided accross the rest of processes
    long long sqrt_N = sqrt(N);
    long long remaining_range = N - sqrt_N;
    long long chunk_size = remaining_range / (num_tasks - 1);
    long long remaining_elements = remaining_range % (num_tasks - 1); // incase N is not divisible by num_tasks

    long long start, end;
    if(rank == 0){
        start = 0;
        end = sqrt_N+1;
    }else{
        start = (sqrt_N+1) + (rank-1)*chunk_size; 
        end = (rank == num_tasks-1)? start + chunk_size + remaining_elements : start + chunk_size;
    }

    // cout <<"Array for rank (" << rank <<"): ";
    // cout << "Start: " << start << ", End: " << end << " Range: " << end - start << endl;
    // cout << endl;

    
    // process 0 finds prime numbers in range of 2->sqrt(N)
    vector<long long> prime_nums;
    if(rank == 0){
        arr[0] = arr[1] = false;
        for(long long i=0; i<end; i++){ 
            if(arr[i]){
                // MPI_Bcast(&prime_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
                prime_nums.push_back(i);
                //cout << "Rank (" << rank << "): Removing prime number: " << i << endl;
                mark_composites_parallel(arr, {i}, start, end);

            }
        }
        arr[0] = arr[1] = true;

    }

    // broadcast the list of prime numbers
    long long size = prime_nums.size();
    MPI_Bcast(&size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);                    // broadcast the size first to allow other processes to resize
    if(rank != 0){
        prime_nums.resize(size);
    }
    MPI_Bcast(prime_nums.data(), size, MPI_LONG_LONG, 0, MPI_COMM_WORLD);     // broadcast the prime numbers
    
    
    mark_composites_parallel(arr, prime_nums, start, end);              // mark all the composites in the local array


    // reduce all local arrays of all processes into one single
    // array (global_arr) using the and operator
    vector<char> global_arr(N+1, true); 
    MPI_Reduce(
        arr.data(),     
        global_arr.data(),
        N+1,     
        MPI_C_BOOL,
        MPI_LAND,  
        0,       
        MPI_COMM_WORLD
    );
    auto stop_p = chrono::high_resolution_clock::now();
    auto duration_p = chrono::duration_cast<std::chrono::microseconds>(stop_p - start_p);   
    

    // output all prime numbers from process 0
    if (rank == 0) {
        auto start_s = chrono::high_resolution_clock::now();
        vector<char> global_arr_sequential = sequential_sieve(N);
        auto stop_s = chrono::high_resolution_clock::now();
        auto duration_s = chrono::duration_cast<std::chrono::microseconds>(stop_s - start_s);

        // cout << "Global array (parallel): ";
        // for(long long i=0; i<global_arr.size(); i++){
        //     if(global_arr[i]){
        //         cout << i << " ";
        //     }
        // }
        // cout << endl;

        // cout << "Global array (sequential): ";
        // for(int i=0; i<global_arr_sequential.size(); i++){
        //     if(global_arr_sequential[i]){
        //         cout << i << " ";
        //     }
        // }
        // cout << endl;
        

        cout << "Parallel Time: " << duration_p.count() << " microseconds" << endl;
        cout << "Sequential Time: " << duration_s.count() << " microseconds" << endl;

        cout << "Is parallel answer valid: " << (vectors_identical(global_arr, global_arr_sequential)? "Yes" : "No") << endl;

        cout << "Speedup: " << duration_s.count() / duration_p.count() << endl;
    }
    


    MPI_Finalize();
    return 0;
}

void mark_composites_parallel(vector<char>& arr, const vector<long long> prime_nums, const long long start, const long long end) {

    long long p_num, first_occurence;
    // iterate over the prime numbers and mark the composites
    #pragma omp parallel for schedule(guided) private(p_num, first_occurence)
    for(long long i=0; i<prime_nums.size(); i++){
        p_num = prime_nums[i];
        first_occurence = (p_num>=start && p_num<end)? p_num : ceil(start/p_num)*p_num; // find the first occurence of this prime number
        for(long long j=first_occurence; j<end; j+=p_num){
            if (p_num != j) arr[j] = false;
        }
    } 

}

vector<char> sequential_sieve(long long n) {
    vector<char> res (n+1, true);
    for (long long p = 2; p * p <= n; p++) {
        if (res[p]) {
            for (long long i = p * 2; i <= n; i += p) {
                res[i] = false;
            }
        }
    }
    return res;
}

bool vectors_identical(const vector<char>& a, const vector<char>& b) {
    // Quick size check
    if (a.size() != b.size()) return false;

    bool identical = true;

    // Parallel-for with a logical‐AND reduction
    #pragma omp parallel for reduction(&&:identical)
    for (long long i=0; i<a.size(); ++i) {
        identical = identical && (a[i] == b[i]);
    }

    return identical;
}
