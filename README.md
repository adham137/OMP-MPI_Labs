# Parallel & Distributed Algorithms – Lab Solutions

**Course:** Parallel & Distributed Algorithms  

## Overview

This repository contains C implementations of the lab assignments for the “Parallel & Distributed Algorithms” course. The exercises leverage OpenMP and MPI to demonstrate task dependencies, custom data aggregation, and hybrid parallelism as specified in the provided LabAssignments.pdf.

## Assignment Summaries

### Problem 1: OpenMP Task Dependencies for Wavefront Computation  
- **Objective:** Parallelize a wavefront computation on a 2D grid using OpenMP tasks with dependencies.  
- **Details:** Each cell `(i, j)` depends on its top `(i-1, j)` and left `(i, j-1)` neighbors. Initialize the first row and column to 1, then use  
  ```c
  #pragma omp task depend(inout: grid[i][j])
  ```  
  with block-based tasks to reduce overhead and validate that each cell equals the sum of its top and left neighbors. citeturn0file0

### Problem 2: MPI Custom Struct for Distributed Matrix Aggregation  
- **Objective:** Define and use a custom MPI struct (`MatrixMeta`) to gather distributed k×k matrices from all processes.  
- **Details:** Each MPI process generates a random k × k matrix. A struct  
  ```c
  typedef struct {
    int rows, cols;
    int data[k*k];
  } MatrixMeta;
  ```  
  is committed via `MPI_Type_create_struct`/`MPI_Type_commit` and used with `MPI_Gather` to collect all matrices on the root process for consistency checks. citeturn0file0

### Problem 3: Hybrid MPI + OpenMP Parallel Prime Sieve  
- **Objective:** Implement a prime sieve over the range [2, N] using MPI for inter-process distribution and OpenMP for intra-process marking.  
- **Details:** Divide the range among MPI ranks, broadcast the primes up to √N with `MPI_Bcast`, and mark composites in parallel using  
  ```c
  #pragma omp for schedule(dynamic)
  ```  
  Finally collect the primes via `MPI_Gather` or `MPI_Reduce`. citeturn0file0

## Repository Structure

```
.
├── problem1_wavefront/       # OpenMP wavefront computation
│   ├── src/
│   │   └── wavefront.cpp
│   │   └── wavefront.exe
│   └── problem1_wavefront.pdf
├── problem2_matrix/          # MPI custom struct for matrix aggregation
│   ├── src/
│   │   └── matrix_aggregation.cpp
│   │   └── matrix_aggregation.exe
│   └── matrix_aggregation.pdf
├── problem3_hybrid_sieve/    # Hybrid MPI+OpenMP prime sieve
│   ├── src/
│   │   └── hybrid_sieve.cpp
│   │   └── hybrid_sieve.exe
│   └── hybrid_sieve.pdf
└── README.md                 # This overview file
```

## Prerequisites

- **Compiler:** GCC or Clang with OpenMP support  
- **MPI:** Microsoft MPI  
- **Build tools:** GNU Make (if provided)

## Build & Run Instructions


1. **Problem 1**  
   ```bash
   cd problem1_wavefront/src
   gcc -fopenmp wavefront.cpp -o wavefront
   OMP_NUM_THREADS=4 ./wavefront N
   ```

2. **Problem 2**  
   ```bash
   cd problem2_matrix/src
   mpicc matrix_aggregation.cpp -o matrix_agg
   mpirun -np 4 ./matrix_agg
   ```

3. **Problem 3**  
   ```bash
   cd problem3_hybrid_sieve/src
   mpicc -fopenmp hybrid_sieve.cpp -o hybrid_sieve
   mpirun -np 4 ./hybrid_sieve N
   ```
