#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "common.h"

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);

        if (abort)
            MPI_Abort(MPI_COMM_WORLD, code);
    }
}

// CUDA kernels

__global__ void calculate_local_sums_kernel(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* __restrict__ matrix,
    const int* __restrict__ row_labels,
    const int* __restrict__ col_labels,
    double* __restrict__ sums,
    int* __restrict__ counts)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= local_cols)
        return;

    const int c_lbl = col_labels[j];
    const int col_offset = j * num_rows;

    for (int i = 0; i < num_rows; i++) {

        const int r_lbl = row_labels[i];
        const int idx = r_lbl * num_col_labels + c_lbl;

        atomicAdd(&sums[idx],
                  (double)matrix[col_offset + i]);

        atomicAdd(&counts[idx], 1);
    }
}

__global__ void compute_row_dist_kernel(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* __restrict__ matrix,
    const int* __restrict__ col_labels,
    const double* __restrict__ cluster_avg,
    double* __restrict__ row_dist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_rows)
        return;

    for (int k = 0; k < num_row_labels; k++) {

        double dist = 0.0;
        const int base = k * num_col_labels;

        for (int j = 0; j < local_cols; j++) {

            const int c_lbl = col_labels[j];

            const double avg =
                cluster_avg[base + c_lbl];

            const double diff =
                avg - (double)matrix[j * num_rows + i];

            dist += diff * diff;
        }

        row_dist[i * num_row_labels + k] = dist;
    }
}

__global__ void update_col_kernel(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* __restrict__ matrix,
    const int* __restrict__ row_labels,
    int* __restrict__ col_labels,
    const double* __restrict__ cluster_avg,
    int* d_changes)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= local_cols)
        return;

    const int col_offset = j * num_rows;

    int best_label = -1;
    double best_dist = INFINITY;

    for (int k = 0; k < num_col_labels; k++) {

        double dist = 0.0;

        for (int i = 0; i < num_rows; i++) {

            const float item =
                matrix[col_offset + i];

            const int r_lbl =
                row_labels[i];

            const double avg =
                cluster_avg[r_lbl * num_col_labels + k];

            const double diff =
                avg - (double)item;

            dist += diff * diff;
        }

        if (dist < best_dist) {
            best_dist = dist;
            best_label = k;
        }
    }

    if (col_labels[j] != best_label) {
        col_labels[j] = best_label;
        atomicAdd(d_changes, 1);
    }
}


int main(int argc, const char* argv[]) {

    // MPI initialization

    MPI_Init(&argc, const_cast<char***>(&argv));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;

    std::string output_file;

    int num_rows;
    int num_cols;
    int num_row_labels;
    int num_col_labels;
    int max_iter;

    if (rank == 0) {
        parse_arguments(
            argc,
            argv,
            &num_rows,
            &num_cols,
            &num_row_labels,
            &num_col_labels,
            &matrix,
            &row_labels,
            &col_labels,
            &output_file,
            &max_iter);
    }

    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        row_labels.resize(num_rows);

    MPI_Bcast(
        row_labels.data(),
        num_rows,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    // Data distribution

    std::vector<int> counts(size), displs(size);

    int base = num_cols / size;
    int rem = num_cols % size;
    int offset = 0;

    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    int local_cols = counts[rank];

    std::vector<label_type> local_col_labels(local_cols);

    MPI_Scatterv(
        rank == 0 ? col_labels.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_INT,
        local_col_labels.data(),
        local_cols,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    std::vector<float> local_matrix(num_rows * local_cols);

    if (rank == 0) {

        std::vector<float> tmp(num_rows * num_cols);

        for (int j = 0; j < num_cols; j++)
            for (int i = 0; i < num_rows; i++)
                tmp[j * num_rows + i] =
                    matrix[i * num_cols + j];

        std::vector<int> sc(size), sd(size);

        for (int r = 0; r < size; r++) {
            sc[r] = counts[r] * num_rows;
            sd[r] = displs[r] * num_rows;
        }

        MPI_Scatterv(
            tmp.data(),
            sc.data(),
            sd.data(),
            MPI_FLOAT,
            local_matrix.data(),
            num_rows * local_cols,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(
            nullptr,
            nullptr,
            nullptr,
            MPI_FLOAT,
            local_matrix.data(),
            num_rows * local_cols,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD);
    }

    // GPU memory allocation

    float *d_matrix;

    int *d_row_labels;
    int *d_col_labels;
    int *d_changes;
    int *d_counts;

    double *d_cluster_avg;
    double *d_sums;
    double *d_row_dist;

    cudaCheck(cudaMalloc(
        &d_matrix,
        num_rows * local_cols * sizeof(float)));

    cudaCheck(cudaMalloc(
        &d_row_labels,
        num_rows * sizeof(int)));

    cudaCheck(cudaMalloc(
        &d_col_labels,
        local_cols * sizeof(int)));

    cudaCheck(cudaMalloc(
        &d_cluster_avg,
        num_row_labels * num_col_labels * sizeof(double)));

    cudaCheck(cudaMalloc(
        &d_sums,
        num_row_labels * num_col_labels * sizeof(double)));

    cudaCheck(cudaMalloc(
        &d_counts,
        num_row_labels * num_col_labels * sizeof(int)));

    cudaCheck(cudaMalloc(
        &d_row_dist,
        num_rows * num_row_labels * sizeof(double)));

    cudaCheck(cudaMalloc(
        &d_changes,
        sizeof(int)));

    cudaCheck(cudaMemcpy(
        d_matrix,
        local_matrix.data(),
        num_rows * local_cols * sizeof(float),
        cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(
        d_col_labels,
        local_col_labels.data(),
        local_cols * sizeof(int),
        cudaMemcpyHostToDevice));

    //  Main clustering loop 

    auto before = std::chrono::high_resolution_clock::now();

    int iter;

    for (iter = 0; iter < max_iter; iter++) {

        cudaCheck(cudaMemset(
            d_sums,
            0,
            num_row_labels * num_col_labels * sizeof(double)));

        cudaCheck(cudaMemset(
            d_counts,
            0,
            num_row_labels * num_col_labels * sizeof(int)));

        cudaCheck(cudaMemcpy(
            d_row_labels,
            row_labels.data(),
            num_rows * sizeof(int),
            cudaMemcpyHostToDevice));

        calculate_local_sums_kernel<<<
            (local_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE>>>(
                num_rows,
                local_cols,
                num_col_labels,
                d_matrix,
                d_row_labels,
                d_col_labels,
                d_sums,
                d_counts);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        std::vector<double> h_sums(
            num_row_labels * num_col_labels);

        std::vector<int> h_counts(
            num_row_labels * num_col_labels);

        cudaCheck(cudaMemcpy(
            h_sums.data(),
            d_sums,
            h_sums.size() * sizeof(double),
            cudaMemcpyDeviceToHost));

        cudaCheck(cudaMemcpy(
            h_counts.data(),
            d_counts,
            h_counts.size() * sizeof(int),
            cudaMemcpyDeviceToHost));

        MPI_Allreduce(
            MPI_IN_PLACE,
            h_sums.data(),
            h_sums.size(),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            MPI_IN_PLACE,
            h_counts.data(),
            h_counts.size(),
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        for (size_t i = 0; i < h_sums.size(); i++)
            h_sums[i] /= h_counts[i];

        cudaCheck(cudaMemcpy(
            d_cluster_avg,
            h_sums.data(),
            h_sums.size() * sizeof(double),
            cudaMemcpyHostToDevice));

        //  Row label update

        compute_row_dist_kernel<<<
            (num_rows + BLOCK_SIZE - 1 ) / BLOCK_SIZE,
            BLOCK_SIZE>>>(
                num_rows,
                local_cols,
                num_row_labels,
                num_col_labels,
                d_matrix,
                d_col_labels,
                d_cluster_avg,
                d_row_dist);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        std::vector<double> row_dist(
            num_rows * num_row_labels);

        cudaCheck(cudaMemcpy(
            row_dist.data(),
            d_row_dist,
            row_dist.size() * sizeof(double),
            cudaMemcpyDeviceToHost));

        MPI_Allreduce(
            MPI_IN_PLACE,
            row_dist.data(),
            row_dist.size(),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        int row_updates = 0;

        if (rank == 0) {

            for (int i = 0; i < num_rows; i++) {

                int best = -1;
                double best_val = INFINITY;

                for (int k = 0; k < num_row_labels; k++) {

                    double d =
                        row_dist[i * num_row_labels + k];

                    if (d < best_val) {
                        best_val = d;
                        best = k;
                    }
                }

                if (row_labels[i] != best) {
                    row_labels[i] = best;
                    row_updates++;
                }
            }
        }

        MPI_Bcast(
            row_labels.data(),
            num_rows,
            MPI_INT,
            0,
            MPI_COMM_WORLD);

        cudaCheck(cudaMemcpy(
            d_row_labels,
            row_labels.data(),
            num_rows * sizeof(int),
            cudaMemcpyHostToDevice));

        // Column label update

        cudaCheck(cudaMemset(d_changes, 0, sizeof(int)));

        update_col_kernel<<<
            (local_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE>>>(
                num_rows,
                local_cols,
                num_col_labels,
                d_matrix,
                d_row_labels,
                d_col_labels,
                d_cluster_avg,
                d_changes);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        int col_upd_local;
        int col_upd_global;
        int total_upd;

        cudaCheck(cudaMemcpy(
            &col_upd_local,
            d_changes,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        cudaCheck(cudaMemcpy(
            local_col_labels.data(),
            d_col_labels,
            local_cols * sizeof(int),
            cudaMemcpyDeviceToHost));

        total_upd = row_updates;

        MPI_Bcast(
            &total_upd,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            &col_upd_local,
            &col_upd_global,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        if ((total_upd + col_upd_global) == 0)
            break;
    }

    auto after = std::chrono::high_resolution_clock::now();

    double time_seconds =
        std::chrono::duration<double>(
            after - before).count();

    if (rank == 0) {

        std::cout
            << "clustering time total: "
            << time_seconds
            << " seconds\n";

        std::cout
            << "clustering time per iteration: "
            << (time_seconds / std::max(iter, 1))
            << " seconds\n";
    }

    // Gather results 

    cudaCheck(cudaMemcpy(
        local_col_labels.data(),
        d_col_labels,
        local_cols * sizeof(int),
        cudaMemcpyDeviceToHost));

    if (rank == 0)
        col_labels.resize(num_cols);

    MPI_Gatherv(
        local_col_labels.data(),
        local_cols,
        MPI_INT,
        rank == 0 ? col_labels.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    if (rank == 0) {

        write_labels(
            output_file,
            num_rows,
            num_cols,
            row_labels.data(),
            col_labels.data());
    }

    //  Cleanup 

    cudaFree(d_matrix);
    cudaFree(d_row_labels);
    cudaFree(d_col_labels);
    cudaFree(d_cluster_avg);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_row_dist);
    cudaFree(d_changes);

    MPI_Finalize();

    return 0;
}