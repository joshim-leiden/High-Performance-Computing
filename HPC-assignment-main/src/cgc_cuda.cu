#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <string>
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

__global__
void calculate_local_sums_kernel(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* matrix,
    const int* row_labels,
    const int* col_labels,
    double* sums,
    int* counts)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= local_cols) return;

    for (int i = 0; i < num_rows; i++) {
        int r_lbl = row_labels[i];
        int c_lbl = col_labels[j];
        int idx = r_lbl * num_col_labels + c_lbl;

        atomicAdd(&sums[idx], (double)matrix[j * num_rows + i]);
        atomicAdd(&counts[idx], 1);
    }
}

__global__
void compute_row_dist_parallel_kernel(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    const int* col_labels,
    const double* cluster_avg,
    double* row_dist)
{
    int j =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(j >= local_cols)
        return;

    int c_lbl =
        col_labels[j];

    for(int row=0; row<num_rows; row++)
    {
        double value =
            matrix[j*num_rows + row];

        for(int k=0; k<num_row_labels; k++)
        {
            double avg =
                cluster_avg[
                    k*num_col_labels +
                    c_lbl];

            double diff =
                avg - value;

            atomicAdd(
                &row_dist[
                    row*num_row_labels + k],
                diff*diff);
        }
    }
}


__global__
void update_row_labels_kernel(
    int num_rows,
    int num_row_labels,
    const double* row_dist,
    int* row_labels,
    int* changes)
{
    int row =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(row >= num_rows)
        return;

    int best =
        row_labels[row];

    double best_val =
        INFINITY;

    for(int k=0; k<num_row_labels; k++)
    {
        double d =
            row_dist[
                row*num_row_labels + k];

        if(d < best_val)
        {
            best_val = d;
            best = k;
        }
    }

    if(best != row_labels[row])
    {
        row_labels[row] = best;
        atomicAdd(changes,1);
    }
}


__global__
void update_col_kernel(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* matrix,
    const int* row_labels,
    int* col_labels,
    const double* cluster_avg,
    int* d_changes,
    double* d_total_dist)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= local_cols) return;

    int best_label = -1;
    double best_dist = INFINITY;

    for (int k = 0; k < num_col_labels; k++) {
        double dist = 0.0;

        for (int i = 0; i < num_rows; i++) {
            float item = matrix[j * num_rows + i];
            int r_lbl = row_labels[i];
            double avg = cluster_avg[r_lbl * num_col_labels + k];
            double diff = avg - (double)item;
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

    atomicAdd(d_total_dist, best_dist);
}

static inline double calculate_distance(double avg, double item) {
    double diff = avg - item;
    return diff * diff;
}

static std::pair<int, double> update_row_labels_serial_like(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    const label_type* col_labels,
    const double* cluster_avg)
{
    int num_updated = 0;
    double total_dist = 0.0;

    for (int i = 0; i < num_rows; i++) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0.0;

            for (int j = 0; j < num_cols; j++) {
                double item = matrix[i * num_cols + j];
                int col_label = col_labels[j];
                double y = cluster_avg[k * num_col_labels + col_label];
                dist += calculate_distance(y, item);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        if (row_labels[i] != best_label) {
            row_labels[i] = best_label;
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}

int main(int argc, const char* argv[]) {
    MPI_Init(&argc, const_cast<char***>(&argv));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;
    std::string output_file;

    int num_rows = 0;
    int num_cols = 0;
    int num_row_labels = 0;
    int num_col_labels = 0;
    int max_iter = 0;

    if (rank == 0) {
        if (!parse_arguments(
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
                &max_iter)) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        row_labels.resize(num_rows);
        col_labels.resize(num_cols);
        matrix.resize((size_t)num_rows * num_cols);
    }

    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_labels.data(), num_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix.data(), num_rows * num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

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
    int start_col = displs[rank];

    std::vector<label_type> local_col_labels(local_cols);
    for (int j = 0; j < local_cols; j++) {
        local_col_labels[j] = col_labels[start_col + j];
    }

    std::vector<float> local_matrix(num_rows * local_cols);
    for (int j = 0; j < local_cols; j++) {
        int gj = start_col + j;
        for (int i = 0; i < num_rows; i++) {
            local_matrix[j * num_rows + i] = matrix[i * num_cols + gj];
        }
    }

    float* d_matrix = nullptr;
    int* d_row_labels = nullptr;
    int* d_col_labels = nullptr;
    int* d_changes = nullptr;
    int* d_counts = nullptr;
    double* d_cluster_avg = nullptr;
    double* d_sums = nullptr;
    double* d_total_dist = nullptr;
    double* d_row_dist = nullptr;
    

    cudaCheck(cudaMalloc(&d_matrix, num_rows * local_cols * sizeof(float)));
    cudaCheck(cudaMalloc(&d_row_labels, num_rows * sizeof(int)));
    cudaCheck(cudaMalloc(&d_col_labels, local_cols * sizeof(int)));
    cudaCheck(cudaMalloc(&d_cluster_avg, num_row_labels * num_col_labels * sizeof(double)));
    cudaCheck(cudaMalloc(&d_sums, num_row_labels * num_col_labels * sizeof(double)));
    cudaCheck(cudaMalloc(&d_counts, num_row_labels * num_col_labels * sizeof(int)));
    cudaCheck(cudaMalloc(&d_changes, sizeof(int)));
    cudaCheck(cudaMalloc(&d_total_dist, sizeof(double)));
    cudaCheck(cudaMalloc(
    &d_row_dist,
    num_rows *
    num_row_labels *
    sizeof(double)));

    cudaCheck(cudaMemcpy(
        d_matrix,
        local_matrix.data(),
        num_rows * local_cols * sizeof(float),
        cudaMemcpyHostToDevice));

    auto before = std::chrono::high_resolution_clock::now();
    int completed_iters = 0;

    for (int iter = 0; iter < max_iter; iter++) {

        for (int j = 0; j < local_cols; j++) {
            local_col_labels[j] = col_labels[start_col + j];
        }

        cudaCheck(cudaMemcpy(
            d_row_labels,
            row_labels.data(),
            num_rows * sizeof(int),
            cudaMemcpyHostToDevice));

        cudaCheck(cudaMemcpy(
            d_col_labels,
            local_col_labels.data(),
            local_cols * sizeof(int),
            cudaMemcpyHostToDevice));

        cudaCheck(cudaMemset(
            d_sums,
            0,
            num_row_labels * num_col_labels * sizeof(double)));

        cudaCheck(cudaMemset(
            d_counts,
            0,
            num_row_labels * num_col_labels * sizeof(int)));

        calculate_local_sums_kernel<<<
            (local_cols + 255) / 256,
            256>>>(
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

        std::vector<double> h_sums(num_row_labels * num_col_labels);
        std::vector<int> h_counts(num_row_labels * num_col_labels);

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
            (int)h_sums.size(),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            MPI_IN_PLACE,
            h_counts.data(),
            (int)h_counts.size(),
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        std::vector<double> h_cluster_avg(num_row_labels * num_col_labels);
        for (size_t i = 0; i < h_cluster_avg.size(); i++) {
            h_cluster_avg[i] = h_sums[i] / (double)h_counts[i];
        }

        cudaCheck(cudaMemcpy(
            d_cluster_avg,
            h_cluster_avg.data(),
            h_cluster_avg.size() * sizeof(double),
            cudaMemcpyHostToDevice));


         cudaCheck(cudaMemset(
    d_row_dist,
    0,
    num_rows *
    num_row_labels *
    sizeof(double)));

compute_row_dist_parallel_kernel<<<
    (local_cols + 255) / 256,
    256>>>(
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

std::vector<double> h_row_dist(
    num_rows *
    num_row_labels);

cudaCheck(cudaMemcpy(
    h_row_dist.data(),
    d_row_dist,
    h_row_dist.size() *
    sizeof(double),
    cudaMemcpyDeviceToHost));

MPI_Allreduce(
    MPI_IN_PLACE,
    h_row_dist.data(),
    (int)h_row_dist.size(),
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD);

cudaCheck(cudaMemcpy(
    d_row_dist,
    h_row_dist.data(),
    h_row_dist.size() *
    sizeof(double),
    cudaMemcpyHostToDevice));


    cudaCheck(cudaMemset(
    d_changes,
    0,
    sizeof(int)));

update_row_labels_kernel<<<
    (num_rows + 255)/256,
    256>>>(
        num_rows,
        num_row_labels,
        d_row_dist,
        d_row_labels,
        d_changes);

cudaCheck(cudaGetLastError());
cudaCheck(cudaDeviceSynchronize());

int row_updates = 0;

cudaCheck(cudaMemcpy(
    &row_updates,
    d_changes,
    sizeof(int),
    cudaMemcpyDeviceToHost));

cudaCheck(cudaMemcpy(
    row_labels.data(),
    d_row_labels,
    num_rows * sizeof(int),
    cudaMemcpyDeviceToHost));


      
//
        cudaCheck(cudaMemcpy(
            d_row_labels,
            row_labels.data(),
            num_rows * sizeof(int),
            cudaMemcpyHostToDevice));

        cudaCheck(cudaMemcpy(
            d_col_labels,
            local_col_labels.data(),
            local_cols * sizeof(int),
            cudaMemcpyHostToDevice));

        int zero_i = 0;
        double zero_d = 0.0;
        cudaCheck(cudaMemcpy(d_changes, &zero_i, sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_total_dist, &zero_d, sizeof(double), cudaMemcpyHostToDevice));

        update_col_kernel<<<
            (local_cols + 255) / 256,
            256>>>(
                num_rows,
                local_cols,
                num_col_labels,
                d_matrix,
                d_row_labels,
                d_col_labels,
                d_cluster_avg,
                d_changes,
                d_total_dist);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        int col_updates_local = 0;
        double total_dist_col_local = 0.0;

        cudaCheck(cudaMemcpy(
            &col_updates_local,
            d_changes,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        cudaCheck(cudaMemcpy(
            &total_dist_col_local,
            d_total_dist,
            sizeof(double),
            cudaMemcpyDeviceToHost));

        cudaCheck(cudaMemcpy(
            local_col_labels.data(),
            d_col_labels,
            local_cols * sizeof(int),
            cudaMemcpyDeviceToHost));

        MPI_Allgatherv(
            local_col_labels.data(),
            local_cols,
            MPI_INT,
            col_labels.data(),
            counts.data(),
            displs.data(),
            MPI_INT,
            MPI_COMM_WORLD);

        int col_updates_global = 0;
        double total_dist_col_global = 0.0;

        MPI_Allreduce(
            &col_updates_local,
            &col_updates_global,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(
            &total_dist_col_local,
            &total_dist_col_global,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        completed_iters++;

        int total_updates = row_updates + col_updates_global;
       double average_dist =
    total_dist_col_global
    / (double)(num_rows * num_cols);

        if (rank == 0) {
            std::cout << "iteration " << completed_iters << ": "
                      << total_updates
                      << " labels were updated, average error is "
                      << average_dist << "\n";
        }

        if (total_updates == 0) {
            break;
        }
    }

    auto after = std::chrono::high_resolution_clock::now();
    double time_seconds = std::chrono::duration<double>(after - before).count();

    if (rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: "
                  << (time_seconds / std::max(completed_iters, 1))
                  << " seconds\n";
    }

    if (rank == 0) {
        write_labels(
            output_file,
            num_rows,
            num_cols,
            row_labels.data(),
            col_labels.data());
    }

    cudaFree(d_matrix);
    cudaFree(d_row_labels);
    cudaFree(d_col_labels);
    cudaFree(d_cluster_avg);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_changes);
    cudaFree(d_total_dist);
    cudaFree(d_row_dist);

    MPI_Finalize();
    return 0;
}
