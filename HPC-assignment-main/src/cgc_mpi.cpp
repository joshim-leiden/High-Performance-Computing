#include <chrono>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <mpi.h>
#include "common.h"

// ---------------------------------------------------
// Calculate cluster averages in parallel (MPI)
// ---------------------------------------------------
std::vector<double> calculate_cluster_average(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels,
    int rank, int size)
{
    int rows_per_proc = num_rows / size;
    int extra_rows = num_rows % size;
    int start_row = rank * rows_per_proc + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);

    std::vector<double> local_sum(num_row_labels * num_col_labels, 0.0);
    std::vector<int> local_count(num_row_labels * num_col_labels, 0);

    for(int i = start_row; i < end_row; i++)
    {
        for(int j = 0; j < num_cols; j++)
        {
            int r = row_labels[i];
            int c = col_labels[j];
            local_sum[r * num_col_labels + c] += matrix[i * num_cols + j];
            local_count[r * num_col_labels + c] += 1;
        }
    }

    std::vector<double> global_sum(num_row_labels * num_col_labels);
    std::vector<int> global_count(num_row_labels * num_col_labels);

    MPI_Allreduce(local_sum.data(), global_sum.data(),
                  num_row_labels * num_col_labels,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_count.data(), global_count.data(),
                  num_row_labels * num_col_labels,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_row_labels * num_col_labels, 0.0);
    for(int i = 0; i < num_row_labels * num_col_labels; i++)
        if(global_count[i] > 0)
            cluster_avg[i] = global_sum[i] / global_count[i];

    return cluster_avg;
}

// ---------------------------------------------------
// Distance calculation (same as serial)
// ---------------------------------------------------
double calculate_distance(double avg, double item)
{
    double diff = avg - item;
    return diff * diff;
}

// ---------------------------------------------------
// Update row labels (parallel by rank)
// ---------------------------------------------------
std::pair<int, double> update_row_labels(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    const label_type* col_labels,
    const double* cluster_avg,
    int rank, int size)
{
    int rows_per_proc = num_rows / size;
    int extra_rows = num_rows % size;
    int start_row = rank * rows_per_proc + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);

    int local_updated = 0;
    double local_dist = 0.0;

    for(int i = start_row; i < end_row; i++)
    {
        int best_label = -1;
        double best_dist = INFINITY;

        for(int k = 0; k < num_row_labels; k++)
        {
            double dist = 0;
            for(int j = 0; j < num_cols; j++)
            {
                int col_label = col_labels[j];
                dist += calculate_distance(cluster_avg[k * num_col_labels + col_label],
                                           matrix[i * num_cols + j]);
            }
            if(dist < best_dist)
            {
                best_dist = dist;
                best_label = k;
            }
        }

        if(row_labels[i] != best_label)
        {
            row_labels[i] = best_label;
            local_updated++;
        }
        local_dist += best_dist;
    }

    int global_updated = 0;
    double global_dist = 0.0;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dist, &global_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return {global_updated, global_dist};
}

// ---------------------------------------------------
// Update column labels (parallel by column partitioning)
// ---------------------------------------------------
std::pair<int, double> update_col_labels(
    int num_rows, int num_cols,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    label_type* col_labels,
    const double* cluster_avg,
    int rank, int size)
{
    int cols_per_proc = num_cols / size;
    int extra_cols = num_cols % size;
    int start_col = rank * cols_per_proc + std::min(rank, extra_cols);
    int end_col = start_col + cols_per_proc + (rank < extra_cols ? 1 : 0);

    std::vector<int> local_labels(end_col - start_col);

    int local_updated = 0;
    double local_dist = 0.0;

    for(int j = start_col; j < end_col; j++)
    {
        int best_label = -1;
        double best_dist = INFINITY;

        for(int k = 0; k < num_col_labels; k++)
        {
            double dist = 0;
            for(int i = 0; i < num_rows; i++)
            {
                int r = row_labels[i];
                dist += calculate_distance(cluster_avg[r * num_col_labels + k],
                                           matrix[i * num_cols + j]);
            }
            if(dist < best_dist)
            {
                best_dist = dist;
                best_label = k;
            }
        }

        local_labels[j - start_col] = best_label;
        if(col_labels[j] != best_label)
            local_updated++;
        local_dist += best_dist;
    }

    // Gather all updated column labels to all ranks
    MPI_Allgather(local_labels.data(), end_col - start_col, MPI_INT,
                  col_labels, end_col - start_col, MPI_INT, MPI_COMM_WORLD);

    int global_updated = 0;
    double global_dist = 0.0;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dist, &global_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return {global_updated, global_dist};
}

// ---------------------------------------------------
// One iteration of co-clustering
// ---------------------------------------------------
std::pair<int, double> cluster_mpi_iteration(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int rank, int size)
{
    // MPI-aware cluster averages
    auto cluster_avg = calculate_cluster_average(num_rows, num_cols,
                                                 num_row_labels, num_col_labels,
                                                 matrix, row_labels, col_labels,
                                                 rank, size);

    auto [rows_updated, dist_rows] = update_row_labels(
        num_rows, num_cols, num_row_labels, num_col_labels,
        matrix, row_labels, col_labels, cluster_avg.data(), rank, size);

    auto [cols_updated, dist_cols] = update_col_labels(
        num_rows, num_cols, num_col_labels,
        matrix, row_labels, col_labels, cluster_avg.data(), rank, size);

    return {rows_updated + cols_updated, dist_rows + dist_cols};
}

// ---------------------------------------------------
// Repeat clustering until convergence
// ---------------------------------------------------
void cluster_mpi(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int max_iterations = 25,
    int rank = 0, int size = 1)
{
    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while(iteration < max_iterations)
    {
        auto [num_updated, total_dist] = cluster_mpi_iteration(
            num_rows, num_cols, num_row_labels, num_col_labels,
            matrix, row_labels, col_labels, rank, size);

        iteration++;

        if(rank == 0)
        {
            double avg_dist = total_dist / (num_rows * num_cols);
            std::cout << "Iteration " << iteration << ": "
                      << num_updated << " labels updated, average error "
                      << avg_dist << "\n";
        }

        if(num_updated == 0)
            break;
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds = std::chrono::duration<double>(after - before).count();

    if(rank == 0)
    {
        std::cout << "Total clustering time: " << time_seconds << " seconds\n";
        std::cout << "Time per iteration: " << (time_seconds / iteration) << " seconds\n";
    }
}

// ---------------------------------------------------
// Main function with MPI
// ---------------------------------------------------

int main(int argc, const char* argv[])
{
    MPI_Init(&argc, (char***)&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file;
    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;
    int num_rows = 0, num_cols = 0;
    int num_row_labels = 0, num_col_labels = 0;
    int max_iter = 25;

    if(!parse_arguments(argc, argv,
                        &num_rows, &num_cols,
                        &num_row_labels, &num_col_labels,
                        &matrix, &row_labels, &col_labels,
                        &output_file, &max_iter))
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Broadcast initial labels to all ranks
    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_labels.data(), num_cols, MPI_INT, 0, MPI_COMM_WORLD);

    cluster_mpi(num_rows, num_cols, num_row_labels, num_col_labels,
                matrix.data(), row_labels.data(), col_labels.data(),
                max_iter, rank, size);

    if(rank == 0)
    {
        write_labels(output_file, num_rows, num_cols,
                     row_labels.data(), col_labels.data());
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}