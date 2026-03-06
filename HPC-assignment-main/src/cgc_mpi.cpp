#include <chrono>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <mpi.h>
#include "common.h"


// Cluster average calculation 

std::vector<double> calculate_cluster_average(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels)
{
    std::vector<double> cluster_sum(num_row_labels * num_col_labels, 0.0);
    std::vector<int> cluster_size(num_row_labels * num_col_labels, 0);

    for(int i = 0; i < num_rows; i++)
    {
        for(int j = 0; j < num_cols; j++)
        {
            auto item = matrix[i * num_cols + j];
            auto row_label = row_labels[i];
            auto col_label = col_labels[j];
            cluster_sum[row_label * num_col_labels + col_label] += item;
            cluster_size[row_label * num_col_labels + col_label] += 1;
        }
    }

    std::vector<double> cluster_avg(num_row_labels * num_col_labels, 0.0);
    for(int i = 0; i < num_row_labels; i++)
    {
        for(int j = 0; j < num_col_labels; j++)
        {
            auto index = i * num_col_labels + j;
            if(cluster_size[index] > 0)
                cluster_avg[index] = cluster_sum[index] / cluster_size[index];
        }
    }
    return cluster_avg;
}

// ---------------------------------------------------
// Distance calculation (same as serial)
// ---------------------------------------------------
double calculate_distance(double avg, double item)
{
    double diff = (avg - item);
    return diff * diff;
}

// ---------------------------------------------------
// Update row labels (parallelized by row partitioning)
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
                double y = cluster_avg[k * num_col_labels + col_label];
                dist += calculate_distance(y, matrix[i * num_cols + j]);
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
// Update column labels (same as serial, allreduce)
// ---------------------------------------------------
std::pair<int, double> update_col_labels(
    int num_rows, int num_cols,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    label_type* col_labels,
    const double* cluster_avg)
{
    int num_updated = 0;
    double total_dist = 0.0;

    for(int j = 0; j < num_cols; j++)
    {
        int best_label = -1;
        double best_dist = INFINITY;

        for(int k = 0; k < num_col_labels; k++)
        {
            double dist = 0;
            for(int i = 0; i < num_rows; i++)
            {
                int row_label = row_labels[i];
                dist += calculate_distance(cluster_avg[row_label * num_col_labels + k],
                                           matrix[i * num_cols + j]);
            }
            if(dist < best_dist)
            {
                best_dist = dist;
                best_label = k;
            }
        }

        if(col_labels[j] != best_label)
        {
            col_labels[j] = best_label;
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
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
    // calculate cluster averages (serial part)
    auto cluster_avg = calculate_cluster_average(
        num_rows, num_cols, num_row_labels, num_col_labels,
        matrix, row_labels, col_labels);

    auto [rows_updated, dist_rows] = update_row_labels(
        num_rows, num_cols, num_row_labels, num_col_labels,
        matrix, row_labels, col_labels, cluster_avg.data(), rank, size);

    auto [cols_updated, dist_cols] = update_col_labels(
        num_rows, num_cols, num_col_labels,
        matrix, row_labels, col_labels, cluster_avg.data());

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