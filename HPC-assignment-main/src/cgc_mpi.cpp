#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h>

#include "common.h"

/**
 * Parallel cluster average calculation
 */
std::vector<double> calculate_cluster_average(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels,
    int rank,
    int size)
{
    std::vector<double> local_sum(num_row_labels * num_col_labels, 0.0);
    std::vector<int> local_count(num_row_labels * num_col_labels, 0);

    int rows_per_rank = num_rows / size;
    int start = rank * rows_per_rank;
    int end = (rank == size - 1) ? num_rows : start + rows_per_rank;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < num_cols; j++) {
            auto item = matrix[i * num_cols + j];
            auto row_label = row_labels[i];
            auto col_label = col_labels[j];

            local_sum[row_label * num_col_labels + col_label] += item;
            local_count[row_label * num_col_labels + col_label] += 1;
        }
    }

    std::vector<double> global_sum(num_row_labels * num_col_labels);
    std::vector<int> global_count(num_row_labels * num_col_labels);

    MPI_Allreduce(local_sum.data(),
                  global_sum.data(),
                  global_sum.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(local_count.data(),
                  global_count.data(),
                  global_count.size(),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_row_labels * num_col_labels);

    for (size_t i = 0; i < cluster_avg.size(); i++) {
        cluster_avg[i] =
            double(global_sum[i]) / double(global_count[i]);
    }

    return cluster_avg;
}

/**
 * Distance function
 */
double calculate_distance(double avg, double item) {
    double diff = (avg - item);
    return diff * diff;
}

/**
 * Row label update (serial, run only on rank 0)
 */
std::pair<int, double> update_row_labels(
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
    double total_dist = 0;

    for (int i = 0; i < num_rows; i++) {

        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_row_labels; k++) {

            double dist = 0;

            for (int j = 0; j < num_cols; j++) {

                double item = matrix[i * num_cols + j];

                int row_label = k;
                int col_label = col_labels[j];

                double y = cluster_avg[row_label * num_col_labels + col_label];

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

/**
 * Column label update (serial, run only on rank 0)
 */
std::pair<int, double> update_col_labels(
    int num_rows,
    int num_cols,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    label_type* col_labels,
    const double* cluster_avg)
{
    int num_updated = 0;
    double total_dist = 0;

    for (int j = 0; j < num_cols; j++) {

        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_col_labels; k++) {

            double dist = 0;

            for (int i = 0; i < num_rows; i++) {

                auto item = matrix[i * num_cols + j];

                auto row_label = row_labels[i];
                auto col_label = k;

                auto y = cluster_avg[row_label * num_col_labels + col_label];

                dist += calculate_distance(y, item);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        if (col_labels[j] != best_label) {
            col_labels[j] = best_label;
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}

/**
 * One iteration of clustering
 */
std::pair<int, double> cluster_mpi_iteration(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int rank,
    int size)
{
    auto cluster_avg = calculate_cluster_average(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix,
        row_labels,
        col_labels,
        rank,
        size);

    int num_rows_updated = 0;
    double total_dist_row = 0;

    if (rank == 0) {

        auto result = update_row_labels(
            num_rows,
            num_cols,
            num_row_labels,
            num_col_labels,
            matrix,
            row_labels,
            col_labels,
            cluster_avg.data());

        num_rows_updated = result.first;
        total_dist_row = result.second;
    }

    MPI_Bcast(row_labels, num_rows, MPI_INT, 0, MPI_COMM_WORLD);

    int num_cols_updated = 0;
    double total_dist_col = 0;

    if (rank == 0) {

        auto result = update_col_labels(
            num_rows,
            num_cols,
            num_col_labels,
            matrix,
            row_labels,
            col_labels,
            cluster_avg.data());

        num_cols_updated = result.first;
        total_dist_col = result.second;
    }

    MPI_Bcast(col_labels, num_cols, MPI_INT, 0, MPI_COMM_WORLD);

    int total_updated = num_rows_updated + num_cols_updated;
    double total_dist = total_dist_row + total_dist_col;

    return {total_updated, total_dist};
}

/**
 * Main clustering loop
 */
void cluster_mpi(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int max_iterations,
    int rank,
    int size)
{
    int iteration = 0;

    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {

        auto [num_updated, total_dist] = cluster_mpi_iteration(
            num_rows,
            num_cols,
            num_row_labels,
            num_col_labels,
            matrix,
            row_labels,
            col_labels,
            rank,
            size);

        iteration++;

        if (rank == 0) {

            auto average_dist = total_dist / (num_rows * num_cols);

            std::cout << "iteration " << iteration << ": "
                      << num_updated
                      << " labels were updated, average error is "
                      << average_dist << "\n";
        }

        if (num_updated == 0)
            break;
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds =
        std::chrono::duration<double>(after - before).count();

    if (rank == 0) {

        std::cout << "clustering time total: "
                  << time_seconds << " seconds\n";

        std::cout << "clustering time per iteration: "
                  << (time_seconds / iteration) << " seconds\n";
    }
}

/**
 * Main
 */
int main(int argc, const char* argv[])
{
    MPI_Init(&argc, (char***)&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file;
    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;

    int num_rows = 0;
    int num_cols = 0;
    int num_row_labels = 0;
    int num_col_labels = 0;
    int max_iter = 0;

    auto before = std::chrono::high_resolution_clock::now();

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

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cluster_mpi(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix.data(),
        row_labels.data(),
        col_labels.data(),
        max_iter,
        rank,
        size);

    if (rank == 0) {

        write_labels(
            output_file,
            num_rows,
            num_cols,
            row_labels.data(),
            col_labels.data());
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds =
        std::chrono::duration<double>(after - before).count();

    if (rank == 0)
        std::cout << "total execution time: "
                  << time_seconds << " seconds\n";

    MPI_Finalize();

    return EXIT_SUCCESS;
}