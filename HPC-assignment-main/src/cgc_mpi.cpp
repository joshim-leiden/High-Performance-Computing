#include <chrono>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <mpi.h>
#include "common.h"

double calculate_distance(double avg, double item) {
    double diff = avg - item;
    return diff * diff;
}

/*
Compute cluster averages
*/
std::vector<double> calculate_cluster_average(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels,
    int rank, int size)
{
    int rows_per_proc = num_rows / size;
    int extra = num_rows % size;
    int start = rank * rows_per_proc + std::min(rank, extra);
    int count = rows_per_proc + (rank < extra ? 1 : 0);
    int end = start + count;

    int num_clusters = num_row_labels * num_col_labels;

    std::vector<double> local_sum(num_clusters, 0.0);
    std::vector<int> local_count(num_clusters, 0);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < num_cols; j++) {
            int r = row_labels[i];
            int c = col_labels[j];
            int idx = r * num_col_labels + c;
            local_sum[idx] += matrix[i * num_cols + j];
            local_count[idx] += 1;
        }
    }

    std::vector<double> global_sum(num_clusters);
    std::vector<int> global_count(num_clusters);

    MPI_Allreduce(local_sum.data(), global_sum.data(), num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_count.data(), global_count.data(), num_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_clusters, 0.0);
    for (int i = 0; i < num_clusters; i++)
        if (global_count[i] > 0)
            cluster_avg[i] = global_sum[i] / global_count[i];

    return cluster_avg;
}

/*
Update row labels parallelized by columns (correct, exact)
*/
std::pair<int,double> update_row_labels(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    const label_type* col_labels,
    const double* cluster_avg,
    int rank, int size)
{
    int cols_per_proc = num_cols / size;
    int extra_cols = num_cols % size;
    int start_col = rank * cols_per_proc + std::min(rank, extra_cols);
    int count_col = cols_per_proc + (rank < extra_cols ? 1 : 0);
    int end_col = start_col + count_col;

    int local_updated = 0;
    double local_dist = 0.0;

    for (int i = 0; i < num_rows; i++) {
        std::vector<double> local_partial_dist(num_row_labels, 0.0);

        // Compute partial distances over assigned columns
        for (int j = start_col; j < end_col; j++) {
            int c = col_labels[j];
            for (int k = 0; k < num_row_labels; k++)
                local_partial_dist[k] += calculate_distance(cluster_avg[k * num_col_labels + c], matrix[i * num_cols + j]);
        }

        // Sum partial distances globally across ranks
        std::vector<double> global_dist(num_row_labels, 0.0);
        MPI_Allreduce(local_partial_dist.data(), global_dist.data(), num_row_labels, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Pick best label after global sum
        int best_label = 0;
        double best_dist = global_dist[0];
        for (int k = 1; k < num_row_labels; k++) {
            if (global_dist[k] < best_dist) {
                best_dist = global_dist[k];
                best_label = k;
            }
        }

        if (row_labels[i] != best_label) local_updated++;
        local_dist += best_dist;
        row_labels[i] = best_label;  // update label globally
    }

    int global_updated;
    double global_dist;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dist, &global_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return {global_updated, global_dist};
}

/*
Update column labels parallelized by columns
*/
std::pair<int,double> update_col_labels(
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
    int count_col = cols_per_proc + (rank < extra_cols ? 1 : 0);
    int end_col = start_col + count_col;

    int local_updated = 0;
    double local_dist = 0.0;
    std::vector<label_type> local_labels(count_col);

    for (int j = start_col; j < end_col; j++) {
        int best_label = -1;
        double best_dist = INFINITY;
        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0.0;
            for (int i = 0; i < num_rows; i++) {
                int r = row_labels[i];
                dist += calculate_distance(cluster_avg[r * num_col_labels + k], matrix[i * num_cols + j]);
            }
            if (dist < best_dist) { best_dist = dist; best_label = k; }
        }
        local_labels[j - start_col] = best_label;
        if (col_labels[j] != best_label) local_updated++;
        local_dist += best_dist;
    }

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; r++) recvcounts[r] = cols_per_proc + (r < extra_cols ? 1 : 0);
    displs[0] = 0;
    for (int r = 1; r < size; r++) displs[r] = displs[r-1] + recvcounts[r-1];

    MPI_Allgatherv(local_labels.data(), count_col, MPI_INT, col_labels, recvcounts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    int global_updated;
    double global_dist;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dist, &global_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return {global_updated, global_dist};
}

/*
One iteration of MPI clustering
*/
std::pair<int,double> cluster_mpi_iteration(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int rank, int size)
{
    auto cluster_avg = calculate_cluster_average(num_rows, num_cols, num_row_labels, num_col_labels, matrix, row_labels, col_labels, rank, size);
    auto [rows_updated, dist_rows] = update_row_labels(num_rows, num_cols, num_row_labels, num_col_labels, matrix, row_labels, col_labels, cluster_avg.data(), rank, size);
    cluster_avg = calculate_cluster_average(num_rows, num_cols, num_row_labels, num_col_labels, matrix, row_labels, col_labels, rank, size);
    auto [cols_updated, dist_cols] = update_col_labels(num_rows, num_cols, num_col_labels, matrix, row_labels, col_labels, cluster_avg.data(), rank, size);
    return {rows_updated + cols_updated, dist_rows + dist_cols};
}

/*
Main MPI clustering loop
*/
void cluster_mpi(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int max_iterations,
    int rank, int size)
{
    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {
        auto [updated, dist] = cluster_mpi_iteration(num_rows, num_cols, num_row_labels, num_col_labels, matrix, row_labels, col_labels, rank, size);
        iteration++;
        if (rank == 0) {
            double avg_dist = dist / (num_rows * num_cols);
            std::cout << "iteration " << iteration << ": " << updated << " labels updated, average error " << avg_dist << "\n";
        }
        if (updated == 0) break;
    }

    auto after = std::chrono::high_resolution_clock::now();
    if (rank == 0)
        std::cout << "clustering time total: " << std::chrono::duration<double>(after - before).count() << " seconds\n";
}

int main(int argc, const char* argv[]) {
    MPI_Init(&argc,(char***)&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    std::string output_file;
    std::vector<float> matrix;
    std::vector<label_type> row_labels;
    std::vector<label_type> col_labels;
    int num_rows=0, num_cols=0, num_row_labels=0, num_col_labels=0, max_iter=25;

    if(!parse_arguments(argc, argv, &num_rows, &num_cols, &num_row_labels, &num_col_labels, &matrix, &row_labels, &col_labels, &output_file, &max_iter)) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cluster_mpi(num_rows, num_cols, num_row_labels, num_col_labels, matrix.data(), row_labels.data(), col_labels.data(), max_iter, rank, size);

    if(rank==0)
        write_labels(output_file, num_rows, num_cols, row_labels.data(), col_labels.data());

    MPI_Finalize();
}