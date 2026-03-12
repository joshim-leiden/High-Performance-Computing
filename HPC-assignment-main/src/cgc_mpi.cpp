
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <mpi.h>

#include "common.h"

/**
 * Partition columns nearly evenly across MPI ranks.
 */
static inline void get_col_partition(
    int num_cols,
    int world_size,
    std::vector<int>& counts,
    std::vector<int>& displs) {
    counts.resize(world_size);
    displs.resize(world_size);

    int base = num_cols / world_size;
    int rem = num_cols % world_size;

    int offset = 0;
    for (int r = 0; r < world_size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

/**
 * Local matrix storage is column-major by local column:
 * local_matrix[j_local * num_rows + i]
 *
 * This function returns a matrix of size (num_row_labels, num_col_labels)
 * that stores the average value for each combination of row label and
 * column label across all MPI ranks.
 */
std::vector<double> calculate_cluster_average_mpi(
    int num_rows,
    int local_num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    const label_type* local_col_labels,
    MPI_Comm comm) {
    auto local_cluster_sum =
        std::vector<double>(num_row_labels * num_col_labels, 0.0);
    auto global_cluster_sum =
        std::vector<double>(num_row_labels * num_col_labels, 0.0);

    auto local_cluster_size =
        std::vector<long long>(num_row_labels * num_col_labels, 0);
    auto global_cluster_size =
        std::vector<long long>(num_row_labels * num_col_labels, 0);

    for (int j = 0; j < local_num_cols; j++) {
        auto col_label = local_col_labels[j];
        for (int i = 0; i < num_rows; i++) {
            auto item = local_matrix[j * num_rows + i];
            auto row_label = row_labels[i];
            auto index = row_label * num_col_labels + col_label;

            local_cluster_sum[index] += item;
            local_cluster_size[index] += 1;
        }
    }

    MPI_Allreduce(
        local_cluster_sum.data(),
        global_cluster_sum.data(),
        int(global_cluster_sum.size()),
        MPI_DOUBLE,
        MPI_SUM,
        comm);

    MPI_Allreduce(
        local_cluster_size.data(),
        global_cluster_size.data(),
        int(global_cluster_size.size()),
        MPI_LONG_LONG,
        MPI_SUM,
        comm);

    auto cluster_avg = std::vector<double>(num_row_labels * num_col_labels, 0.0);

    for (int i = 0; i < num_row_labels; i++) {
        for (int j = 0; j < num_col_labels; j++) {
            auto index = i * num_col_labels + j;
            if (global_cluster_size[index] > 0) {
                cluster_avg[index] =
                    global_cluster_sum[index] / double(global_cluster_size[index]);
            } else {
                cluster_avg[index] = 0.0;
            }
        }
    }

    return cluster_avg;
}

double calculate_distance(double avg, double item) {
    double diff = (avg - item);
    return diff * diff;
}

/**
 * Update the labels along the rows of the matrix in parallel.
 * Each rank computes partial distances for its local columns, then all ranks
 * combine them using MPI_Allreduce.
 *
 * This function returns both the number of rows that changed their label and
 * the total distance. If the first return value is zero, then no row was updated.
 */
std::pair<int, double> update_row_labels_mpi(
    int num_rows,
    int local_num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    const label_type* local_col_labels,
    const double* cluster_avg,
    MPI_Comm comm) {
    auto local_distances =
        std::vector<double>(num_rows * num_row_labels, 0.0);
    auto global_distances =
        std::vector<double>(num_rows * num_row_labels, 0.0);

    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0.0;

            for (int j = 0; j < local_num_cols; j++) {
                double item = local_matrix[j * num_rows + i];

                int row_label = k;
                int col_label = local_col_labels[j];
                double y = cluster_avg[row_label * num_col_labels + col_label];

                dist += calculate_distance(y, item);
            }

            local_distances[i * num_row_labels + k] = dist;
        }
    }

    MPI_Allreduce(
        local_distances.data(),
        global_distances.data(),
        int(global_distances.size()),
        MPI_DOUBLE,
        MPI_SUM,
        comm);

    int num_updated = 0;
    double total_dist = 0.0;

    for (int i = 0; i < num_rows; i++) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_row_labels; k++) {
            double dist = global_distances[i * num_row_labels + k];

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
 * Update the labels along the columns of the matrix.
 * Each MPI rank updates only its local columns.
 *
 * This function returns the number of local columns that changed their label
 * and the local total distance.
 */
std::pair<int, double> update_col_labels_local(
    int num_rows,
    int local_num_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* cluster_avg) {
    int num_updated = 0;
    double total_dist = 0.0;

    for (int j = 0; j < local_num_cols; j++) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0.0;

            for (int i = 0; i < num_rows; i++) {
                auto item = local_matrix[j * num_rows + i];

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

        if (local_col_labels[j] != best_label) {
            local_col_labels[j] = best_label;
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}

/**
 * Perform one iteration of the co-clustering algorithm in parallel.
 * This function updates the labels in both `row_labels` and `local_col_labels`,
 * and returns the total number of labels that changed across all ranks.
 */
std::pair<int, double> cluster_mpi_iteration(
    int num_rows,
    int local_num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    label_type* local_col_labels,
    MPI_Comm comm) {
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    // Calculate the average value per cluster
    auto cluster_avg = calculate_cluster_average_mpi(
        num_rows,
        local_num_cols,
        num_row_labels,
        num_col_labels,
        local_matrix,
        row_labels,
        local_col_labels,
        comm);

    // Update labels along the rows (global)
    auto [num_rows_updated, total_dist_row] = update_row_labels_mpi(
        num_rows,
        local_num_cols,
        num_row_labels,
        num_col_labels,
        local_matrix,
        row_labels,
        local_col_labels,
        cluster_avg.data(),
        comm);

    // Update labels along the columns (local)
    auto [num_cols_updated_local, total_dist_col_local] = update_col_labels_local(
        num_rows,
        local_num_cols,
        num_col_labels,
        local_matrix,
        row_labels,
        local_col_labels,
        cluster_avg.data());

    int num_cols_updated = 0;
    double total_dist_col = 0.0;

    MPI_Allreduce(
        &num_cols_updated_local,
        &num_cols_updated,
        1,
        MPI_INT,
        MPI_SUM,
        comm);

    MPI_Allreduce(
        &total_dist_col_local,
        &total_dist_col,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        comm);

    return {num_rows_updated + num_cols_updated, total_dist_row + total_dist_col};
}

/**
 * Repeatedly calls `cluster_mpi_iteration` to iteratively update the
 * labels along the rows and columns. This function performs `max_iterations`
 * iterations or until convergence.
 */
void cluster_mpi(
    int num_rows,
    int num_cols,
    int local_num_cols,
    int num_row_labels,
    int num_col_labels,
    float* local_matrix,
    label_type* row_labels,
    label_type* local_col_labels,
    int max_iterations,
    MPI_Comm comm) {
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {
        auto [num_updated, total_dist] = cluster_mpi_iteration(
            num_rows,
            local_num_cols,
            num_row_labels,
            num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            comm);

        iteration++;

        if (world_rank == 0) {
            auto average_dist = total_dist / (double(num_rows) * double(num_cols));
            std::cout << "iteration " << iteration << ": " << num_updated
                      << " labels were updated, average error is " << average_dist
                      << "\n";
        }

        if (num_updated == 0) {
            break;
        }
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds = std::chrono::duration<double>(after - before).count();

    if (world_rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: "
                  << (time_seconds / iteration) << " seconds\n";
    }
}

int main(int argc, const char* argv[]) {
    MPI_Init(&argc, const_cast<char***>(&argv));

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string output_file;
    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;
    int num_rows = 0, num_cols = 0;
    int num_row_labels = 0, num_col_labels = 0;
    int max_iter = 0;

    auto before = std::chrono::high_resolution_clock::now();

    int parse_ok = 1;

    // Parse arguments only on rank 0
    if (world_rank == 0) {
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
            parse_ok = 0;
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Broadcast scalar metadata
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast row labels
    if (world_rank != 0) {
        row_labels.resize(num_rows);
    }
    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute column partition
    std::vector<int> col_counts, col_displs;
    get_col_partition(num_cols, world_size, col_counts, col_displs);

    int local_num_cols = col_counts[world_rank];

    // Scatter column labels
    std::vector<label_type> local_col_labels(local_num_cols);
    MPI_Scatterv(
        world_rank == 0 ? col_labels.data() : nullptr,
        col_counts.data(),
        col_displs.data(),
        MPI_INT,
        local_col_labels.data(),
        local_num_cols,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    // Prepare local matrix storage: local columns, each column contiguous in rows
    std::vector<float> local_matrix(num_rows * local_num_cols);

    // Root converts from row-major global matrix to column-major-by-column buffer
    std::vector<float> matrix_col_major;
    std::vector<int> send_counts_matrix, send_displs_matrix;

    if (world_rank == 0) {
        matrix_col_major.resize(num_rows * num_cols);

        for (int j = 0; j < num_cols; j++) {
            for (int i = 0; i < num_rows; i++) {
                matrix_col_major[j * num_rows + i] = matrix[i * num_cols + j];
            }
        }

        send_counts_matrix.resize(world_size);
        send_displs_matrix.resize(world_size);
        for (int r = 0; r < world_size; r++) {
            send_counts_matrix[r] = col_counts[r] * num_rows;
            send_displs_matrix[r] = col_displs[r] * num_rows;
        }
    }

    MPI_Scatterv(
        world_rank == 0 ? matrix_col_major.data() : nullptr,
        world_rank == 0 ? send_counts_matrix.data() : nullptr,
        world_rank == 0 ? send_displs_matrix.data() : nullptr,
        MPI_FLOAT,
        local_matrix.data(),
        num_rows * local_num_cols,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    // Cluster labels
    cluster_mpi(
        num_rows,
        num_cols,
        local_num_cols,
        num_row_labels,
        num_col_labels,
        local_matrix.data(),
        row_labels.data(),
        local_col_labels.data(),
        max_iter,
        MPI_COMM_WORLD);

    // Gather resulting column labels back to rank 0
    if (world_rank == 0) {
        col_labels.resize(num_cols);
    }

    MPI_Gatherv(
        local_col_labels.data(),
        local_num_cols,
        MPI_INT,
        world_rank == 0 ? col_labels.data() : nullptr,
        col_counts.data(),
        col_displs.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    // Write resulting labels
    if (world_rank == 0) {
        write_labels(
            output_file,
            num_rows,
            num_cols,
            row_labels.data(),
            col_labels.data());

        auto after = std::chrono::high_resolution_clock::now();
        auto time_seconds = std::chrono::duration<double>(after - before).count();

        std::cout << "total execution time: " << time_seconds << " seconds\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}