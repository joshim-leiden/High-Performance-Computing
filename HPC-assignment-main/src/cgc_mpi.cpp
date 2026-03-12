#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "common.h"


 // matrix column partitioning across mpi ranks - evenly

static void get_col_partition(
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


// computes avg value for each row-label and column-label cluster

std::vector<double> calculate_cluster_average(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels) {
    auto cluster_sum =
        std::vector<double>(num_row_labels * num_col_labels, 0.0);
    auto cluster_size = std::vector<int>(num_row_labels * num_col_labels, 0);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            auto item = matrix[i * num_cols + j];
            auto row_label = row_labels[i];
            auto col_label = col_labels[j];

            cluster_sum[row_label * num_col_labels + col_label] += item;
            cluster_size[row_label * num_col_labels + col_label] += 1;
        }
    }

    auto cluster_avg = std::vector<double>(num_row_labels * num_col_labels);

    for (int i = 0; i < num_row_labels; i++) {
        for (int j = 0; j < num_col_labels; j++) {
            auto index = i * num_col_labels + j;
            cluster_avg[index] =
                double(cluster_sum[index]) / double(cluster_size[index]);
        }
    }

    return cluster_avg;
}

// computes sq distance betn cluster avg and matrix value

double calculate_distance(double avg, double item) {
    double diff = (avg - item);
    return diff * diff;
}


// updates row labels on rank 0 by selecting the best cluster

std::pair<int, double> update_row_labels(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    const label_type* col_labels,
    const double* cluster_avg) {
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


 // updates column labels locally on each MPI rank   

std::pair<int, double> update_col_labels_local(
    int num_rows,
    int local_num_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* cluster_avg) {
    int num_updated = 0;
    double total_dist = 0;

    for (int j = 0; j < local_num_cols; j++) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0;

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

// performs iterative MPI co-clustering until convergence

void cluster_mpi(
    int num_rows,
    int num_cols,
    int local_num_cols,
    int num_row_labels,
    int num_col_labels,
    float* matrix,              
    float* local_matrix,
    label_type* row_labels,
    label_type* col_labels,     
    label_type* local_col_labels,
    int max_iterations,
    MPI_Comm comm) {
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // compute column partitioning for each rank

    std::vector<int> col_counts, col_displs;
    get_col_partition(num_cols, world_size, col_counts, col_displs);

    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    std::vector<double> cluster_avg(num_row_labels * num_col_labels, 0.0);

    while (iteration < max_iterations) {
        int num_rows_updated = 0;
        double total_dist_row = 0.0;

            // rank 0 computes cluster averages and updates row labels

        if (world_rank == 0) {
            cluster_avg = calculate_cluster_average(
                num_rows,
                num_cols,
                num_row_labels,
                num_col_labels,
                matrix,
                row_labels,
                col_labels);

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

        // broadcasts updated row labels and cluster averages

        MPI_Bcast(row_labels, num_rows, MPI_INT, 0, comm);
        MPI_Bcast(
            cluster_avg.data(),
            int(cluster_avg.size()),
            MPI_DOUBLE,
            0,
            comm);

        // updates column labels in parallel on each rank

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

        // sum column updates across ranks

        MPI_Reduce(
            &num_cols_updated_local,
            &num_cols_updated,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            comm);

          // sum distance across ranks

        MPI_Reduce(
            &total_dist_col_local,
            &total_dist_col,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            0,
            comm);

        // gather updated column labels back to rank 0

        MPI_Gatherv(
            local_col_labels,
            local_num_cols,
            MPI_INT,
            world_rank == 0 ? col_labels : nullptr,
            col_counts.data(),
            col_displs.data(),
            MPI_INT,
            0,
            comm);

        int num_updated = 0;
        double total_dist = 0.0;

        if (world_rank == 0) {
            num_updated = num_rows_updated + num_cols_updated;
            total_dist = total_dist_row + total_dist_col;
        }

         // broadcast convergence information

        MPI_Bcast(&num_updated, 1, MPI_INT, 0, comm);

        iteration++;

        if (world_rank == 0) {
            auto average_dist = total_dist / (num_rows * num_cols);
            std::cout << "iteration " << iteration << ": " << num_updated
                      << " labels were updated, average error is " << average_dist
                      << "\n";
        }

         // stop if labels stop changing
        if (num_updated == 0) {
            break;
        }
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds = std::chrono::duration<double>(after - before).count();

    if (world_rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: " << (time_seconds / iteration)
                  << " seconds\n";
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

    // parses argument only on rank 0
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

    // broadcasts scalar metadata
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

   
    if (world_rank != 0) {
        row_labels.resize(num_rows);
    }
    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);

    // partition columns

    std::vector<int> col_counts, col_displs;
    get_col_partition(num_cols, world_size, col_counts, col_displs);
    int local_num_cols = col_counts[world_rank];

    // scatters initial column labels

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

   
    std::vector<float> local_matrix(num_rows * local_num_cols);

   
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

    // run clustering
    cluster_mpi(
        num_rows,
        num_cols,
        local_num_cols,
        num_row_labels,
        num_col_labels,
        world_rank == 0 ? matrix.data() : nullptr,
        local_matrix.data(),
        row_labels.data(),
        world_rank == 0 ? col_labels.data() : nullptr,
        local_col_labels.data(),
        max_iter,
        MPI_COMM_WORLD);

    // write resulting labels
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