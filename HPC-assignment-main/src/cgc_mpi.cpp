#include <chrono>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <mpi.h>
#include "common.h"

// ---------------------------------------------------
double calculate_distance(double avg, double item) {
    double diff = avg - item;
    return diff * diff;
}

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
    int extra = num_rows % size;

    int start = rank * rows_per_proc + std::min(rank, extra);
    int end = start + rows_per_proc + (rank < extra);

    std::vector<double> local_sum(num_row_labels * num_col_labels, 0.0);
    std::vector<int> local_count(num_row_labels * num_col_labels, 0);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < num_cols; j++) {

            int r = row_labels[i];
            int c = col_labels[j];

            local_sum[r * num_col_labels + c] += matrix[i * num_cols + j];
            local_count[r * num_col_labels + c] += 1;
        }
    }

    std::vector<double> global_sum(num_row_labels * num_col_labels);
    std::vector<int> global_count(num_row_labels * num_col_labels);

    MPI_Allreduce(local_sum.data(), global_sum.data(),
                  global_sum.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(local_count.data(), global_count.data(),
                  global_count.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_row_labels * num_col_labels);

    for (size_t i = 0; i < cluster_avg.size(); i++)
        if (global_count[i] > 0)
            cluster_avg[i] = global_sum[i] / global_count[i];

    return cluster_avg;
}

// ---------------------------------------------------
std::pair<int,double> update_row_labels(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    const label_type* col_labels,
    const double* cluster_avg,
    int rank, int size)
{
    int rows_per_proc = num_rows / size;
    int extra = num_rows % size;

    int start = rank * rows_per_proc + std::min(rank, extra);
    int count = rows_per_proc + (rank < extra);

    int end = start + count;

    int local_updated = 0;
    double local_dist = 0;

    std::vector<label_type> local_labels(count);

    for (int i = start; i < end; i++) {

        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_row_labels; k++) {

            double dist = 0;

            for (int j = 0; j < num_cols; j++) {

                int c = col_labels[j];

                dist += calculate_distance(
                    cluster_avg[k * num_col_labels + c],
                    matrix[i * num_cols + j]);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        local_labels[i - start] = best_label;

        if (row_labels[i] != best_label)
            local_updated++;

        local_dist += best_dist;
    }

    int global_updated;
    double global_dist;

    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dist, &global_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int r=0; r<size; r++) {
        recvcounts[r] = rows_per_proc + (r < extra);
        displs[r] = r * rows_per_proc + std::min(r, extra);
    }

    MPI_Allgatherv(local_labels.data(),
                   count,
                   MPI_INT,
                   row_labels,
                   recvcounts.data(),
                   displs.data(),
                   MPI_INT,
                   MPI_COMM_WORLD);

    return {global_updated, global_dist};
}

// ---------------------------------------------------
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
    int extra = num_cols % size;

    int start = rank * cols_per_proc + std::min(rank, extra);
    int count = cols_per_proc + (rank < extra);

    int end = start + count;

    int local_updated = 0;
    double local_dist = 0;

    std::vector<label_type> local_labels(count);

    for (int j = start; j < end; j++) {

        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_col_labels; k++) {

            double dist = 0;

            for (int i = 0; i < num_rows; i++) {

                int r = row_labels[i];

                dist += calculate_distance(
                    cluster_avg[r * num_col_labels + k],
                    matrix[i * num_cols + j]);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        local_labels[j - start] = best_label;

        if (col_labels[j] != best_label)
            local_updated++;

        local_dist += best_dist;
    }

    int global_updated;
    double global_dist;

    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dist, &global_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int r=0; r<size; r++) {
        recvcounts[r] = cols_per_proc + (r < extra);
        displs[r] = r * cols_per_proc + std::min(r, extra);
    }

    MPI_Allgatherv(local_labels.data(),
                   count,
                   MPI_INT,
                   col_labels,
                   recvcounts.data(),
                   displs.data(),
                   MPI_INT,
                   MPI_COMM_WORLD);

    return {global_updated, global_dist};
}

std::pair<int, double> cluster_mpi_iteration(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int rank, int size)
{
    // averages based on current labels
    auto cluster_avg = calculate_cluster_average(
        num_rows, num_cols,
        num_row_labels, num_col_labels,
        matrix, row_labels, col_labels,
        rank, size);

    // update rows
    auto [rows_updated, dist_rows] = update_row_labels(
        num_rows, num_cols,
        num_row_labels, num_col_labels,
        matrix, row_labels, col_labels,
        cluster_avg.data(), rank, size);

    // recompute averages after row updates
    cluster_avg = calculate_cluster_average(
        num_rows, num_cols,
        num_row_labels, num_col_labels,
        matrix, row_labels, col_labels,
        rank, size);

    // update columns with fresh averages
    auto [cols_updated, dist_cols] = update_col_labels(
        num_rows, num_cols,
        num_col_labels,
        matrix, row_labels, col_labels,
        cluster_avg.data(), rank, size);

    // recompute averages again after column updates for next iteration
    cluster_avg = calculate_cluster_average(
        num_rows, num_cols,
        num_row_labels, num_col_labels,
        matrix, row_labels, col_labels,
        rank, size);

    return {rows_updated + cols_updated, dist_rows + dist_cols};
}

// ---------------------------------------------------
void cluster_mpi(
    int num_rows, int num_cols,
    int num_row_labels, int num_col_labels,
    float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int max_iterations,
    int rank, int size)
{
    int iter=0;

    auto before=std::chrono::high_resolution_clock::now();

    while(iter<max_iterations){

        auto [updated,dist]=cluster_mpi_iteration(
            num_rows,num_cols,
            num_row_labels,num_col_labels,
            matrix,row_labels,col_labels,
            rank,size);

        iter++;

        if(rank==0){
            std::cout<<"Iteration "<<iter
                     <<" : "<<updated
                     <<" labels updated\n";
        }

        if(updated==0) break;
    }

    auto after=std::chrono::high_resolution_clock::now();
    double t=std::chrono::duration<double>(after-before).count();

    if(rank==0){
        std::cout<<"Total time "<<t<<" sec\n";
    }
}

// ---------------------------------------------------
int main(int argc,const char* argv[])
{
    MPI_Init(&argc,(char***)&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    std::string output_file;

    std::vector<float> matrix;
    std::vector<label_type> row_labels;
    std::vector<label_type> col_labels;

    int num_rows=0;
    int num_cols=0;
    int num_row_labels=0;
    int num_col_labels=0;
    int max_iter=25;

    if(!parse_arguments(
        argc,argv,
        &num_rows,&num_cols,
        &num_row_labels,&num_col_labels,
        &matrix,&row_labels,&col_labels,
        &output_file,&max_iter))
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cluster_mpi(
        num_rows,num_cols,
        num_row_labels,num_col_labels,
        matrix.data(),
        row_labels.data(),
        col_labels.data(),
        max_iter,
        rank,size);

    if(rank==0)
        write_labels(output_file,
                     num_rows,num_cols,
                     row_labels.data(),
                     col_labels.data());

    MPI_Finalize();
}