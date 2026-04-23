__global__ void update_col_labels_kernel(
    int num_rows,
    int local_num_cols,
    int num_col_labels,
    const float* local_matrix,
    const int* row_labels,
    int* local_col_labels,
    const double* cluster_avg,
    int* d_num_updated,
    double* d_total_dist
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= local_num_cols) return;

    int best_label = -1;
    double best_dist = INFINITY;

    for (int k = 0; k < num_col_labels; k++) {
        double dist = 0;

        for (int i = 0; i < num_rows; i++) {
            float item = local_matrix[j * num_rows + i];

            int row_label = row_labels[i];
            int col_label = k;
            double y = cluster_avg[row_label * num_col_labels + col_label];

            double diff = y - item;
            dist += diff * diff;
        }

        if (dist < best_dist) {
            best_dist = dist;
            best_label = k;
        }
    }

    if (local_col_labels[j] != best_label) {
        atomicAdd(d_num_updated, 1);
        local_col_labels[j] = best_label;
    }

    atomicAdd(d_total_dist, best_dist);
}