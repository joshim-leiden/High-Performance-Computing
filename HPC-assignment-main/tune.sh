#!/bin/bash

for BS in 64 128 256 512
do
    echo "Testing BLOCK_SIZE=$BS"

    make clean
    make cgc_cuda_bonus BLOCK_SIZE=$BS

    sbatch run_cuda_bonus.job

    echo "Finished BLOCK_SIZE=$BS"
done

# clustering time total
# block 64- 2.13793 seconds
#128- 2.15271 seconds
#256-  3.45888 seconds
#512-  3.79084 seconds
