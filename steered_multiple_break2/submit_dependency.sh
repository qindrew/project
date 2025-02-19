#!/bin/bash

# Initialize a variable to store the job ID of the previous job
prev_job_id=""

# Loop through folders from 1 to 10
for folder in {10..13}; do
    # Submit the job using sbatch
    if [ -z "$prev_job_id" ]; then
        # Submit the first job without dependency
        cd $folder
        job_id=$(sbatch "asubmit.sbatch" | awk '{print $4}')
        cd ..
    else
        # Submit subsequent jobs with dependency on the previous job
        cd $folder
        job_id=$(sbatch --dependency=afterany:$prev_job_id --job-name="2multi${folder}" "asubmit.sbatch" | awk '{print $4}')
        cd ..
    fi

    # Update prev_job_id with the current job ID
    prev_job_id=$job_id
    echo "Submitted job for folder $folder with Job ID $job_id"
done
