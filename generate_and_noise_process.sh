#!/bin/bash

# Define the input volume and output base directory
input_volume="data/volume.mrc"
output_base_dir="/scratch/network/jj4485/cryoDATA/cryoDATA"

# Generate 10k projection images of a volume
python project3d.py $input_volume -N 10000 -o ${output_base_dir}/output_projections.mrcs --out-pose ${output_base_dir}/poses.pkl --t-extent 10


snr_values=(1 0.1778 0.0316 0.0056 0.001)

# Loop to process at each SNR level
for snr in "${snr_values[@]}"
do
  
  python add_ctf.py ${output_base_dir}/output_projections.mrcs --Apix 6 --s1 0 --s2 0 -o ${output_base_dir}/output_particles_w_ctf_${snr}.mrcs
  
  # Add gaussian noise to the current SNR level
  python add_noise.py ${output_base_dir}/output_particles_w_ctf_${snr}.mrcs -o ${output_base_dir}/output_noisy_particles_${snr}.mrcs --snr $snr

  echo "Processed SNR level $snr"
done

echo "All processes completed."
