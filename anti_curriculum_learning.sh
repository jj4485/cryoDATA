#!/bin/bash

# Curriculum learning with cryodrgn for noisy particle datasets
#
# First batch of training
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind fifth_10000_indices.pkl -o anti_batch1

# Second batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind fourth_10000_indices.pkl --load anti_batch1/weights.pkl --num-epochs 40 -o anti_batch2

# Third batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind third_10000_indices.pkl --load anti_batch2/weights.pkl --num-epochs 60 -o anti_batch3

# Fourth batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind second_10000_indices.pkl --load anti_batch3/weights.pkl --num-epochs 80 -o anti_batch4

# Fifth and final batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind first_10000_indices.pkl --load anti_batch4/weights.pkl --num-epochs 100 -o anti_batch5
