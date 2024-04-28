#!/bin/bash

# Curriculum learning with cryodrgn for noisy particle datasets
#
# First batch of training
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind first_10000_indices.pkl -o batch1

# Second batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind second_10000_indices.pkl --load batch1/weights.pkl --num-epochs 40 -o batch2

# Third batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind third_10000_indices.pkl --load batch2/weights.pkl --num-epochs 60 -o batch3

# Fourth batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind fourth_10000_indices.pkl --load batch3/weights.pkl --num-epochs 80 -o batch4

# Fifth and final batch of training with previous weights
cryodrgn train_nn output_noisy_particles.mrcs --poses poses.pkl --ctf output_particles_w_ctf.mrcs.pkl --uninvert-data --ind fifth_10000_indices.pkl --load batch4/weights.pkl --num-epochs 100 -o batch5
