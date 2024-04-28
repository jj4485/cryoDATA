import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from cryodrgn import mrc, dataset
from cryodrgn.lattice import EvenLattice

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mrcs', help='Input particles (.mrcs, .star, or .txt)')
    #parser.add_argument('--snr', type=float)
    #parser.add_argument('--sigma', type=float)
    parser.add_argument('--mask', choices=('none','strict','circular'), help='Type of mask for computing signal variance')
    parser.add_argument('--mask-r', type=int, help='Radius for circular mask')
    parser.add_argument('--datadir', help='Optionally overwrite path to starfile .mrcs if loading from a starfile')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output particle stack')
    parser.add_argument('--out-png')
    return parser

def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)

def mkbasedir(out):
    os.makedirs(os.path.dirname(out), exist_ok=True)

def main(args):
    mkbasedir(args.o)
    particles = dataset.load_particles(args.mrcs)
    print(f"Loaded particles with shape: {particles.shape}")
    Nimg, D, _ = particles.shape
    
    # SNR segments and their corresponding SNR values
    snr_values = [1.0, 0.1778, 0.0316, 0.0056, 0.001]
    particles_per_snr = 10000
    assert Nimg >= particles_per_snr * len(snr_values)

    shuffled_indices = np.random.permutation(Nimg)
    
    for i, snr in enumerate(snr_values):
        print("randomly adding noise")
        start_index = i * particles_per_snr
        end_index = start_index + particles_per_snr
        selected_indices = shuffled_indices[start_index:end_index]
        segment = particles[selected_indices]
        std = np.std(segment, axis=(1, 2), keepdims=True)
        sigma = std/np.sqrt(snr)
        noise = np.random.normal(0, sigma, segment.shape)
        particles[selected_indices] += noise
        print(f"SNR: {snr}, Std Dev Added: {sigma.mean()}")  # Printing the mean standard deviation of the added noise
    

    mrc.write(args.o, particles.astype(np.float32))
    
    if args.out_png:
        plot_projections(args.out_png, particles[:9])
    
    print('All processing completed.')

if __name__ == '__main__':
    main(parse_args().parse_args())
