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
    snr_segments = [(0, 10000, 1.0), (10000, 20000, 0.1778), (20000, 30000, 0.0316), (30000, 40000, 0.0056), (40000, 50000, 0.001)]
    
    # Process each segment
    for start, end, snr in snr_segments:
        segment = particles[start:end]
        std = np.std(segment)
        sigma = std / np.sqrt(snr)
        print(f'Adding noise with std {sigma} to particles from {start} to {end}')
        particles[start:end] += np.random.normal(0, sigma, segment.shape)
    
    # Save the processed particles
    mrc.write(args.o, particles.astype(np.float32))
    
    if args.out_png:
        plot_projections(args.out_png, particles[:9])
    
    print('All processing completed.')

if __name__ == '__main__':
    main(parse_args().parse_args())
