'''Skeleton script'''

import argparse
import numpy as np
import sys, os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input')
    parser.add_argument('-D', type=int, help='Input')
    parser.add_argument('-o', help='Output')
    return parser

def main(args):
    ctf_params = pickle.load(open(args.input,'rb'))
    print(ctf_params.shape)
    Nimg = len(ctf_params)
    if ctf_params.shape[1] == 7: # backwards compatibility with no parsing of phase shift
        ctf_params = np.concatenate([ctf_params,np.zeros((Nimg,1),dtype=np.float32)], axis=1)
    if ctf_params.shape[1] == 8:
        ctf_params = np.concatenate([np.ones((Nimg,1),dtype=np.float32)*args.D, ctf_params], axis=1)
    print(ctf_params.shape)
    print(ctf_params[0])
    pickle.dump(ctf_params, open(args.o,'wb'))

if __name__ == '__main__':
    main(parse_args().parse_args())
