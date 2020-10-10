import sys
sys.path.append('.')
import os

import argparse
from tqdm import tqdm
from shutil import copyfile

def copy_frontal_faces(frontal_file, out_dir):
    with open(frontal_file) as f:
        n_faces = int(f.readline().split(' ')[-1])
        for line in tqdm(f.readlines()):
            line = line.replace('\n', '')
            tgt = os.path.join(out_dir, line.split('/')[-1])
            #print('copying from %s to %s' % (line, tgt))
            copyfile(line, tgt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='', help='file indexing frontal faces')
    parser.add_argument('-o', default=None, help='dir which frontal faces should be copied to')
    args = parser.parse_args()
    if args.o is not None:
        copy_frontal_faces(args.f, args.o)


