import sys
sys.path.append('.')
import os

import numpy as np
import cv2
import argparse

from sklearn.decomposition import PCA
from tqdm import tqdm

def computeEigenFaces(data, k):
    ### data instances should be on dimension 0
    pca = PCA(k, whiten=True)
    pca.fit(data.reshape(data.shape[0], -1))
    e_vector = pca.components_
    e_values = pca.singular_values_
    return e_vector.reshape(k, *data.shape[1:]), e_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default=20, type=int, help='number of eigenfaces to be computed')
    parser.add_argument('-n', default=200, type=int, help='number of images to be used')
    parser.add_argument('-f', dest='frontal_file', required='True', help='frontal faces index file')
    parser.add_argument('-o', dest='out_file', default='eigenfaces.npy', help='file to dump eigenfaces')
    parser.add_argument('--draw_faces', default=None, help='draw eigen faces on output dir')
    parser.add_argument('--shuffle', required=False, action='store_true', help='shuffle dataset before computing PCA if k first faces')
    args = parser.parse_args()
    print('Loading frontal faces')
    with open(args.frontal_file) as f:
        n_frontal = int(f.readline().split(' ')[-1])
        images = None
        for line in tqdm(f.readlines()[:args.n]):
            line = line.replace('\n','')
            img = cv2.imread(line)
            if images is None:
                images = img[np.newaxis]
            else:
                images = np.vstack((images, img[np.newaxis]))
    e_faces, e_values = computeEigenFaces(images, args.k)
    with open(args.out_file, 'wb') as dump_file:
        np.save(dump_file, e_faces)
     
    if args.draw_faces is not None:
        for i, face in enumerate(e_faces):
            fileout = os.path.join(args.draw_faces, 'face_k%d.jpg' % i)
            min_ = np.min(face)
            max_ = np.max(face)
            face = 255. * (face - min_)/(max_ - min_)
            cv2.imwrite(fileout, face)
            print('eigen face drawn at: %s' % fileout)

