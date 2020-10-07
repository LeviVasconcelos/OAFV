#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 5 20:47:29 2020
@author: levi
"""

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import h5py

from tqdm import tqdm
import pickle

class LFWDataset(data.Dataset):
    def __init__(self, root, split, transforms):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.tuples = []
        self.index_files = os.path.join(self.root, self.split)
        self._parse_pairs(self.index_files)

    def _parse_pairs(self, pairs_file):
        print('Loading dataset split: %s' % pairs_file)
        def get_image_name(name, i):
            return os.path.join(self.root, name, name + ('_%.4d.jpg' % i))

        def __parse_line(line, isPair):
            out = {}
            out['person1'] = get_image_name(line[0], int(line[1]))
            out['person2'] = get_image_name(line[0], int(line[2])) if isPair else  get_image_name(line[2], int(line[3]))
            out['isPair'] = isPair
            return out

        def __parse_batch(kLength, f, isPair):
            for x in range(kLength):
                l = f.readline().split('\t')
                if len(l) < 3:
                    print('Error: line inconsistency', l)
                self.tuples.append(__parse_line(l, isPair))
        try:
            with open(pairs_file, 'r') as f:
                header_line = f.readline()
                kLength = int(header_line)
                __parse_batch(kLength, f, True)
                __parse_batch(kLength, f, False)

        except EOFError as e:
            raise EOFError

    def __len__(self):
        return len(self.tuples)

    def load_image(self, image):
        return cv2.imread(image)

    def __getitem__(self, index):
        t = self.tuples[index]
        img1 = self.load_image(t['person1'])
        img2 = self.load_image(t['person2'])
        isPair = t['isPair']

        return [img1, img2, isPair]
