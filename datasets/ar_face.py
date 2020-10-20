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
from pathlib import Path
from tqdm import tqdm

class ARFaceDataset(data.Dataset):
    def __init__(self, root, split='cropped'):
        self.root = root
        self.split = split
        self.gender_value = {'M': 0, 'W': 71}
        self.n_categories = 26
        self.n_subjects = 126 
        if 'cropped' in self.split:
             self.gender_value = {'M': 0, 'W': 50}
             self.n_subjects = 100 
        self.n_samples = self.n_categories * self.n_subjects
        self.categories = {
                '01': 'neutral',
                '02': 'expression',
                '03': 'expression',
                '04': 'expression',
                '05': 'light',
                '06': 'light',
                '07': 'light',
                '08': 'sun glasses',
                '09': 'sun glasses',
                '10': 'sun glasses',
                '11': 'scarf',
                '12': 'scarf',
                '13': 'scarf',
               }
        self.ordered_samples = []
        self.ordered_labels = []
        self.shuffled_idxs = None 

        self._parse_dataset()

    def _parse_dataset(self):
        def _parse_filename(filename):
            label = {}
            plabel = filename.replace('.bmp', '').split('/')[-1].split('-')
            label['id'] = int(plabel[1]) + self.gender_value[plabel[0]]
            category = int(plabel[-1]) 
            if category > 13:
                category -= 13
            label['category'] = self.categories['%02.d' % category]
            return label, plabel

        image_files = list(Path(self.root).rglob("*.[bB][mM][pP]"))
        image_files.sort()
        for filename in tqdm(image_files):
            label, plabel = _parse_filename(str(filename))
            self.ordered_labels.append(label)
            self.ordered_samples.append(str(filename))

    def __len__(self):
        return len(self.ordered_samples)

    def load_image(self, image):
        return cv2.imread(image)

    def __getitem__(self, index):
        image_filename = self.ordered_samples[index]
        img = self.load_image(image_filename)
        return [img, self.ordered_labels[index], image_filename]

