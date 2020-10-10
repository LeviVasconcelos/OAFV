import sys
sys.path.append('.')

import argparse
import cv2
from pathlib import Path
from face_frontalizer import FaceFrontalizer
from tqdm import tqdm
from shutil import copyfile

def preprocess(root, frontal_file='frontal_faces.txt'):
    ## Preprocess lfw dataset by applying frontalization
    ## and saving frontal image paths for eigen face computation
    image_files = list(Path(root).rglob("*.[jJ][pP][gG]"))
    out_size = [0, 0, 128, 128]
    frontal_threshold = 15
    frontalizer = FaceFrontalizer(out_size=out_size, 
                                 frontal_threshold=frontal_threshold,
                                 single_face=True)
    frontal_faces = []
    for filename in tqdm(image_files):
        filename = str(filename)
        image = cv2.imread(filename)
        face_array, frontal_array = frontalizer(image)
        if len(face_array) > 1:
            print('More than one face detected: %s' % filename)
            break
        face, isFrontal = [face_array[0], frontal_array[0]]
        out_filename = filename.split('.')[0] + '_frontalized.jpg' 
        cv2.imwrite(out_filename, face)
        if isFrontal:
            frontal_faces.append(out_filename)
    ## Dump frontal faces into a textfile
    if frontal_file is not None:
        with open(frontal_file, 'w') as f:
            f.write('frontal threshold: ' + str(frontal_threshold) + '\n')
            for filename in frontal_faces:
                f.write(filename + '\n')
        print('frontal faces saved at: %s, with %d entries' % (frontal_file, len(frontal_faces)))
    return frontal_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data/lfw', help='root dir of lfw dataset')
    parser.add_argument('--copy_frontal', default=None, type=string, help='dir which frontal faces should be copied to')
    args = parser.parse_args()
    frontal_file = preprocess(root=args.root_dir)
    if parser.copy_frontal_faces is not None:
        copy_frontal_faces(frontal_file, args.copy_frontal)

