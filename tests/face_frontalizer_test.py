import sys
sys.path.append('.')
from preprocess.face_frontalizer import FaceFrontalizer
import cv2

def main():
    filenames = ['tests/input/trump_hillary.jpg', 
                'tests/input/Adisai_Bodharamik_0001.jpg', 
                'tests/input/Angela_Merkel_0004.jpg']
    count = 1
    for filename in filenames:
        image = cv2.imread(filename)
        frontalizer = FaceFrontalizer()
        face_images = frontalizer(image)
        for face in face_images:
            file_out = ('tests/output/frontalizer_face_%d.jpg' % count)
            cv2.imwrite(file_out,face)
            print('image saved: %s' % file_out)
            count += 1

if __name__ == '__main__':
    main()
