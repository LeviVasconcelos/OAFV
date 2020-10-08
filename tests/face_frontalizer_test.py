import sys
sys.path.append('.')
from preprocess.face_frontalizer import FaceFrontalizer
import cv2

def main():
    filenames = ['tests/input/trump_hillary.jpg', 
                 'tests/input/emma.jpg', 
                'tests/input/Adisai_Bodharamik_0001.jpg', 
                'tests/input/Aaron_Eckhart_0001.jpg',
                'tests/input/Angela_Merkel_0004.jpg']
    count = 1
    for filename in filenames:
        image = cv2.imread(filename)
        frontalizer = FaceFrontalizer()
        face_images, frontal_array = frontalizer(image)
        for face, isFrontal in zip(face_images, frontal_array):
            frontal = "frontal" if isFrontal else "occluded"
            file_out = ('tests/output/frontalizer_face_%d_%s.jpg' % (count,frontal))
            cv2.imwrite(file_out,face)
            print('image saved: %s' % file_out)
            count += 1

if __name__ == '__main__':
    main()
