import sys
sys.path.append('./TDDFA')
import os
import yaml
import numpy as np

from TDDFA.TDDFA import TDDFA
from TDDFA.utils.tddfa_util import (
        u, w_shp, w_exp,
        tri, _parse_param
        )
from TDDFA.utils.pose import P2sRt, matrix2angle
from TDDFA.FaceBoxes import FaceBoxes
from TDDFA.utils.serialization import get_colors 
from TDDFA.utils.tddfa_util import _to_ctype
from TDDFA.Sim3DR import rasterize

class FaceFrontalizer():
    def __init__(self, out_size=[0, 0, 128, 128], frontal_threshold=15, single_face=False):
        self.single_face = single_face
        self.frontal_threshold = frontal_threshold
        self.out_size = np.asarray(out_size)
        self.config_file = './TDDFA/configs/mb1_120x120.yml'
        self.cfg = yaml.load(open(self.config_file), Loader=yaml.SafeLoader)
        self.gpu_mode = 'gpu'
        self.tddfa = TDDFA(gpu_mode=self.gpu_mode, **self.cfg)
        self.face_boxes = FaceBoxes()

    def _draw_face(self, img, ver, ver_o, box):
        overlap = np.zeros((box[2].astype(int), box[3].astype(int), 3), dtype=img.dtype) 
        colors = get_colors(img, ver)
        ver = _to_ctype(ver.T)  # transpose to m x 3
        #colors = bilinear_interpolate(img, ver[:, 0], ver[:, 1]) / 255.
        ver_o_ = _to_ctype(ver_o.T)
        overlap = rasterize(ver_o_, tri, colors, bg=overlap)
        return overlap

    def _similarity_transform(self, points, roi_box, size, center=False):
        # Inspired from tddfa/utils/tddfa_util.similar_transform
        points[0, :] -= 1  # for Python compatibility
        points[2, :] -= 1
        points[1, :] = size - points[1, :]

        sx, sy, ex, ey = roi_box
        scale_x = (ex - sx) / size
        scale_y = (ey - sy) / size
        points[0, :] = points[0, :] * scale_x + sx 
        points[1, :] = points[1, :] * scale_y + sy
        s = (scale_x + scale_y) / 2
        points[2, :] *= s
        points[2, :] -= np.min(points[2, :])
        if center:
            cx, cy = roi_box[2:]/2.
            pixel_center = np.mean(points, axis=-1)
            shift = np.asarray([[cx, cy, 0]], dtype=points.dtype) - pixel_center
            points += shift.T
        return np.array(points, dtype=np.float32)


    def _compute_face_models(self, param, roi_box):
        """Dense points reconstruction: 53215 points"""
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        P = param[:12].reshape(3,-1)
        s, R_, t3d = P2sRt(P)
        yaw, pitch, roll = [x * 180 / np.pi for x in matrix2angle(R_)]
        size = self.tddfa.size
        mask_model = (u + w_shp @ alpha_shp + w_exp @ alpha_exp)
        pts3d = (R_ * s) @ mask_model.reshape(3, -1, order='F') + offset
        pts3d = self._similarity_transform(pts3d, roi_box, size)
        pts3d_origin = (s * mask_model.reshape(3, -1, order='F')) + offset
        pts3d_origin = self._similarity_transform(pts3d_origin, self.out_size, size, center=True) 

        isFrontal = (yaw < self.frontal_threshold) and (pitch < self.frontal_threshold)
        return pts3d, pts3d_origin, isFrontal

    def __call__(self, img):
        boxes = self.face_boxes(img)
        n_faces = len(boxes)
        if n_faces == 0:
            return
        if self.single_face and len(boxes) > 1:
            area = -1
            bbox = None
            for box in boxes:
                left, top, right, bottom = box[:4]
                box_area = (right - left) * (bottom - top)
                if box_area > area:
                    area = box_area
                    bbox = box
            boxes = [bbox] 

        param_lst, roi_box_lst = self.tddfa(img, boxes)
        face_images = []
        frontal_array = []
        for param,roi_box in zip(param_lst, roi_box_lst):
            fitted_model, front_model, isFrontal = self._compute_face_models(param, roi_box)
            face_images.append(self._draw_face(img, fitted_model, front_model, self.out_size))
            frontal_array.append(isFrontal)
        return face_images, frontal_array






