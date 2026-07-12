"""
lib/dataset/facescape_multiview.py

Dataset loader for FaceScape multi_view_data.
Supports 4 training variants:
    Variant 1: RGB only,       raw crop
    Variant 2: RGB only,       RetinaFace crop
    Variant 3: RGB + depth,    raw crop
    Variant 4: RGB + depth,    RetinaFace crop

Coordinate convention (matches virtual_camera_data / facescape.py baseline):
    - GT landmarks: TU-scale mm = lm_world_m * scale (IOD ~96mm, face near origin)
    - Camera t: params.json meters * scale -> TU-scale mm
    - T = -R.T @ t  (same convention as facescape.py)
    - space_size/center: auto-derived from GT bounding box (same as facescape.py)
    - Per-capture centering: subtract face centroid from landmarks, adjust camera t accordingly
"""

import os
import json
import logging
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.transforms import get_affine_transform, get_scale

logger = logging.getLogger(__name__)

NUM_LANDMARKS    = 68
NUM_VIEWS_SAMPLE = 5
SPLIT_AT_SUBJ    = 300
ROOT_LANDMARK    = 30
SPACE_MARGIN_RATIO = 1.0 / 0.60  # matches facescape.py baseline

INVALID_CAM_INDICES = {45, 46, 49, 50, 51, 52, 57}

LM_INDICES = [
    5696, 23350, 5702,  4651,  4650, 20322, 21351,  5013,  1681,  1692,
   11486, 10439, 1338,  1339,  2369, 13524,  2363, 24759,  3549, 24702,
   24687, 24632,14837, 14899, 14914,   237, 14968,  6053,  6041,  1870,
    1855,  4728, 4870,  1807,  1551,  1419,  3434,  3414,  3447,  3457,
    3309,  3373, 3179,   151,   127,   143,  3236,    47, 21018,  4985,
    4898,  6571, 1575,  1663,  1599,  1899, 12138,  5231, 21978,  5101,
   21067, 21239,11378, 11369, 11553, 12048,  5212, 21892,
]


def _load_tu_landmarks_world(obj_path, scale, Rt):
    """Load TU .obj -> world-frame meters via inverse of Rt_scale alignment."""
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    verts = np.array(verts, dtype=np.float32)
    tu_lm = verts[LM_INDICES]
    Rt = np.array(Rt, dtype=np.float64)
    lm_world = (Rt[:3, :3].T @ (tu_lm.astype(np.float64) - Rt[:3, 3]).T).T / scale
    return lm_world.astype(np.float32)  # (68,3) meters


def _tu_landmarks_scaled(obj_path, scale, Rt):
    """Load TU landmarks in TU-scale mm (world_m * scale). IOD ~96mm."""
    lm_m = _load_tu_landmarks_world(obj_path, scale, Rt)
    return (lm_m * scale).astype(np.float32)  # (68,3) TU-scale mm


def _parse_params_json(params_path, scale_factor, global_center=None):
    """Parse params.json -> {view_idx: cam_dict}. t in meters -> TU-scale mm."""
    with open(params_path, 'r') as f:
        raw = json.load(f)

    view_indices = set()
    for key in raw:
        if key.endswith('_K'):
            try:
                view_indices.add(int(key[:-2]))
            except ValueError:
                pass

    cameras = {}
    for i in sorted(view_indices):
        if i in INVALID_CAM_INDICES:
            continue
        if not raw.get(f'{i}_valid', False):
            continue

        K    = np.array(raw[f'{i}_K'],  dtype=np.float64)[:3, :3]
        Rt   = np.array(raw[f'{i}_Rt'], dtype=np.float64)
        R    = Rt[:, :3]
        t    = Rt[:, 3] * scale_factor
        if global_center is not None:
            t = t + R @ global_center.astype(np.float64)  # keep projection consistent after centering
        dist = np.array(raw[f'{i}_distortion'], dtype=np.float64)
        W    = int(raw[f'{i}_width'])
        H    = int(raw[f'{i}_height'])

        T          = (-R.T @ t).reshape(3, 1)
        standard_T = t.reshape(3, 1)

        cameras[i] = {
            'R':          R,
            'T':          T,
            'standard_T': standard_T,
            'K':          K,
            'fx':         K[0, 0],
            'fy':         K[1, 1],
            'cx':         K[0, 2],
            'cy':         K[1, 2],
            'k':          dist[[0, 1, 4]].reshape(3, 1),
            'p':          dist[[2, 3]].reshape(2, 1),
            'distCoef':   dist,
            'width':      W,
            'height':     H,
        }
    return cameras


class FaceScapeMultiView(Dataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__()
        self.cfg            = cfg
        self.is_train       = is_train
        self.transform      = transform
        self.num_joints     = NUM_LANDMARKS
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id        = cfg.DATASET.ROOTIDX
        self.image_size     = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size   = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.num_views      = getattr(cfg.DATASET, 'NUM_VIEWS', NUM_VIEWS_SAMPLE)
        self.use_retinaface = getattr(cfg.DATASET, 'USE_RETINAFACE', False)
        self.use_depth      = getattr(cfg.DATASET, 'USE_DEPTH', False)

        self.data_root  = os.path.join(cfg.DATASET.ROOT, 'multi_view_data')
        self.image_root = os.path.join(self.data_root, 'image')
        self.tu_root    = os.path.join(self.data_root, 'tu')

        rt_scale_path = os.path.join(self.tu_root, 'Rt_scale_dict.json')
        with open(rt_scale_path, 'r') as f:
            self.rt_scale_dict = json.load(f)
        logger.info(f'Loaded Rt_scale_dict from {rt_scale_path}')

        self.detector = None
        if self.use_retinaface:
            self.detector = self._init_retinaface(cfg)

        self.db = self._build_db(is_train)
        self.space_size, self.space_center = self._compute_space_bounds()
        split = 'train' if is_train else 'val'
        logger.info(
            f'FaceScapeMultiView {split}: {len(self.db)} captures, '
            f'retinaface={self.use_retinaface}, depth={self.use_depth}, '
            f'views_per_sample={self.num_views}'
        )
        logger.info(
            f'FaceScapeMultiView auto space_size={self.space_size.tolist()} '
            f'space_center={self.space_center.tolist()}'
        )


    def _init_retinaface(self, cfg):
        try:
            import sys
            retinaface_root = getattr(cfg.DATASET, 'RETINAFACE_ROOT',
                                      '/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface')
            checkpoint = getattr(cfg.DATASET, 'RETINAFACE_CHECKPOINT',
                                 '/nfs/turbo/coe-igmr-pub/seoin/trained_weights/Resnet50_Final.pth')
            sys.path.insert(0, retinaface_root)
            from models.retinaface import RetinaFace
            from data import cfg_re50
            net = RetinaFace(cfg=cfg_re50, phase='test')
            state = torch.load(checkpoint, map_location='cpu')
            net.load_state_dict(state)
            net.eval()
            logger.info('RetinaFace loaded successfully')
            return net
        except Exception as e:
            logger.warning(f'RetinaFace init failed: {e}. Falling back to raw crop.')
            return None

    def _build_db(self, is_train):
        db = []
        subject_dirs = sorted(
            int(d) for d in os.listdir(self.image_root)
            if os.path.isdir(os.path.join(self.image_root, d)) and d.isdigit()
        )

        for subj in subject_dirs:
            if is_train     and subj >= SPLIT_AT_SUBJ: continue
            if not is_train and subj <  SPLIT_AT_SUBJ: continue

            subj_str = str(subj)
            if subj_str not in self.rt_scale_dict:
                logger.warning(f'Subject {subj} not in Rt_scale_dict, skipping')
                continue

            subj_image_dir = os.path.join(self.image_root, subj_str)
            subj_tu_dir    = os.path.join(self.tu_root, subj_str, 'models_reg')

            if not os.path.isdir(subj_tu_dir):
                logger.warning(f'No TU models for subject {subj}, skipping')
                continue

            for expr in sorted(os.listdir(subj_image_dir)):
                expr_image_dir = os.path.join(subj_image_dir, expr)
                if not os.path.isdir(expr_image_dir):
                    continue

                expr_idx = expr.split('_')[0]
                if expr_idx not in self.rt_scale_dict[subj_str]:
                    continue

                params_path = os.path.join(expr_image_dir, 'params.json')
                obj_path    = os.path.join(subj_tu_dir, f'{expr}.obj')

                if not os.path.isfile(params_path) or not os.path.isfile(obj_path):
                    continue

                entry = self.rt_scale_dict[subj_str][expr_idx]

                try:
                    lm_check  = _load_tu_landmarks_world(obj_path, entry[0], entry[1])
                    iod_check = np.linalg.norm(lm_check[45] - lm_check[36])
                    if iod_check > 1.0:
                        logger.warning(f'Skipping subj {subj} {expr}: IOD={iod_check:.3f}m (bad registration)')
                        continue
                except Exception as e:
                    logger.warning(f'IOD check failed for subj {subj} {expr}: {e}')
                    continue

                view_indices = sorted(
                    int(os.path.splitext(f)[0])
                    for f in os.listdir(expr_image_dir)
                    if f.endswith('.jpg')
                )

                if len(view_indices) < self.num_views:
                    continue

                db.append({
                    'subject':     subj,
                    'expression':  expr,
                    'image_dir':   expr_image_dir,
                    'params_path': params_path,
                    'obj_path':    obj_path,
                    'all_views':   view_indices,
                    'scale':       entry[0],
                    'Rt':          entry[1],
                })

        return db

    def _compute_space_bounds(self):
        """Auto-derive space_size and space_center. Cached to disk."""
        cache_path = os.path.join(self.tu_root, 'space_bounds_cache.json')
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            logger.info(f'Loaded space bounds from cache: {cache}')
            return (np.array(cache['size'], dtype=np.float32),
                    np.array(cache['center'], dtype=np.float32))

        logger.info('Computing space bounds (first time, will be cached)...')
        all_lm = []
        for rec in self.db:
            lm = _tu_landmarks_scaled(rec['obj_path'], rec['scale'], rec['Rt'])
            all_lm.append(lm - lm.mean(axis=0))  # per-capture centering
        all_lm = np.concatenate(all_lm, axis=0)

        mins   = all_lm.min(axis=0)
        maxs   = all_lm.max(axis=0)
        extent = (maxs - mins).max()
        size   = np.array([extent, extent, extent]) * SPACE_MARGIN_RATIO
        center = np.array([0.0, 0.0, 0.0])  # per-capture centering -> origin

        with open(cache_path, 'w') as f:
            json.dump({'size': size.tolist(), 'center': center.tolist()}, f)
        logger.info(f'Space bounds cached to {cache_path}')
        return size.astype(np.float32), center.astype(np.float32)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        rec = self.db[idx]

        joints_3d   = _tu_landmarks_scaled(rec['obj_path'], rec['scale'], rec['Rt'])
        face_center = joints_3d.mean(axis=0)  # per-capture centering
        joints_3d   = joints_3d - face_center
        all_cameras = _parse_params_json(rec['params_path'], scale_factor=rec['scale'],
                                         global_center=face_center)
        valid_views = [v for v in rec['all_views'] if v in all_cameras]

        if len(valid_views) < self.num_views:
            valid_views = (valid_views * ((self.num_views // len(valid_views)) + 1))[:self.num_views]

        if self.is_train:
            chosen_views = random.sample(valid_views, self.num_views)
        else:
            chosen_views = valid_views[:self.num_views]

        MAX_P = self.maximum_person
        joints_3d_u     = np.zeros((MAX_P, NUM_LANDMARKS, 3), dtype=np.float32)
        joints_3d_vis_u = np.zeros((MAX_P, NUM_LANDMARKS, 3), dtype=np.float32)
        joints_3d_u[0]     = joints_3d
        joints_3d_vis_u[0] = 1.0
        voxelpose_pred_u = np.zeros((MAX_P, NUM_LANDMARKS, 5), dtype=np.float32)
        roots_3d = joints_3d_u[:, ROOT_LANDMARK, :]

        all_inputs = []
        all_meta   = []

        for view_idx in chosen_views:
            cam      = all_cameras[view_idx]
            img_path = os.path.join(rec['image_dir'], f'{view_idx}.jpg')

            img = cv2.imread(img_path)
            if img is None:
                raise IOError(f'Cannot read: {img_path}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.undistort(img,
                                cam['K'].astype(np.float32),
                                cam['distCoef'].astype(np.float32))

            orig_h, orig_w = img.shape[:2]

            if self.use_retinaface and self.detector is not None:
                img, crop_offset = self._retinaface_crop(img, cam)
            else:
                crop_offset = np.array([0.0, 0.0], dtype=np.float32)

            crop_h, crop_w = img.shape[:2]
            c = np.array([crop_w / 2.0, crop_h / 2.0])
            s = get_scale((crop_w, crop_h), self.image_size)
            r = 0
            trans = get_affine_transform(c, s, r, self.image_size, inv=0)
            img_warped = cv2.warpAffine(
                img, trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR
            )

            if self.transform:
                img_tensor = self.transform(img_warped)
            else:
                img_norm = img_warped.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_norm = (img_norm - mean) / std
                img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))

            if self.use_depth:
                depth = self._render_depth(rec['obj_path'], rec['scale'],
                                           rec['Rt'], cam, orig_w, orig_h)
                depth_crop = depth[
                    int(crop_offset[1]):int(crop_offset[1]) + crop_h,
                    int(crop_offset[0]):int(crop_offset[0]) + crop_w
                ]
                depth_warped = cv2.warpAffine(
                    depth_crop, trans,
                    (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_NEAREST
                )
                depth_tensor = torch.from_numpy(depth_warped[None].astype(np.float32))
                img_tensor = torch.cat([img_tensor, depth_tensor], dim=0)

            aff_trans     = np.eye(3, dtype=np.float64)
            aff_trans[:2] = trans
            inv_trans     = get_affine_transform(c, s, r, self.image_size, inv=1)
            inv_aff_trans = np.eye(3, dtype=np.float64)
            inv_aff_trans[:2] = inv_trans
            hm_scale    = self.heatmap_size / self.image_size
            scale_trans = np.eye(3, dtype=np.float64)
            scale_trans[0, 0] = hm_scale[1]
            scale_trans[1, 1] = hm_scale[0]
            aug_trans = scale_trans @ aff_trans

            joints_2d_u  = np.zeros((MAX_P, NUM_LANDMARKS, 2), dtype=np.float32)
            joints_vis_u = np.zeros((MAX_P, NUM_LANDMARKS, 2), dtype=np.float32)
            j2d, j2d_vis = self._project_to_image(joints_3d, cam)
            joints_2d_u[0]  = j2d
            joints_vis_u[0] = np.concatenate([j2d_vis, j2d_vis], axis=1)

            cam_intri = np.eye(3, dtype=np.float64)
            cam_intri[0, 0] = float(cam['fx'])
            cam_intri[1, 1] = float(cam['fy'])
            cam_intri[0, 2] = float(cam['cx'])
            cam_intri[1, 2] = float(cam['cy'])

            camera_focal = np.array(
                [float(cam['fx']), float(cam['fy']), 1.0], dtype=np.float32
            )

            meta = {
                'image':                    img_path,
                'num_person':               1,
                'joints_3d':                joints_3d_u,
                'joints_3d_vis':            joints_3d_vis_u,
                'joints_3d_voxelpose_pred': voxelpose_pred_u,
                'roots_3d':                 roots_3d,
                'joints':                   joints_2d_u,
                'joints_vis':               joints_vis_u,
                'center':                   c.astype(np.float32),
                'scale':                    s.astype(np.float32),
                'rotation':                 np.float32(r),
                'camera':                   cam,
                'camera_Intri':             cam_intri,
                'camera_R':                 cam['R'],
                'camera_focal':             camera_focal,
                'camera_T':                 cam['T'],
                'camera_standard_T':        cam['standard_T'],
                'affine_trans':             aff_trans,
                'inv_affine_trans':         inv_aff_trans,
                'aug_trans':                aug_trans,
            }

            all_inputs.append(img_tensor)
            all_meta.append(meta)

        return all_inputs, all_meta

    def _project_to_image(self, joints_3d_scaled_mm, cam):
        """Project TU-scale mm landmarks. T = -R.T @ t convention (matches facescape.py)."""
        R  = cam['R']
        t  = cam['standard_T'].reshape(3)
        fx, fy = float(cam['fx']), float(cam['fy'])
        cx, cy = float(cam['cx']), float(cam['cy'])

        pts_cam = (R @ joints_3d_scaled_mm.T).T + t
        z = pts_cam[:, 2].clip(min=1e-8)
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy

        joints_2d = np.stack([u, v], axis=1).astype(np.float32)
        in_front  = pts_cam[:, 2] > 0
        in_bounds = ((u >= 0) & (u < cam['width']) &
                     (v >= 0) & (v < cam['height']))
        vis = (in_front & in_bounds).astype(np.float32).reshape(-1, 1)
        return joints_2d, vis

    def _retinaface_crop(self, img, cam):
        try:
            import torch
            from layers.functions.prior_box import PriorBox
            from utils.box_utils import decode
            from utils.nms.py_cpu_nms import py_cpu_nms
            from data import cfg_re50

            device = next(self.detector.parameters()).device
            h, w = img.shape[:2]
            scale_img = torch.Tensor([w, h, w, h]).to(device)
            img_t = np.float32(img) - (104, 117, 123)
            img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                loc, conf, _ = self.detector(img_t)

            priorbox = PriorBox(cfg_re50, image_size=(h, w))
            priors   = priorbox.forward().to(device)
            boxes    = decode(loc.squeeze(0), priors, cfg_re50['variance'])
            boxes    = (boxes * scale_img).cpu().numpy()
            scores   = conf.squeeze(0).cpu().numpy()[:, 1]
            keep     = py_cpu_nms(
                np.hstack([boxes, scores[:, None]]).astype(np.float32), 0.4)
            boxes, scores = boxes[keep], scores[keep]

            if len(boxes) == 0:
                return img, np.array([0.0, 0.0], dtype=np.float32)

            x1, y1, x2, y2 = boxes[np.argmax(scores)].astype(int)
            pad_x = int((x2 - x1) * 0.2);  pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x);       y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x);       y2 = min(h, y2 + pad_y)
            return img[y1:y2, x1:x2], np.array([x1, y1], dtype=np.float32)

        except Exception as e:
            logger.warning(f'RetinaFace detection failed: {e}')
            return img, np.array([0.0, 0.0], dtype=np.float32)

    def _render_depth(self, obj_path, scale, Rt, cam, width, height):
        try:
            import os
            os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
            import pyrender, trimesh
            mesh  = trimesh.load(obj_path, force='mesh')
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(mesh))

            R = cam['R']
            t = cam['standard_T'].reshape(3)
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = R.T
            cam_pose[:3,  3] = -R.T @ t
            cam_pose[:, 1] *= -1
            cam_pose[:, 2] *= -1

            camera = pyrender.IntrinsicsCamera(
                fx=float(cam['fx']), fy=float(cam['fy']),
                cx=float(cam['cx']), cy=float(cam['cy']),
                znear=0.01, zfar=20.0)
            scene.add(camera, pose=cam_pose)

            renderer = pyrender.OffscreenRenderer(width, height)
            _, depth  = renderer.render(scene)
            renderer.delete()
            return (depth * 1000.0).astype(np.float32)

        except Exception as e:
            logger.warning(f'Depth render failed: {e}')
            return np.zeros((height, width), dtype=np.float32)

    def evaluate(self, preds, output_dir=None, *args, **kwargs):
        gts = []
        for rec in self.db:
            lm = _tu_landmarks_scaled(rec['obj_path'], rec['scale'], rec['Rt'])
            gts.append(lm - lm.mean(axis=0))  # per-capture centering
        gts = np.array(gts, dtype=np.float32)

        pred_joints = []
        for pred in preds:
            pred = np.array(pred)
            if pred.ndim == 3:
                valid = pred[pred[:, 0, 3] >= 0]
                if len(valid) == 0:
                    valid = pred
                pred_joints.append(valid[valid[:, :, 3].mean(axis=1).argmax(), :, :3])
            else:
                pred_joints.append(pred[:, :3])
        pred_joints = np.array(pred_joints, dtype=np.float32)

        iod  = np.linalg.norm(gts[:, 45] - gts[:, 36], axis=1, keepdims=True).clip(min=1e-6)
        dist = np.linalg.norm(pred_joints - gts, axis=2)
        nme  = (dist / iod).mean()

        logger.info(f'[FaceScapeMultiView] NME (3D inter-ocular): {nme * 100:.3f} %')
        return [('NME_3D', nme)], nme
