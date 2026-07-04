"""
lib/dataset/facescape.py  — version using nose tip (landmark 30) as root.
This is the version that achieved NME 4.615% at epoch 99.

Confirmed structure:
    FaceScape/
        <capture_id>/          e.g. 301/, 302/, ...
            <cam_id>/          0/, 1/, 2/, 3/, 4/
                meta.json      K (3x3), R (3x3), t (3,)  camera-local extrinsics
                landmarks_3d.npy  (68, 3) float64  camera-local coords, mm
                landmarks_2d.npy  (68, 2) float64  pixel coords
                depth.npy         (512, 512) float32  mm
                rgb.png           512x512

World-frame GT: R.T @ (lm_cam - t), confirmed std=0.000 across all cameras.
Camera dict T convention matches panoptic.py: T = -R.T @ t.
k/p shapes match panoptic.py: (3,1) and (2,1).
roots_3d uses nose tip (landmark 30).
mean_face_shape used for query initialisation in dq_transformer.py.
"""

import os
import json
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.transforms import get_affine_transform, get_scale

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 68
NUM_CAMERAS   = 5
SPLIT_AT      = 818   # captures 301-817 -> train (117), 818-847 -> val (29)
ROOT_LANDMARK = 30    # nose tip as face root
# Target fraction of the query volume that face landmarks should occupy.
# 0.6 means the face spans 60% of each axis, leaving 20% padding on each side.
# Larger values = tighter volume (more accurate init but less tolerance for
# outlier captures); smaller values = looser volume (more robust init).
TARGET_FILL = 0.60
SPACE_MARGIN_RATIO = 1.0 / TARGET_FILL  # ≈ 1.67


def _cam_local_to_world(lm_cam, R, t):
    """camera-local mm -> world mm.  R.T @ (lm_cam - t)"""
    return (R.T @ (lm_cam - t).T).T


def _load_meta(meta_path):
    """Parse meta.json -> camera dict matching panoptic.py / cameras.py format."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    K  = np.array(meta['K'], dtype=np.float64)
    R  = np.array(meta['R'], dtype=np.float64)
    t  = np.array(meta['t'], dtype=np.float64)
    W  = int(meta.get('W', 512))
    H  = int(meta.get('H', 512))

    T          = (-R.T @ t).reshape(3, 1)
    standard_T = t.reshape(3, 1)

    return {
        'R':          R,
        'T':          T,
        'standard_T': standard_T,
        'K':          K,
        'fx':         np.array(K[0, 0]),
        'fy':         np.array(K[1, 1]),
        'cx':         np.array(K[0, 2]),
        'cy':         np.array(K[1, 2]),
        'k':          np.zeros((3, 1), dtype=np.float64),
        'p':          np.zeros((2, 1), dtype=np.float64),
        'distCoef':   np.zeros(5, dtype=np.float64),
        'width':      W,
        'height':     H,
    }


class FaceScape(Dataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__()
        self.cfg            = cfg
        self.is_train       = is_train
        self.transform      = transform
        self.num_views      = NUM_CAMERAS
        self.num_joints     = NUM_LANDMARKS
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id        = cfg.DATASET.ROOTIDX
        self.image_size     = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size   = np.array(cfg.NETWORK.HEATMAP_SIZE)

        this_dir  = os.path.dirname(__file__)
        self.root = os.path.abspath(
            os.path.join(cfg.DATASET.ROOT, 'virtual_camera_data')
        )

        self.db = self._build_db(is_train)
        split = 'train' if is_train else 'val'
        logger.info(f'FaceScape {split}: {len(self.db)} captures loaded')

        # Query volume (MULTI_PERSON.SPACE_SIZE/CENTER) is derived from the GT
        # landmark bbox instead of a value hand-tuned per dataset scale. Computed
        # over ALL captures (train+val) so train/val/standalone-eval scripts agree
        # regardless of which split instantiates this dataset first.
        self.space_size, self.space_center = self._compute_space_bounds()
        logger.info(
            f'FaceScape auto space_size={self.space_size.tolist()} '
            f'space_center={self.space_center.tolist()}'
        )

    def _build_db(self, is_train):
        db = []
        if not os.path.isdir(self.root):
            raise RuntimeError(f'FaceScape root not found: {self.root}')

        capture_ids = sorted(
            int(d) for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d)) and d.isdigit()
        )

        for cid in capture_ids:
            if is_train     and cid >= SPLIT_AT: continue
            if not is_train and cid <  SPLIT_AT: continue

            cap_dir  = os.path.join(self.root, str(cid))
            cam_data = []
            all_ok   = True

            for cam_id in range(NUM_CAMERAS):
                cam_dir   = os.path.join(cap_dir, str(cam_id))
                meta_path = os.path.join(cam_dir, 'meta.json')
                rgb_path  = os.path.join(cam_dir, 'rgb.png')
                dep_path  = os.path.join(cam_dir, 'depth.npy')
                lm3_path  = os.path.join(cam_dir, 'landmarks_3d.npy')

                if not all(os.path.isfile(p)
                           for p in [meta_path, rgb_path, dep_path, lm3_path]):
                    all_ok = False
                    break

                cam_data.append({
                    'meta':  meta_path,
                    'rgb':   rgb_path,
                    'depth': dep_path,
                    'lm3':   lm3_path,
                })

            if not all_ok:
                logger.warning(f'Skipping incomplete capture: {cid}')
                continue

            db.append({'capture_id': cid, 'cams': cam_data})

        return db

    def _compute_space_bounds(self):
        """Bounding cube (size, center) around all GT landmarks across every
        capture on disk (train+val), independent of any dataset-scale constant."""
        capture_ids = sorted(
            int(d) for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d)) and d.isdigit()
        )

        all_joints = []
        for cid in capture_ids:
            cam0_dir  = os.path.join(self.root, str(cid), '0')
            meta_path = os.path.join(cam0_dir, 'meta.json')
            lm3_path  = os.path.join(cam0_dir, 'landmarks_3d.npy')
            if not (os.path.isfile(meta_path) and os.path.isfile(lm3_path)):
                continue

            cam0    = _load_meta(meta_path)
            lm_cam0 = np.load(lm3_path).astype(np.float64)
            t0      = (-cam0['R'] @ cam0['T'].reshape(3))
            all_joints.append(_cam_local_to_world(lm_cam0, cam0['R'], t0))

        all_joints = np.concatenate(all_joints, axis=0)
        mins = all_joints.min(axis=0)
        maxs = all_joints.max(axis=0)

        # Uniform cube from the largest extent, centered at origin so the
        # sample_space initialization covers the face consistently.
        extent = (maxs - mins).max()
        space_size   = np.array([extent, extent, extent]) * SPACE_MARGIN_RATIO
        space_center = np.array([0.0, 0.0, 0.0])

        return space_size.astype(np.float32), space_center.astype(np.float32)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        rec = self.db[idx]

        cameras = [_load_meta(c['meta']) for c in rec['cams']]

        lm_cam0 = np.load(rec['cams'][0]['lm3']).astype(np.float64)
        R0      = cameras[0]['R']
        t0      = (-R0 @ cameras[0]['T'].reshape(3))
        joints_3d = _cam_local_to_world(lm_cam0, R0, t0).astype(np.float32)

        MAX_P = self.maximum_person
        joints_3d_u     = np.zeros((MAX_P, NUM_LANDMARKS, 3), dtype=np.float32)
        joints_3d_vis_u = np.zeros((MAX_P, NUM_LANDMARKS, 3), dtype=np.float32)
        joints_3d_u[0]     = joints_3d
        joints_3d_vis_u[0] = 1.0
        voxelpose_pred_u = np.zeros((MAX_P, NUM_LANDMARKS, 5), dtype=np.float32)

        # roots_3d: nose tip (landmark 30), shape (MAX_P, 3)
        roots_3d = joints_3d_u[:, ROOT_LANDMARK, :]   # (MAX_P, 3)

        all_inputs = []
        all_meta   = []

        for cam_idx, (cam, paths) in enumerate(zip(cameras, rec['cams'])):
            img = cv2.imread(paths['rgb'])
            if img is None:
                raise IOError(f'Cannot read: {paths["rgb"]}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            c = np.array([orig_w / 2.0, orig_h / 2.0])
            s = get_scale((orig_w, orig_h), self.image_size)
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
                'image':                    paths['rgb'],
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

    def _project_to_image(self, joints_3d_world, cam):
        R  = cam['R']
        T  = cam['T'].reshape(3)
        fx, fy = float(cam['fx']), float(cam['fy'])
        cx, cy = float(cam['cx']), float(cam['cy'])

        pts_cam = (R @ joints_3d_world.T).T - T
        z = pts_cam[:, 2].clip(min=1e-8)
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy

        joints_2d = np.stack([u, v], axis=1).astype(np.float32)
        in_front  = pts_cam[:, 2] > 0
        in_bounds = ((u >= 0) & (u < cam['width']) &
                     (v >= 0) & (v < cam['height']))
        vis = (in_front & in_bounds).astype(np.float32).reshape(-1, 1)
        return joints_2d, vis

    def evaluate(self, preds, output_dir=None, *args, **kwargs):
        gts = []
        for rec in self.db:
            lm_cam = np.load(rec['cams'][0]['lm3']).astype(np.float64)
            cam    = _load_meta(rec['cams'][0]['meta'])
            R0     = cam['R']
            t0     = (-R0 @ cam['T'].reshape(3))
            gts.append(_cam_local_to_world(lm_cam, R0, t0))
        gts = np.array(gts, dtype=np.float32)

        pred_joints = []
        for pred in preds:
            pred = np.array(pred)
            if pred.ndim == 3:
                valid = pred[pred[:, 0, 3] >= 0]
                if len(valid) == 0:
                    valid = pred
                best_idx = valid[:, :, 3].mean(axis=1).argmax()
                pred_joints.append(valid[best_idx, :, :3])
            else:
                pred_joints.append(pred[:, :3])
        pred_joints = np.array(pred_joints, dtype=np.float32)

        iod  = np.linalg.norm(gts[:, 45] - gts[:, 36], axis=1, keepdims=True).clip(min=1e-6)
        dist = np.linalg.norm(pred_joints - gts, axis=2)
        nme  = (dist / iod).mean()

        logger.info(f'[FaceScape] NME (3D inter-ocular): {nme * 100:.3f} %')
        return [('NME_3D', nme)], nme