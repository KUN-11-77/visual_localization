import numpy as np


def qvec2rotmat(qvec):
    """Convert quaternion (qw,qx,qy,qz) to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ], dtype=np.float64)


class COLMAPModel:
    """Read-only COLMAP model for visual localization."""

    def __init__(self, model_path):
        from pathlib import Path
        self.model_path = Path(model_path)
        self.cameras = self._read_cameras()
        self.images = self._read_images()
        self.points3D = self._read_points3D()
        self._build_name_map()

    def _read_cameras(self):
        cameras = {}
        cameras_path = self.model_path / 'cameras.txt'
        if not cameras_path.exists():
            return cameras
        with open(cameras_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                w, h = int(parts[2]), int(parts[3])
                params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
                cameras[cam_id] = {
                    'model': model, 'width': w, 'height': h, 'params': params
                }
        return cameras

    def _read_images(self):
        images = {}
        images_path = self.model_path / 'images.txt'
        if not images_path.exists():
            return images
        with open(images_path, 'r') as f:
            lines = f.readlines()

        current_img_id = None
        current_name = None
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 10 and parts[0].isdigit():
                img_id = int(parts[0])
                qvec = np.array([float(x) for x in parts[1:5]])
                tvec = np.array([float(x) for x in parts[5:8]])
                cam_id = int(parts[8])
                name = parts[9]
                current_name = name
                current_img_id = img_id
                images[img_id] = {
                    'name': name, 'qvec': qvec, 'tvec': tvec,
                    'camera_id': cam_id, 'point3D_ids': []
                }
            elif len(parts) >= 3 and current_img_id is not None:
                try:
                    x, y, pt3d_id = float(parts[0]), float(parts[1]), int(parts[2])
                    images[current_img_id]['point3D_ids'].append(pt3d_id if pt3d_id != -1 else -1)
                except (ValueError, IndexError):
                    continue
        return images

    def _read_points3D(self):
        points3D = {}
        pts_path = self.model_path / 'points3D.txt'
        if not pts_path.exists():
            return points3D
        with open(pts_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                pt_id = int(parts[0])
                xyz = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
                points3D[pt_id] = {'xyz': xyz}
        return points3D

    def _build_name_map(self):
        self.name_to_image_id = {}
        for img_id, img in self.images.items():
            self.name_to_image_id[img['name']] = img_id

    def get_keypoints_with_3D(self, image_name):
        """Get keypoints and their associated 3D point IDs."""
        if image_name not in self.name_to_image_id:
            return np.array([]), np.array([])

        img_id = self.name_to_image_id[image_name]
        img = self.images[img_id]

        keypoints = []
        point3D_ids = []
        for pt_id in img['point3D_ids']:
            point3D_ids.append(pt_id)

        return np.array(keypoints, dtype=np.float64), np.array(point3D_ids)

    def get_3D_points(self, point3D_ids):
        """Get 3D world coordinates for point IDs."""
        points = []
        valid_mask = []
        for pid in point3D_ids:
            if pid >= 0 and pid in self.points3D:
                points.append(self.points3D[pid]['xyz'])
                valid_mask.append(True)
            else:
                points.append(np.zeros(3, dtype=np.float64))
                valid_mask.append(False)
        return np.array(points, dtype=np.float64), np.array(valid_mask)

    def get_camera_pose(self, image_name):
        """Get camera rotation and translation in world frame."""
        if image_name not in self.name_to_image_id:
            return None, None
        img_id = self.name_to_image_id[image_name]
        img = self.images[img_id]
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']
        return R, t

    def get_camera_intrinsics(self, image_name):
        """Get camera intrinsic matrix K."""
        if image_name not in self.name_to_image_id:
            return None
        img_id = self.name_to_image_id[image_name]
        img = self.images[img_id]
        cam = self.cameras[img['camera_id']]
        params = cam['params']
        w, h = cam['width'], cam['height']
        if cam['model'] == 'SIMPLE_RADIAL':
            fx, fy, cx, r = params[0], params[0], params[1], params[3]
            return np.array([[fx, 0, cx], [0, fy, h/2], [0, 0, 1]], dtype=np.float64)
        elif cam['model'] == 'PINHOLE':
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return np.array([[params[0], [0, params[0], [w/2], [h/2], [1]]], dtype=np.float64)


def read_model(model_path):
    """Convenience function to read entire COLMAP model."""
    return COLMAPModel(model_path)