"""COLMAP model reader supporting both text and binary formats for visual localization."""
import struct
import numpy as np
from pathlib import Path


class COLMAPLocalizationModel:
    """Reads COLMAP reconstruction for 2D-3D correspondence in localization.

    Supports both text format (cameras.txt, images.txt, points3D.txt) and
    binary format (cameras.bin, images.bin, points3D.bin).

    Binary format key: point3D_id is uint64 (8 bytes) in both images.bin
    and points3D.bin. Track entries are (image_id uint32, point2D_idx int32).
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.cameras = {}
        self.images = {}
        self.points3D = {}
        self.name_to_id = {}
        self.image_width = None
        self.image_height = None

        if (self.model_path / "images.bin").exists():
            self._read_binary()
        else:
            self._read_text()

        self._build_index()
        self._compute_image_size()

    def _compute_image_size(self):
        """Infer the reconstruction image size from cameras."""
        if not self.cameras:
            return
        cam = next(iter(self.cameras.values()))
        self.image_width = int(cam["width"])
        self.image_height = int(cam["height"])

    # ---- binary format --------------------------------------------------------

    def _read_binary(self):
        self._read_cameras_bin()
        self._read_images_bin()
        self._read_points3D_bin()

    def _read_cameras_bin(self):
        path = self.model_path / "cameras.bin"
        if not path.exists():
            return
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<I", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                n = _camera_num_params(model_id)
                params = struct.unpack(f"<{n}d", f.read(8 * n))
                model_name = _camera_model_name(model_id)
                self.cameras[camera_id] = {
                    "model": model_name,
                    "width": int(width),
                    "height": int(height),
                    "params": np.array(params, dtype=np.float64),
                }
        print(f"  {len(self.cameras)} COLMAP cameras (binary)")

    def _read_images_bin(self):
        path = self.model_path / "images.bin"
        if not path.exists():
            return
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = np.array(struct.unpack("<dddd", f.read(32)), dtype=np.float64)
                tvec = np.array(struct.unpack("<ddd", f.read(24)), dtype=np.float64)
                camera_id = struct.unpack("<I", f.read(4))[0]
                # null-terminated name
                name_bytes = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name_bytes += c
                name = name_bytes.decode("utf-8")
                num_points2D = struct.unpack("<Q", f.read(8))[0]
                # x(double), y(double), point3D_id(int64) = 8+8+8 = 24 bytes per point
                raw = f.read(num_points2D * 24)
                x_y_id_s = struct.unpack('<' + 'ddq' * num_points2D, raw)
                xy = np.column_stack([
                    np.array(x_y_id_s[0::3], dtype=np.float64),
                    np.array(x_y_id_s[1::3], dtype=np.float64),
                ])
                pids = np.array(x_y_id_s[2::3], dtype=np.int64)
                # filter valid
                valid = pids >= 0
                self.images[image_id] = {
                    "name": name,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "keypoints": xy[valid].astype(np.float32),
                    "point3D_ids": pids[valid],
                }
        print(f"  {len(self.images)} COLMAP images with keypoints (binary)")

    def _read_points3D_bin(self):
        path = self.model_path / "points3D.bin"
        if not path.exists():
            return
        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack("<Q", f.read(8))[0]
                xyz = np.array(struct.unpack("<ddd", f.read(24)), dtype=np.float64)
                f.read(3)  # rgb (unused)
                f.read(8)  # error (unused)
                track_length = struct.unpack("<Q", f.read(8))[0]
                f.seek(track_length * 8, 1)  # skip tracks: uint32+int32 each
                self.points3D[point3D_id] = {"xyz": xyz}
        print(f"  {len(self.points3D)} 3D points (binary)")

    # ---- text format (legacy) ------------------------------------------------

    def _read_text(self):
        self._read_cameras_txt()
        self._read_images_txt()
        self._read_points3D_txt()

    def _read_cameras_txt(self):
        path = self.model_path / "cameras.txt"
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                w, h = int(parts[2]), int(parts[3])
                params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
                self.cameras[cam_id] = {
                    "model": model, "width": w, "height": h, "params": params,
                }
        print(f"  {len(self.cameras)} COLMAP cameras (text)")

    def _read_images_txt(self):
        path = self.model_path / "images.txt"
        if not path.exists():
            return
        with open(path) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 9 and parts[0].isdigit():
                img_id = int(parts[0])
                qvec = np.array([float(x) for x in parts[1:5]])
                tvec = np.array([float(x) for x in parts[5:8]])
                cam_id = int(parts[8])
                name = parts[9]
                kpts = []
                p3d_ids = []
                i += 1
                if i < len(lines):
                    pts_line = lines[i].strip()
                    if pts_line and not pts_line.startswith("#"):
                        pts_parts = pts_line.split()
                        for j in range(0, len(pts_parts) - 2, 3):
                            try:
                                x = float(pts_parts[j])
                                y = float(pts_parts[j + 1])
                                pid = int(pts_parts[j + 2])
                                if pid >= 0:
                                    kpts.append((x, y))
                                    p3d_ids.append(pid)
                            except (ValueError, IndexError):
                                break
                self.images[img_id] = {
                    "name": name,
                    "qvec": qvec, "tvec": tvec, "cam_id": cam_id,
                    "keypoints": np.array(kpts, dtype=np.float32),
                    "point3D_ids": np.array(p3d_ids, dtype=np.int64),
                }
            i += 1
        print(f"  {len(self.images)} COLMAP images with keypoints (text)")

    def _read_points3D_txt(self):
        path = self.model_path / "points3D.txt"
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                pt_id = int(parts[0])
                xyz = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
                self.points3D[pt_id] = {"xyz": xyz}
        print(f"  {len(self.points3D)} 3D points (text)")

    # ---- common ---------------------------------------------------------------

    def _build_index(self):
        for img_id, img in self.images.items():
            self.name_to_id[img["name"]] = img_id

    def get_image(self, image_name: str):
        """Get image data by name, trying .png/.jpg variants."""
        for name in (image_name,
                     image_name.replace(".png", ".jpg"),
                     image_name.replace(".jpg", ".png")):
            if name in self.name_to_id:
                return self.images[self.name_to_id[name]]
        return None

    def get_keypoints_and_3d_ids(self, image_name: str):
        """Get COLMAP keypoints and their 3D point IDs for an image."""
        img = self.get_image(image_name)
        if img is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return img["keypoints"], img["point3D_ids"]

    def get_reconstruction_scale(self) -> float:
        """Return scale factor: dataset_pixels / colmap_pixels."""
        return 1.0  # caller handles scaling via resize


def _camera_model_name(model_id):
    """COLMAP camera model IDs."""
    names = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE",
    }
    return names.get(model_id, f"UNKNOWN_{model_id}")


def _camera_num_params(model_id):
    """Number of parameters for each COLMAP camera model."""
    counts = {
        0: 3,   # SIMPLE_PINHOLE: f, cx, cy
        1: 4,   # PINHOLE: fx, fy, cx, cy
        2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
        3: 5,   # RADIAL: f, cx, cy, k1, k2
        4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        5: 8,   # OPENCV_FISHEYE
        6: 12,  # FULL_OPENCV
        7: 5,   # FOV: fx, fy, cx, cy, omega
        8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
        9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
        10: 12, # THIN_PRISM_FISHEYE
    }
    return counts.get(model_id, 4)
