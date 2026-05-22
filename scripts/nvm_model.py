"""NVM (VisualSFM) reconstruction parser for 2D-3D correspondence lookup."""
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


class NVMModel:
    """Parse VisualSFM .nvm file and provide 2D→3D lookup for localization."""

    def __init__(self, nvm_path: str):
        self.nvm_path = Path(nvm_path)
        # image_name -> (N, 3) array of (x, y, point3D_id)
        self.image_keypoints: dict = {}
        # point3D_id -> (3,) xyz
        self.point3D_xyz: dict = {}
        self._parse()

    def _parse(self):
        print(f"Parsing NVM model: {self.nvm_path} ({self.nvm_path.stat().st_size / 1e6:.0f} MB)...")
        with open(self.nvm_path, "r") as f:
            # Skip header
            line = f.readline()
            while line == "\n" or line.startswith("NVM_V3"):
                line = f.readline()

            num_images = int(line.strip())
            print(f"  {num_images} cameras")

            # Parse camera entries
            image_names = []
            for i in range(num_images):
                line = f.readline()
                while line == "\n":
                    line = f.readline()
                # NVM format: <filename>\t<focal_length> <qw> <qx> <qy> <qz> <tx> <ty> <tz> <radial> 0
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    name = parts[0]
                else:
                    data = line.strip().split(" ")
                    name = data[0]
                image_names.append(name)
            print(f"  Parsed {len(image_names)} camera entries")

            # Parse 3D points
            line = f.readline()
            while line == "\n":
                line = f.readline()
            num_points = int(line.strip())
            print(f"  {num_points} 3D points")

            # Build image->keypoints accumulator
            img_kp_accum = defaultdict(list)

            for pt_id in tqdm(range(num_points), desc="  Loading 3D points"):
                line = f.readline()
                while line == "\n":
                    line = f.readline()
                data = line.strip().split(" ")
                x, y, z = map(float, data[:3])
                num_obs = int(data[6])
                self.point3D_xyz[pt_id] = np.array([x, y, z], dtype=np.float64)

                # Observations are on the same line: [img_idx feat_idx feat_x feat_y] * num_obs
                for j in range(num_obs):
                    s = 7 + 4 * j
                    img_idx = int(data[s])
                    feat_x = float(data[s + 2])
                    feat_y = float(data[s + 3])
                    if img_idx < len(image_names):
                        img_kp_accum[image_names[img_idx]].append(
                            (feat_x, feat_y, pt_id)
                        )

        # Convert lists to arrays
        for name, kps in img_kp_accum.items():
            kps_array = np.array(kps, dtype=np.float64)
            self.image_keypoints[name] = kps_array

        n_with_3d = len(self.image_keypoints)
        n_total_kps = sum(v.shape[0] for v in self.image_keypoints.values())
        print(f"  Done: {n_with_3d} images with 3D, {n_total_kps} total keypoints")

    def has_image(self, image_name: str) -> bool:
        """Check if image is in the NVM model."""
        return image_name in self.image_keypoints

    def lookup_3d(
        self, image_name: str, kpts: np.ndarray, radius: float = 4.0
    ) -> tuple:
        """Find 3D points for detected keypoints by spatial proximity.

        Args:
            image_name: NVM image name (e.g. 'seq8/frame00064.jpg' or '.png')
            kpts: (N, 2) array of detected keypoint (x, y) positions
            radius: max pixel distance for a valid match

        Returns:
            points3D: (N, 3) array of 3D coordinates (zeros for unmatched)
            valid_mask: (N,) bool array
        """
        # Normalize extension: NVM uses .jpg, dataset may use .png
        name_jpg = image_name
        name_png = image_name
        if image_name.endswith(".png"):
            name_jpg = image_name[:-4] + ".jpg"
        elif image_name.endswith(".jpg"):
            name_png = image_name[:-4] + ".png"

        nvm_kps = None
        for name in (image_name, name_jpg, name_png):
            if name in self.image_keypoints:
                nvm_kps = self.image_keypoints[name]
                break

        if nvm_kps is None or len(kpts) == 0:
            return (
                np.zeros((len(kpts), 3), dtype=np.float64),
                np.zeros(len(kpts), dtype=bool),
            )

        nvm_xy = nvm_kps[:, :2]  # (M, 2)
        nvm_pids = nvm_kps[:, 2].astype(np.int64)  # (M,)

        points3D = np.zeros((len(kpts), 3), dtype=np.float64)
        valid = np.zeros(len(kpts), dtype=bool)

        for i, (kx, ky) in enumerate(kpts):
            dists = np.sqrt((nvm_xy[:, 0] - kx) ** 2 + (nvm_xy[:, 1] - ky) ** 2)
            j = np.argmin(dists)
            if dists[j] < radius:
                pid = nvm_pids[j]
                if pid in self.point3D_xyz:
                    points3D[i] = self.point3D_xyz[pid]
                    valid[i] = True

        return points3D, valid
