from extensions.matchers.base import BaseMatcher
from extensions import register_matcher
import numpy as np


@register_matcher("NNMatcher")
class NNMatcher(BaseMatcher):
    """
    Nearest neighbor matcher with mutual nearest neighbor (cross-check)
    and optional Lowe's ratio test on Euclidean distances.
    """

    DEFAULT_CONFIG = {
        "cross_check": True,
        "ratio_test": 0.9,
        "use_ratio_test": False,
    }

    def _load_model(self):
        pass

    def match(self, query_data, db_data):
        q_descs = query_data["descriptors"]  # (N, D)
        db_descs = db_data["descriptors"]    # (M, D)

        if len(q_descs) == 0 or len(db_descs) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # Euclidean distance: d^2 = 2 - 2*cos_sim (for L2-normalized descriptors)
        sims = q_descs @ db_descs.T
        dists = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))

        # For each query, find nearest DB descriptor
        nn_db_idx = np.argmin(dists, axis=1)       # (N,)
        nn_db_dist = np.take_along_axis(dists, nn_db_idx[:, None], axis=1).flatten()

        valid = np.ones(len(nn_db_idx), dtype=bool)

        # Mutual nearest neighbor (cross-check)
        if self.config.get("cross_check", True):
            # For each DB descriptor, find nearest query descriptor
            nn_q_idx = np.argmin(dists, axis=0)      # (M,)
            # A match (qi, di) is mutual if nn_q_idx[di] == qi
            for qi, di in enumerate(nn_db_idx):
                if nn_q_idx[di] != qi:
                    valid[qi] = False

        # Ratio test on distances (optional)
        if self.config.get("use_ratio_test", False) and dists.shape[1] >= 2:
            ratio = self.config.get("ratio_test", 0.9)
            # Find second nearest
            for qi in range(len(nn_db_idx)):
                if not valid[qi]:
                    continue
                row = dists[qi]
                best_di = nn_db_idx[qi]
                # Mask out the best match and find next min
                d1 = row[best_di]
                mask = np.ones(len(row), dtype=bool)
                mask[best_di] = False
                if mask.sum() > 0:
                    d2 = row[mask].min()
                    if d1 >= ratio * d2:
                        valid[qi] = False
                else:
                    valid[qi] = False

        q_idx = np.where(valid)[0]
        db_idx = nn_db_idx[valid]
        conf = 1.0 - nn_db_dist[valid] / 2.0  # convert dist back to similarity-like

        return q_idx.astype(np.int32), db_idx.astype(np.int32), conf.astype(np.float32)
