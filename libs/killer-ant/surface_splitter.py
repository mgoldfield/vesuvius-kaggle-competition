"""
Surface splitting post-processing for 3D segmentation.

Adapted from hengck23's "Killer Ant / Marching Ants" algorithm.
Original: https://www.kaggle.com/datasets/hengck23/vesuvius-surface-marching-ants

Detects merged papyrus sheets in binary segmentation masks using raycasting,
then splits them using Dijkstra3D shortest paths. Model-agnostic — works on
any binary mask output.

Dependencies: cc3d, dijkstra3d, scipy, numpy
"""
import numpy as np
import scipy.ndimage as ndi
import cc3d
import dijkstra3d


# ── Parameters ──────────────────────────────────────────────
RAYCAST_IOU = 0.1
SPLIT_RANGE = 20
MIN_POINT_LENGTH = 8
SPLIT_2_MAX_TRIAL = 100
SPLIT_2_REMOVE_THR = 0.9
SPLIT_ALL_MAX_DEPTH = 50


# ── Helpers ─────────────────────────────────────────────────
def _add_z(point_yx, z):
    """Convert 2D yx points to 3D zyx by prepending a z coordinate."""
    return np.concatenate([
        np.full((len(point_yx), 1), fill_value=z, dtype=point_yx.dtype),
        point_yx,
    ], -1)


def _compute_iou_1d(x1, x2):
    """Compute IoU of two 1D binary arrays."""
    x1 = x1 != 0
    x2 = x2 != 0
    inter = (x1 & x2).sum()
    union = (x1 | x2).sum()
    return inter / (union + 1e-6)


def _do_ray_casting(point1, point2, h, w):
    """
    Raycast test: check if two 2D point sets overlap in YX projection.
    point1, point2: Nx2 arrays in (y, x) format.
    Returns IoU of their Y and X projections combined.
    """
    by1 = np.bincount(point1[:, 0], minlength=h)
    by2 = np.bincount(point2[:, 0], minlength=h)
    bx1 = np.bincount(point1[:, 1], minlength=w)
    bx2 = np.bincount(point2[:, 1], minlength=w)
    iou = _compute_iou_1d(
        np.concatenate([by1, bx1]),
        np.concatenate([by2, bx2]),
    )
    return iou


def _split_range(d, min_size=64):
    """Split range [0, d) into approximately equal parts of size >= min_size."""
    k = max(1, d // min_size)
    edges = np.linspace(0, d, k + 1, dtype=int)
    edges[-1] = d
    return [(edges[i], edges[i + 1]) for i in range(k)]


# ── Core algorithm ──────────────────────────────────────────
def _find_different_surface_seed(problem, iou_threshold=RAYCAST_IOU):
    """
    Check if a binary component contains multiple overlapping surfaces.

    Scans Z-slices, finds 2D connected components within each slice,
    and uses raycasting (YX projection overlap) to detect if two components
    represent different surfaces that overlap when viewed from above.

    Returns: (is_multi_surface, found_dict_or_None)
    """
    d, h, w = problem.shape
    splits = _split_range(d, min_size=SPLIT_RANGE)

    for z1, z2 in splits:
        z = (z1 + z2) // 2
        ccz = cc3d.connected_components(problem[z])

        points = []
        for i in range(1, ccz.max() + 1):
            p = np.stack(np.where(ccz == i)).T
            if len(p) >= MIN_POINT_LENGTH:
                points.append(p)

        if len(points) <= 1:
            continue

        for i1 in range(len(points)):
            for i2 in range(i1 + 1, len(points)):
                iou = _do_ray_casting(points[i1], points[i2], h, w)
                if iou >= iou_threshold:
                    return True, {
                        'z': (z1, z2, z),
                        'point1_yx': points[i1],
                        'point2_yx': points[i2],
                    }

    return False, None


def _split_problem_to_two(problem, point1_zyx, point2_zyx, is_dilate=True):
    """
    Split a binary component into two using Dijkstra shortest paths.

    Finds paths between seed points on the two surfaces, identifies the
    "bridge" voxels where paths converge, and removes them to separate
    the component. Uses dilation to widen the cut for robustness.

    Returns: (success, labeled_or_problem)
    """
    problem = problem.copy()

    for trial in range(SPLIT_2_MAX_TRIAL):
        paths = []

        k1 = np.arange(len(point1_zyx))
        np.random.shuffle(k1)
        startpoints = point1_zyx[k1[:8]]

        for (sz, sy, sx) in startpoints:
            parent = dijkstra3d.parental_field(
                np.where(problem, 1.0, 1e6).astype(np.float32),
                source=(sz, sy, sx), connectivity=26
            )
            k2 = np.arange(len(point2_zyx))
            np.random.shuffle(k2)
            endpoints = point2_zyx[k2[:8]]
            paths.extend([
                dijkstra3d.path_from_parents(parent, (ez, ey, ex))
                for (ez, ey, ex) in endpoints
            ])

        path_flat = np.concatenate(paths)
        jump = np.any(~problem[path_flat[:, 0], path_flat[:, 1], path_flat[:, 2]])

        if jump:
            separated = cc3d.connected_components(problem)
            return True, separated

        # Remove bridge voxels (most-traversed points)
        uniq, cnt = np.unique(path_flat, axis=0, return_counts=True)
        order = np.argsort(-cnt)
        uniq = uniq[order]
        cnt = cnt[order]

        threshold = SPLIT_2_REMOVE_THR * cnt.max()
        u = uniq[cnt >= threshold]
        if not is_dilate:
            problem[u[:, 0], u[:, 1], u[:, 2]] = False
        else:
            larger = np.zeros_like(problem, dtype=bool)
            larger[u[:, 0], u[:, 1], u[:, 2]] = True
            larger = ndi.morphology.binary_dilation(
                larger, structure=np.ones((3, 3, 3), dtype=bool), iterations=1
            )
            problem[larger] = False

    return False, problem


def _split_all_surface(problem, result, depth=0, max_depth=SPLIT_ALL_MAX_DEPTH):
    """Recursively split a component until each piece is a single surface."""
    if depth >= max_depth:
        result.append(problem)
        return

    is_multi, found = _find_different_surface_seed(problem, iou_threshold=RAYCAST_IOU)
    if not is_multi:
        result.append(problem)
        return

    point1_zyx = _add_z(found['point1_yx'], found['z'][2])
    point2_zyx = _add_z(found['point2_yx'], found['z'][2])
    success, solved = _split_problem_to_two(problem, point1_zyx, point2_zyx, is_dilate=True)
    if not success:
        result.append(problem)
        return

    _split_all_surface(solved == 1, result, depth + 1, max_depth)
    _split_all_surface(solved == 2, result, depth + 1, max_depth)


# ── Public API ──────────────────────────────────────────────
def split_merged_surfaces(binary_mask, min_component_size=100):
    """
    Split merged papyrus surfaces in a binary segmentation mask.

    Takes a binary mask (0/1 uint8), finds 3D connected components,
    checks each for multiple overlapping surfaces via raycasting,
    and splits them using Dijkstra shortest paths.

    Args:
        binary_mask: 3D numpy array (uint8, values 0 or 1)
        min_component_size: ignore components smaller than this

    Returns:
        Instance-labeled array (uint8) where each separate surface
        has a unique label. Background is 0.
    """
    labeled = cc3d.connected_components(binary_mask.astype(np.uint8))
    n_components = labeled.max()

    if n_components == 0:
        return binary_mask

    all_surfaces = []
    for comp_id in range(1, n_components + 1):
        component = (labeled == comp_id)
        if component.sum() < min_component_size:
            # Keep small components as-is (single surface assumed)
            all_surfaces.append(component)
            continue

        # Try to split this component
        result = []
        _split_all_surface(component, result)
        all_surfaces.extend(result)

    # Reassemble into instance-labeled output
    output = np.zeros_like(binary_mask, dtype=np.uint8)
    for i, surface in enumerate(all_surfaces):
        # Clamp label to uint8 range (max 254 surfaces + background)
        label = min(i + 1, 254)
        output[surface > 0] = label

    return output


def split_merged_surfaces_binary(binary_mask, min_component_size=100):
    """
    Like split_merged_surfaces but returns a binary mask (0/1 uint8).

    The splitting may remove bridge voxels between merged surfaces,
    so the output binary mask can have fewer foreground voxels than
    the input.

    Args:
        binary_mask: 3D numpy array (uint8, values 0 or 1)
        min_component_size: ignore components smaller than this

    Returns:
        Binary mask (uint8) with merged surfaces split apart.
    """
    instance_labels = split_merged_surfaces(binary_mask, min_component_size)
    return (instance_labels > 0).astype(np.uint8)
