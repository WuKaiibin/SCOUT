import os
import glob
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from skimage import measure
from scipy.ndimage import binary_closing, binary_dilation
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QComboBox,
    QPushButton, QDialog, QTextEdit, QFileDialog, QLineEdit, QHBoxLayout,
    QInputDialog
)
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.draw import polygon, polygon_perimeter
from PIL import Image
import tifffile
import cv2
import scipy.io as sio

# -------------------------
# helper: get contours from mask robustly
def mask_to_contours(mask):
    contours = measure.find_contours(mask.astype(float), 0.3)
    filtered = [c for c in contours if c.shape[0] > 4]
    return filtered


def _parse_dims(dims):
    if dims is None:
        return None
    arr = np.array(dims).astype(int).ravel()
    if arr.size >= 2:
        return int(arr[0]), int(arr[1])
    if arr.size == 1:
        side = int(arr[0])
        return side, side
    return None


def _get_layer_store(layer):
    """Return the mutable SCOUT-specific store attached to a Shapes layer.

    Napari's ``Shapes.metadata`` property is intended for per-shape tabular
    data and may clear or coerce arbitrary dictionaries.  To keep the original
    ROI matrices available after user interactions (saving workspaces, using
    other plugins, etc.), we cache everything inside a private attribute on the
    layer.  All helpers fall back to this store and keep it in sync when
    updating values.
    """

    if not isinstance(layer, napari.layers.Shapes):
        return {}
    store = getattr(layer, "_scout_store", None)
    if store is None:
        store = {}
        setattr(layer, "_scout_store", store)
    return store


def _update_layer_store(layer, updates):
    if not isinstance(layer, napari.layers.Shapes):
        return
    store = _get_layer_store(layer)
    store.update(updates)
    setattr(layer, "_scout_store", store)


def _ensure_edge_color(layer, base_color=(0.5, 0.5, 0.5, 1.0)):
    n_shapes = len(layer.data)
    if n_shapes == 0:
        return np.empty((0, 4), dtype=float)
    try:
        edge_color = np.array(layer.edge_color, dtype=float)
    except Exception:
        edge_color = np.zeros((0, 4), dtype=float)
    if edge_color.shape[0] != n_shapes:
        edge_color = np.tile(np.array(base_color, dtype=float), (n_shapes, 1))
    return edge_color


def _get_roi_map_array(layer):
    if not isinstance(layer, napari.layers.Shapes):
        return np.empty((0,), dtype=int)
    meta = _get_layer_store(layer)
    raw_roi_map = meta.get('roi_map', [])
    roi_map = np.array(raw_roi_map, dtype=int).ravel()
    n_shapes = len(layer.data)
    if roi_map.size == 0 and n_shapes > 0:
        orig_ids = np.array(meta.get('orig_ids', []), dtype=int)
        if orig_ids.size >= n_shapes:
            roi_map = np.arange(n_shapes, dtype=int)
            print(f"ℹ️ {layer.name}: roi_map 缺失，使用顺序索引重建。")
        else:
            roi_map = -np.ones(n_shapes, dtype=int)
            print(f"ℹ️ {layer.name}: 未找到 roi_map，使用 -1 填充 {n_shapes} 个轮廓。")
    if roi_map.size < n_shapes:
        pad_len = n_shapes - roi_map.size
        pad = -np.ones(pad_len, dtype=int)
        roi_map = np.concatenate([roi_map, pad])
        print(
            f"ℹ️ {layer.name}: roi_map 长度({roi_map.size - pad_len}) < 轮廓数量({n_shapes})，"
            f"已为新增的 {pad_len} 个轮廓填充 -1。"
        )
    elif roi_map.size > n_shapes:
        print(
            f"ℹ️ {layer.name}: roi_map 长度({roi_map.size}) 超过轮廓数量({n_shapes})，"
            "已自动截断以保持一致。"
        )
        roi_map = roi_map[:n_shapes]
    if roi_map.size != 0:
        meta['roi_map'] = roi_map.astype(int)
        _update_layer_store(layer, meta)
    return roi_map


def _find_contours_for_orig(layer, target_orig_id):
    roi_map = _get_roi_map_array(layer)
    meta = _get_layer_store(layer)
    orig_ids = np.array(meta.get('orig_ids', []), dtype=int)
    indices = []
    if roi_map.size == 0:
        return indices
    for idx, orig_idx in enumerate(roi_map):
        if orig_idx < 0:
            continue
        if orig_idx < orig_ids.size:
            actual = int(orig_ids[orig_idx])
        else:
            actual = int(orig_idx)
        if actual == target_orig_id:
            indices.append(idx)
    return indices


def cache_layer_base_edge_color(layer):
    if not isinstance(layer, napari.layers.Shapes):
        return
    try:
        base = np.array(layer.edge_color, dtype=float)
    except Exception:
        base = _ensure_edge_color(layer)
    meta = _get_layer_store(layer)
    meta['base_edge_color'] = base
    _update_layer_store(layer, meta)


def refresh_layer_match_colors(layer):
    if not isinstance(layer, napari.layers.Shapes):
        return
    meta = _get_layer_store(layer)
    matches = meta.get('matches', {})
    if isinstance(matches, list):
        matches = {int(i): info for i, info in enumerate(matches)}
    base = np.array(meta.get('base_edge_color'), dtype=float) if 'base_edge_color' in meta else None
    if base is None or base.shape[0] != len(layer.data):
        base = _ensure_edge_color(layer)
        meta['base_edge_color'] = base
        _update_layer_store(layer, meta)
    edge_color = np.array(base, dtype=float)
    if isinstance(matches, dict):
        for key, info in matches.items():
            try:
                orig_id = int(key)
            except Exception:
                continue
            color = np.array(info.get('color', [0.6, 0.6, 0.6, 1.0]), dtype=float)
            contours = _find_contours_for_orig(layer, orig_id)
            for ci in contours:
                if 0 <= ci < edge_color.shape[0]:
                    edge_color[ci] = color
    layer.edge_color = edge_color


def remove_matches_between(layer, other_layer_name, source_filter=None):
    if not isinstance(layer, napari.layers.Shapes):
        return []
    meta = _get_layer_store(layer)
    raw_matches = meta.get('matches', {})
    if isinstance(raw_matches, dict):
        matches = dict(raw_matches)
    elif isinstance(raw_matches, list):
        matches = {int(i): info for i, info in enumerate(raw_matches)}
    else:
        matches = {}
    removed = []
    for key, info in list(matches.items()):
        target_layer = info.get('target_layer')
        source = info.get('source')
        if target_layer == other_layer_name and (source_filter is None or source == source_filter):
            removed.append(int(key))
            matches.pop(key, None)
    if removed:
        meta['matches'] = matches
        _update_layer_store(layer, meta)
    raw_auto = meta.get('auto_matches', {})
    auto_matches = dict(raw_auto) if isinstance(raw_auto, dict) else {}
    changed = False
    for key in list(auto_matches.keys()):
        info = auto_matches[key]
        if info.get('match_layer') == other_layer_name and (source_filter is None or info.get('source', 'auto') == source_filter):
            auto_matches.pop(key, None)
            changed = True
    if changed:
        meta['auto_matches'] = auto_matches
        _update_layer_store(layer, meta)
    return removed


def store_match_entry(layer, orig_id, other_layer_name, other_orig_id, color, source, score=None):
    if not isinstance(layer, napari.layers.Shapes):
        return
    meta = _get_layer_store(layer)
    raw_matches = meta.get('matches', {})
    if isinstance(raw_matches, dict):
        matches = dict(raw_matches)
    elif isinstance(raw_matches, list):
        matches = {int(i): info for i, info in enumerate(raw_matches)}
    else:
        matches = {}
    entry = {
        'target_layer': other_layer_name,
        'target_orig_id': int(other_orig_id),
        'color': [float(c) for c in color],
        'source': source,
    }
    if score is not None:
        entry['score'] = float(score)
    matches[int(orig_id)] = entry
    meta['matches'] = matches
    raw_auto = meta.get('auto_matches', {})
    auto_matches = dict(raw_auto) if isinstance(raw_auto, dict) else {}
    if source == 'auto':
        auto_entry = {
            'match_layer': other_layer_name,
            'match_orig_id': int(other_orig_id),
            'source': source,
        }
        if score is not None:
            auto_entry['score'] = float(score)
        auto_matches[int(orig_id)] = auto_entry
        meta['auto_matches'] = auto_matches
    _update_layer_store(layer, meta)


def compute_layer_features(layer):
    metadata = _get_layer_store(layer)
    layer_name = getattr(layer, 'name', 'unknown')
    A_full = metadata.get('A_full_orig')
    use_view = False
    if A_full is None:
        A_view = metadata.get('A_view')
        if A_view is not None:
            A_full = np.asarray(A_view)
            use_view = True
    if A_full is None:
        print(f"⚠️ {layer_name}: 元数据缺少空间矩阵 A，无法计算特征。")
        return None
    A_full = np.asarray(A_full)
    if A_full.ndim != 2 or A_full.size == 0:
        print(f"⚠️ {layer_name}: 空间矩阵 A 形状异常 {A_full.shape}。")
        return None
    C_full = metadata.get('C_full_orig')
    if C_full is None:
        C_view = metadata.get('C_view')
        if C_view is not None:
            C_full = np.asarray(C_view)
    kept_mask = metadata.get('kept_mask')
    if kept_mask is None or use_view:
        kept_mask = np.ones(A_full.shape[1], dtype=bool)
    else:
        kept_mask = np.asarray(kept_mask, dtype=bool)
        if kept_mask.size != A_full.shape[1]:
            # fall back to all available if mismatch
            print(
                f"ℹ️ {layer_name}: kept_mask 长度 {kept_mask.size} 与 A_full 列数 {A_full.shape[1]} 不符，使用全部 ROI。"
            )
            kept_mask = np.ones(A_full.shape[1], dtype=bool)
    kept_indices = np.where(kept_mask)[0]
    if kept_indices.size == 0:
        print(f"⚠️ {layer_name}: kept_mask 没有有效 ROI。")
        return None
    dims = _parse_dims(metadata.get('dims'))
    if dims is None:
        roi_masks = metadata.get('roi_masks')
        if roi_masks is not None and len(roi_masks) > 0:
            dims = tuple(np.array(roi_masks[0]).shape[:2])
        else:
            print(f"⚠️ {layer_name}: 缺少图像尺寸信息 dims，且 roi_masks 为空。")
            return None
    H, W = dims
    try:
        footprints = A_full[:, kept_indices].astype(np.float32)
    except Exception:
        return None
    footprints = footprints.reshape(H, W, -1, order='F')
    flat = footprints.reshape(-1, footprints.shape[-1])
    flat_mean = flat.mean(axis=0, keepdims=True)
    flat -= flat_mean
    flat_norm = np.linalg.norm(flat, axis=0, keepdims=True)
    flat_norm[flat_norm == 0] = 1.0
    flat_normed = flat / flat_norm

    weights = np.maximum(footprints, 0)
    ys, xs = np.indices((H, W))
    denom = weights.reshape(-1, weights.shape[-1]).sum(axis=0)
    denom[denom == 0] = 1.0
    centroid_y = (weights * ys[:, :, None]).reshape(-1, weights.shape[-1]).sum(axis=0) / denom
    centroid_x = (weights * xs[:, :, None]).reshape(-1, weights.shape[-1]).sum(axis=0) / denom
    centroids = np.stack([centroid_y, centroid_x], axis=1)

    areas = weights.reshape(-1, weights.shape[-1]).sum(axis=0)

    traces_norm = None
    if C_full is not None:
        traces = np.asarray(C_full, dtype=np.float32)[kept_indices, :]
        if traces.ndim == 1:
            traces = traces[:, None]
        traces = traces - traces.mean(axis=1, keepdims=True)
        trace_norm = np.linalg.norm(traces, axis=1, keepdims=True)
        trace_norm[trace_norm == 0] = 1.0
        traces_norm = traces / trace_norm

    orig_ids = np.array(metadata.get('orig_ids', []), dtype=int)
    if orig_ids.size == 0:
        actual_ids = kept_indices.astype(int)
    else:
        max_idx = kept_indices.max() if kept_indices.size else -1
        if orig_ids.size <= max_idx:
            actual_ids = kept_indices.astype(int)
        else:
            actual_ids = orig_ids[kept_indices]

    return {
        'flat_normed': flat_normed,
        'centroids': np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0),
        'areas': np.maximum(areas, 1e-6),
        'traces_norm': traces_norm,
        'orig_ids': actual_ids.astype(int),
        'kept_indices': kept_indices.astype(int),
        'dims': (H, W)
    }


def parse_weight_text(text, default=(0.5, 0.3, 0.1, 0.1)):
    if text is None:
        return default
    stripped = text.strip()
    if not stripped:
        return default
    try:
        parts = [float(x.strip()) for x in stripped.replace(';', ',').split(',') if x.strip()]
    except ValueError:
        print("⚠️ 权重格式错误，使用默认值")
        return default
    if len(parts) == 3:
        parts.append(default[3])
    if len(parts) != 4:
        print("⚠️ 需要 3 或 4 个权重值，使用默认值")
        return default
    total = sum(parts)
    if total <= 0:
        return default
    return tuple(parts)


def compute_auto_matches(features_a, features_b, max_dist=45.0, min_score=0.3, weights=(0.5, 0.3, 0.1, 0.1)):
    if features_a is None or features_b is None:
        return [], None
    flat_a = features_a['flat_normed']
    flat_b = features_b['flat_normed']
    spatial = flat_a.T @ flat_b
    spatial = np.clip(spatial, -1.0, 1.0)

    if features_a['traces_norm'] is not None and features_b['traces_norm'] is not None:
        temporal = features_a['traces_norm'] @ features_b['traces_norm'].T
        temporal = np.clip(temporal, -1.0, 1.0)
    else:
        temporal = np.zeros_like(spatial)

    areas_a = features_a['areas']
    areas_b = features_b['areas']
    area_min = np.minimum.outer(areas_a, areas_b)
    area_max = np.maximum.outer(areas_a, areas_b)
    area_ratio = np.divide(area_min, area_max, out=np.zeros_like(area_min), where=area_max > 0)

    centroids_a = features_a['centroids']
    centroids_b = features_b['centroids']
    if centroids_a.shape[0] == 0 or centroids_b.shape[0] == 0:
        return [], None
    distance = cdist(centroids_a, centroids_b)
    if max_dist is None or max_dist <= 0:
        dist_score = np.ones_like(distance)
        invalid = np.zeros_like(distance, dtype=bool)
    else:
        sigma = max_dist / 3.0 if max_dist > 0 else 1.0
        sigma = max(sigma, 1.0)
        dist_score = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        invalid = distance > max_dist

    w_spatial, w_temporal, w_area, w_dist = weights
    combined = w_spatial * spatial + w_temporal * temporal + w_area * area_ratio + w_dist * dist_score
    combined[invalid] = -1e6

    if combined.size == 0:
        return [], combined

    cost = -combined
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        score = combined[r, c]
        if score >= min_score and combined[r, c] > -1e5:
            matches.append({
                'idx_a': int(r),
                'idx_b': int(c),
                'score': float(score),
                'spatial': float(spatial[r, c]),
                'temporal': float(temporal[r, c]),
                'distance': float(distance[r, c]),
                'area_ratio': float(area_ratio[r, c])
            })
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches, combined


def apply_matches(layer_a, layer_b, features_a, features_b, matches, colors=None, source='auto'):
    if not matches:
        remove_matches_between(layer_a, layer_b.name, source_filter=source)
        remove_matches_between(layer_b, layer_a.name, source_filter=source)
        refresh_layer_match_colors(layer_a)
        refresh_layer_match_colors(layer_b)
        return

    remove_matches_between(layer_a, layer_b.name, source_filter=source)
    remove_matches_between(layer_b, layer_a.name, source_filter=source)

    cmap_local = cm.get_cmap('tab20', max(1, len(matches)))

    for idx, match in enumerate(matches):
        if colors is not None and idx < len(colors):
            color = np.array(colors[idx], dtype=float)
        else:
            color = np.array(cmap_local(idx % cmap_local.N))
        if color.size == 3:
            color = np.concatenate([color, [1.0]])
        if color.size == 4:
            color[3] = 1.0
        orig_a = int(features_a['orig_ids'][match['idx_a']])
        orig_b = int(features_b['orig_ids'][match['idx_b']])
        store_match_entry(layer_a, orig_a, layer_b.name, orig_b, color, source, score=match.get('score'))
        store_match_entry(layer_b, orig_b, layer_a.name, orig_a, color, source, score=match.get('score'))

    refresh_layer_match_colors(layer_a)
    refresh_layer_match_colors(layer_b)


# -------------------------
# load h5 -> create a shapes layer and metadata that uses permanent original IDs
def load_h5_to_viewer(h5_path, viewer):
    with h5py.File(h5_path, "r") as f:
        H, W = f["dims"][:]
        if "A_dense" in f["estimates"]:
            A = f["estimates/A_dense"][:]
        else:
            A_data = f["estimates/A/data"][:]
            A_indices = f["estimates/A/indices"][:]
            A_indptr = f["estimates/A/indptr"][:]
            A_shape = f["estimates/A/shape"][:]
            A_csr = csr_matrix((A_data, A_indices, A_indptr), shape=(A_shape[1], A_shape[0]))
            A = A_csr.toarray().T
        C = f["estimates/C"][:]

    n_pixels, n_rois = A.shape
    dims = (H, W)

    roi_masks = []
    roi_contours = []
    roi_colors = []
    roi_map = []  # contour -> original roi id
    cmap_local = cm.get_cmap("viridis")

    for i in range(n_rois):
        mask = A[:, i].reshape(dims, order="F")
        roi_masks.append(mask)
        mask_norm = mask / (mask.max() if mask.max() > 0 else 1)
        binary_mask = mask_norm > 0.5
        binary_mask = binary_closing(binary_mask, iterations=2)
        binary_mask = binary_dilation(binary_mask, iterations=1)
        contours = mask_to_contours(binary_mask)
        if len(contours) == 0:
            ys, xs = np.where(binary_mask)
            if len(ys) > 3:
                pts = np.column_stack([ys, xs])
                roi_contours.append(pts.astype(float))
                roi_colors.append(cmap_local(i / max(1, n_rois)))
                roi_map.append(i)
        else:
            for c in contours:
                roi_contours.append(c)
                roi_colors.append(cmap_local(i / max(1, n_rois)))
                roi_map.append(i)

    # create shapes layer
    shapes_layer = viewer.add_shapes(
        roi_contours,
        shape_type="polygon",
        edge_color=np.array(roi_colors) if len(roi_colors) else np.array([[0.5, 0.5, 0.5, 1]]),
        face_color=np.array([[0, 0, 0, 0]] * max(1, len(roi_contours))),
        edge_width=2,
        name=os.path.basename(h5_path),
        opacity=1.0,
    )
    shapes_layer.mode = 'select'
    shapes_layer.editable = True

    # metadata: preserve original full matrices and orig IDs; kept_mask marks which orig IDs are kept
    orig_ids = np.arange(n_rois, dtype=int)
    _update_layer_store(
        shapes_layer,
        {
            'roi_masks': np.array(roi_masks),   # indexed by original id
            'roi_map': np.array(roi_map, dtype=int),  # contour -> original id
            'orig_ids': orig_ids,               # original ids (permanent)
            'C_full_orig': np.array(C),         # DO NOT modify this in place
            'A_full_orig': np.array(A),         # DO NOT modify
            'kept_mask': np.ones(n_rois, dtype=bool),  # which original ids still exist
            'dims': np.array(dims)
        },
    )
    try:
        shapes_layer.metadata = {}
    except Exception:
        pass

    print(f"Loaded {os.path.basename(h5_path)}: n_rois={n_rois}, contours={len(roi_contours)}, C.shape={C.shape}")
    cache_layer_base_edge_color(shapes_layer)
    return shapes_layer


# -------------------------
viewer = napari.Viewer()
shapes_layers = []

# Optional: if you want automatic folder loading, set folder_path; otherwise use UI import
folder_path = r"D:\实验数据\cnmfe_h5"
if os.path.isdir(folder_path):
    h5_paths = glob.glob(os.path.join(folder_path, "*.h5")) + glob.glob(os.path.join(folder_path, "*.hdf5"))
else:
    h5_paths = []

# load backgrounds if any (optional)
background_folder = r"\\NIEL\home\吴锴镔实验组数据\吴锴镔实验组\relief\钙成像数据\ACC\rawdata\2025年钙成像\attack chuli\example\std_png"
if os.path.isdir(background_folder):
    png_paths = sorted(glob.glob(os.path.join(background_folder, "*.png")))
    for i, p in enumerate(png_paths):
        try:
            img = np.array(Image.open(p).convert("L"))
            viewer.add_image(img, name=f"Background {i+1}", blending="additive", opacity=0.8, visible=True)
        except Exception:
            pass

# auto-load files found in folder_path (optional)
for p in h5_paths:
    try:
        layer = load_h5_to_viewer(p, viewer)
        shapes_layers.append(layer)
    except Exception as e:
        print("Load failed:", p, e)


# -------------------------
# Control panel widget
class ROIControlPanel(QWidget):
    def __init__(self, viewer, shapes_layers):
        super().__init__()
        self.viewer = viewer
        self.shapes_layers = shapes_layers
        self.match_cmap = cm.get_cmap('tab20')
        self.match_color_counter = 0

        layout = QVBoxLayout()

        # display controls
        self.edge_width = 2
        self.fill_alpha = 0.8

        layout.addWidget(QLabel("ROI 色阶"))
        self.combo_colormap = QComboBox()
        self.colormap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cool', 'hot', 'jet']
        self.combo_colormap.addItems(self.colormap_list)
        layout.addWidget(self.combo_colormap)

        layout.addWidget(QLabel("显示模式"))
        self.combo_display_mode = QComboBox()
        self.combo_display_mode.addItems(["彩边中空", "黑边填充", "单色中空"])
        layout.addWidget(self.combo_display_mode)

        self.checkbox_data = QCheckBox("点击 ROI 弹出数据")
        self.checkbox_data.setChecked(True)
        layout.addWidget(self.checkbox_data)

        self.checkbox_plot = QCheckBox("点击 ROI 弹出曲线图")
        self.checkbox_plot.setChecked(True)
        layout.addWidget(self.checkbox_plot)

        # actions
        self.btn_save_img = QPushButton("保存当前 ROI 图像")
        layout.addWidget(self.btn_save_img)

        self.btn_delete = QPushButton("删除选中 ROI")
        layout.addWidget(self.btn_delete)

        self.btn_recolor = QPushButton("手动重排 ROI 颜色")
        layout.addWidget(self.btn_recolor)

        self.btn_export = QPushButton("导出当前层 C/A (.mat)")
        layout.addWidget(self.btn_export)

        layout.addSpacing(6)
        # imports
        layout.addWidget(QLabel("数据导入"))
        btn_import_h5 = QPushButton("导入原始数据 (.h5)")
        btn_import_mat = QPushButton("导入处理数据 (.mat)")
        btn_import_png = QPushButton("导入背景 (.png 文件夹)")
        layout.addWidget(btn_import_h5)
        layout.addWidget(btn_import_mat)
        layout.addWidget(btn_import_png)

        layout.addSpacing(6)
        # color by orig ids
        layout.addWidget(QLabel("按 原始 ROI id 修改颜色（逗号分隔）"))
        hl = QHBoxLayout()
        self.input_roi_indices = QLineEdit()
        self.input_roi_indices.setPlaceholderText("例如：0,3,7")
        self.input_rgb = QLineEdit()
        self.input_rgb.setPlaceholderText("例如：255,0,0 或 #FF0000")
        btn_apply_color = QPushButton("修改颜色")
        hl.addWidget(self.input_roi_indices)
        hl.addWidget(self.input_rgb)
        hl.addWidget(btn_apply_color)
        layout.addLayout(hl)

        layout.addSpacing(6)
        # segments
        layout.addWidget(QLabel("帧段候选（格式: start-end, start-end）"))
        self.input_segments = QLineEdit()
        self.input_segments.setPlaceholderText("例如：0-500,800-1200")
        btn_update_segments = QPushButton("更新帧段按钮")
        self.combo_segments = QComboBox()
        layout.addWidget(self.input_segments)
        layout.addWidget(btn_update_segments)
        layout.addWidget(QLabel("选择帧段"))
        layout.addWidget(self.combo_segments)

        layout.addSpacing(6)
        # tiff overlay
        btn_import_tiff = QPushButton("导入 TIFF 视频并叠加")
        layout.addWidget(btn_import_tiff)

        layout.addSpacing(8)
        layout.addWidget(QLabel("自动配准（SCOUT 特征融合）"))
        layout.addWidget(QLabel("参考 session"))
        self.combo_ref_layer = QComboBox()
        layout.addWidget(self.combo_ref_layer)
        layout.addWidget(QLabel("待配准 session"))
        self.combo_target_layer = QComboBox()
        layout.addWidget(self.combo_target_layer)

        auto_params_layout = QHBoxLayout()
        self.input_max_dist = QLineEdit()
        self.input_max_dist.setPlaceholderText("最大距离, 默认45")
        self.input_min_score = QLineEdit()
        self.input_min_score.setPlaceholderText("最小得分, 默认0.3")
        auto_params_layout.addWidget(self.input_max_dist)
        auto_params_layout.addWidget(self.input_min_score)
        layout.addLayout(auto_params_layout)

        self.input_weights = QLineEdit()
        self.input_weights.setPlaceholderText("权重: spatial,temporal,area,distance")
        layout.addWidget(self.input_weights)

        self.btn_auto_match = QPushButton("运行自动配准")
        layout.addWidget(self.btn_auto_match)

        layout.addSpacing(8)
        layout.addWidget(QLabel("手动匹配调整"))
        layout.addWidget(QLabel("Session A"))
        self.combo_manual_layer_a = QComboBox()
        layout.addWidget(self.combo_manual_layer_a)
        layout.addWidget(QLabel("Session B"))
        self.combo_manual_layer_b = QComboBox()
        layout.addWidget(self.combo_manual_layer_b)
        manual_pair_layout = QHBoxLayout()
        self.input_manual_roi_a = QLineEdit()
        self.input_manual_roi_a.setPlaceholderText("ROI ID A")
        self.input_manual_roi_b = QLineEdit()
        self.input_manual_roi_b.setPlaceholderText("ROI ID B")
        manual_pair_layout.addWidget(self.input_manual_roi_a)
        manual_pair_layout.addWidget(self.input_manual_roi_b)
        layout.addLayout(manual_pair_layout)
        manual_btn_layout = QHBoxLayout()
        self.btn_add_manual_match = QPushButton("添加手动匹配")
        self.btn_remove_manual_match = QPushButton("取消手动匹配")
        manual_btn_layout.addWidget(self.btn_add_manual_match)
        manual_btn_layout.addWidget(self.btn_remove_manual_match)
        layout.addLayout(manual_btn_layout)

        # layout finalize
        self.setLayout(layout)

        # connect
        self.combo_display_mode.currentIndexChanged.connect(self.update_all_shapes)
        self.combo_colormap.currentIndexChanged.connect(self.update_all_shapes)

        self.btn_delete.clicked.connect(self.delete_selected_rois)
        self.btn_recolor.clicked.connect(self.recolor_rois)
        self.btn_save_img.clicked.connect(self.save_current_image)
        self.btn_export.clicked.connect(self.save_current_layer_data)

        btn_import_h5.clicked.connect(self.dialog_load_h5)
        btn_import_mat.clicked.connect(self.dialog_load_mat)
        btn_import_png.clicked.connect(self.dialog_load_png)

        btn_apply_color.clicked.connect(self.apply_color_to_indices)
        btn_update_segments.clicked.connect(self.update_segment_list)
        btn_import_tiff.clicked.connect(self.import_tiff_and_overlay)
        self.btn_auto_match.clicked.connect(self.run_auto_match)
        self.btn_add_manual_match.clicked.connect(self.add_manual_match)
        self.btn_remove_manual_match.clicked.connect(self.remove_manual_match)

        # default segment list
        self.segment_list = [(0, None)]
        self.combo_segments.addItem("全段(0-end)")
        self.refresh_layer_choices()
        self.cache_all_edge_colors()
        self.refresh_all_match_colors()

    def refresh_layer_choices(self):
        names = [layer.name for layer in self.shapes_layers if isinstance(layer, napari.layers.Shapes)]
        self.combo_ref_layer.blockSignals(True)
        self.combo_target_layer.blockSignals(True)
        self.combo_manual_layer_a.blockSignals(True)
        self.combo_manual_layer_b.blockSignals(True)
        self.combo_ref_layer.clear()
        self.combo_target_layer.clear()
        self.combo_manual_layer_a.clear()
        self.combo_manual_layer_b.clear()
        if names:
            self.combo_ref_layer.addItems(names)
            self.combo_target_layer.addItems(names)
            if len(names) > 1:
                self.combo_target_layer.setCurrentIndex(1)
            self.combo_manual_layer_a.addItems(names)
            self.combo_manual_layer_b.addItems(names)
            if len(names) > 1:
                self.combo_manual_layer_b.setCurrentIndex(1)
        self.combo_ref_layer.blockSignals(False)
        self.combo_target_layer.blockSignals(False)
        self.combo_manual_layer_a.blockSignals(False)
        self.combo_manual_layer_b.blockSignals(False)

    def register_shapes_layer(self, layer):
        self.refresh_layer_choices()
        cache_layer_base_edge_color(layer)
        self.refresh_all_match_colors()

    def unregister_shapes_layer(self, layer):
        self.refresh_layer_choices()

    def cache_all_edge_colors(self):
        for layer in self.shapes_layers:
            if isinstance(layer, napari.layers.Shapes):
                cache_layer_base_edge_color(layer)

    def refresh_all_match_colors(self):
        for layer in self.shapes_layers:
            if isinstance(layer, napari.layers.Shapes):
                refresh_layer_match_colors(layer)

    def _apply_style_to_layer(self, layer):
        if not isinstance(layer, napari.layers.Shapes):
            return
        alpha = self.fill_alpha
        edge_w = self.edge_width
        mode = self.combo_display_mode.currentText()
        cmap_local = cm.get_cmap(self.combo_colormap.currentText())
        n_shapes = max(1, len(layer.data))
        colors = [cmap_local(i / max(n_shapes - 1, 1)) for i in range(n_shapes)]
        if mode == "彩边中空":
            layer.edge_color = np.array(colors)
            layer.face_color = np.array([[0, 0, 0, 0]] * n_shapes)
        elif mode == "黑边填充":
            layer.edge_color = np.array([[0, 0, 0, 1]] * n_shapes)
            layer.face_color = np.array([(r, g, b, alpha) for r, g, b, _ in colors])
        elif mode == "单色中空":
            layer.edge_color = np.array([[0.8, 0.8, 0.8, 1]] * n_shapes)
            layer.face_color = np.array([[0, 0, 0, 0]] * n_shapes)
        layer.edge_width = edge_w
        cache_layer_base_edge_color(layer)

    def _report_missing_data(self, layer):
        meta = _get_layer_store(layer)
        missing = []
        if meta.get('A_full_orig') is None and meta.get('A_view') is None:
            missing.append('空间矩阵 A')
        if meta.get('C_full_orig') is None and meta.get('C_view') is None:
            missing.append('时间矩阵 C')
        dims = _parse_dims(meta.get('dims'))
        if dims is None and meta.get('roi_masks') is None:
            missing.append('图像尺寸信息')
        if missing:
            print(f"⚠️ {layer.name} 缺少配准所需数据：{', '.join(missing)}。请重新导入包含完整估计量的 .h5 或导出 .mat 文件。")
        else:
            print(f"⚠️ {layer.name} 的元数据不完整，无法提取特征。请尝试重新导入原始文件。")

    def _next_match_color(self):
        color = np.array(self.match_cmap(self.match_color_counter % self.match_cmap.N))
        self.match_color_counter += 1
        if color.size == 3:
            color = np.concatenate([color, [1.0]])
        else:
            color = color.copy()
            color[3] = 1.0
        return color

    def _find_layer_by_name(self, name):
        for layer in self.shapes_layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == name:
                return layer
        return None

    def _get_layer_by_combo(self, combo):
        name = combo.currentText()
        if not name:
            return None
        return self._find_layer_by_name(name)

    def _get_match_info(self, layer, orig_id):
        if not isinstance(layer, napari.layers.Shapes):
            return None
        meta = _get_layer_store(layer)
        raw_matches = meta.get('matches', {})
        if isinstance(raw_matches, dict):
            matches = raw_matches
        elif isinstance(raw_matches, list):
            matches = {int(i): info for i, info in enumerate(raw_matches)}
        else:
            matches = {}
        key = int(orig_id)
        if key in matches:
            return matches[key]
        if str(key) in matches:
            return matches[str(key)]
        return None

    def _roi_exists(self, layer, orig_id):
        if not isinstance(layer, napari.layers.Shapes):
            return False
        meta = _get_layer_store(layer)
        orig_ids = np.array(meta.get('orig_ids', []), dtype=int)
        if orig_ids.size == 0:
            return False
        matches = np.where(orig_ids == orig_id)[0]
        if matches.size == 0:
            return False
        idx = matches[0]
        kept_mask = meta.get('kept_mask')
        if kept_mask is not None and idx < len(kept_mask):
            return bool(kept_mask[idx])
        return True

    def _remove_match_entry(self, layer, orig_id, update_counterpart=True):
        if not isinstance(layer, napari.layers.Shapes):
            return False
        meta = _get_layer_store(layer)
        raw_matches = meta.get('matches', {})
        if isinstance(raw_matches, dict):
            matches = dict(raw_matches)
        elif isinstance(raw_matches, list):
            matches = {int(i): info for i, info in enumerate(raw_matches)}
        else:
            matches = {}
        key = int(orig_id)
        info = matches.pop(key, None)
        if info is None:
            _update_layer_store(layer, meta)
            return False
        meta['matches'] = matches
        raw_auto = meta.get('auto_matches', {})
        auto_matches = dict(raw_auto) if isinstance(raw_auto, dict) else {}
        if key in auto_matches:
            auto_matches.pop(key, None)
            meta['auto_matches'] = auto_matches
        _update_layer_store(layer, meta)
        other_layer = None
        if update_counterpart:
            other_layer = self._find_layer_by_name(info.get('target_layer'))
            if other_layer is not None:
                other_meta = _get_layer_store(other_layer)
                raw_other_matches = other_meta.get('matches', {})
                if isinstance(raw_other_matches, dict):
                    other_matches = dict(raw_other_matches)
                elif isinstance(raw_other_matches, list):
                    other_matches = {int(i): info for i, info in enumerate(raw_other_matches)}
                else:
                    other_matches = {}
                other_key = int(info.get('target_orig_id'))
                other_matches.pop(other_key, None)
                other_meta['matches'] = other_matches
                raw_other_auto = other_meta.get('auto_matches', {})
                other_auto = dict(raw_other_auto) if isinstance(raw_other_auto, dict) else {}
                if other_key in other_auto:
                    other_auto.pop(other_key, None)
                    other_meta['auto_matches'] = other_auto
                _update_layer_store(other_layer, other_meta)
        refresh_layer_match_colors(layer)
        if other_layer is not None:
            refresh_layer_match_colors(other_layer)
        return True

    def add_manual_match(self):
        layer_a = self._get_layer_by_combo(self.combo_manual_layer_a)
        layer_b = self._get_layer_by_combo(self.combo_manual_layer_b)
        if layer_a is None or layer_b is None:
            print("⚠️ 请选择两个 session")
            return
        if layer_a == layer_b:
            print("⚠️ 手动匹配需要选择不同的 session")
            return
        text_a = self.input_manual_roi_a.text().strip()
        text_b = self.input_manual_roi_b.text().strip()
        if not text_a or not text_b:
            print("⚠️ 请填写两侧的 ROI ID")
            return
        try:
            orig_a = int(text_a)
            orig_b = int(text_b)
        except ValueError:
            print("⚠️ ROI ID 需要是整数")
            return
        if not self._roi_exists(layer_a, orig_a):
            print(f"⚠️ session {layer_a.name} 中找不到 ROI {orig_a}")
            return
        if not self._roi_exists(layer_b, orig_b):
            print(f"⚠️ session {layer_b.name} 中找不到 ROI {orig_b}")
            return
        existing_a = self._get_match_info(layer_a, orig_a)
        if existing_a is not None:
            self._remove_match_entry(layer_a, orig_a, update_counterpart=True)
        existing_b = self._get_match_info(layer_b, orig_b)
        if existing_b is not None:
            self._remove_match_entry(layer_b, orig_b, update_counterpart=True)
        color = self._next_match_color()
        store_match_entry(layer_a, orig_a, layer_b.name, orig_b, color, source='manual')
        store_match_entry(layer_b, orig_b, layer_a.name, orig_a, color, source='manual')
        refresh_layer_match_colors(layer_a)
        refresh_layer_match_colors(layer_b)
        print(f"✅ 已手动匹配 {layer_a.name}:ROI {orig_a} ↔ {layer_b.name}:ROI {orig_b}")

    def remove_manual_match(self):
        layer_a = self._get_layer_by_combo(self.combo_manual_layer_a)
        layer_b = self._get_layer_by_combo(self.combo_manual_layer_b)
        if layer_a is None or layer_b is None:
            print("⚠️ 请选择两个 session")
            return
        text_a = self.input_manual_roi_a.text().strip()
        text_b = self.input_manual_roi_b.text().strip()
        if not text_a or not text_b:
            print("⚠️ 请填写两侧的 ROI ID")
            return
        try:
            orig_a = int(text_a)
            orig_b = int(text_b)
        except ValueError:
            print("⚠️ ROI ID 需要是整数")
            return
        info_a = self._get_match_info(layer_a, orig_a)
        if info_a is None or info_a.get('target_layer') != layer_b.name or int(info_a.get('target_orig_id', -1)) != orig_b:
            print("⚠️ 未找到指定的匹配关系")
            return
        self._remove_match_entry(layer_a, orig_a, update_counterpart=True)
        print(f"🗑️ 已移除匹配 {layer_a.name}:ROI {orig_a} ↔ {layer_b.name}:ROI {orig_b}")

    # update visuals
    def update_all_shapes(self):
        for layer in self.shapes_layers:
            if not isinstance(layer, napari.layers.Shapes):
                continue
            self._apply_style_to_layer(layer)
        self.refresh_all_match_colors()

    # delete: do NOT renumber orig ids. mark kept_mask False for removed orig ids and remove corresponding contours.
    def delete_selected_rois(self):
        for layer in self.shapes_layers:
            if not isinstance(layer, napari.layers.Shapes):
                continue
            selected = sorted(list(layer.selected_data), reverse=True)
            if not selected:
                continue
            n_contours = len(layer.data)
            selected_safe = [i for i in selected if 0 <= i < n_contours]
            if len(selected_safe) == 0:
                continue

            roi_map = _get_roi_map_array(layer)
            meta = _get_layer_store(layer)
            kept_mask = meta.get('kept_mask', None)
            roi_masks = meta.get('roi_masks', None)
            C_full_orig = meta.get('C_full_orig', None)
            A_full_orig = meta.get('A_full_orig', None)

            # which orig ids are removed because of removing these contours
            removed_orig_ids = [int(x) for x in np.unique(roi_map[selected_safe]).tolist() if int(x) >= 0]
            orig_ids_arr = np.array(meta.get('orig_ids', []), dtype=int)
            removed_actual_ids = []
            for rid in removed_orig_ids:
                if rid < len(orig_ids_arr):
                    removed_actual_ids.append(int(orig_ids_arr[rid]))
                else:
                    removed_actual_ids.append(int(rid))

            # mark kept_mask False for these orig ids (permanent deletion marker)
            if kept_mask is not None:
                for oid in removed_orig_ids:
                    if 0 <= oid < len(kept_mask):
                        kept_mask[oid] = False
                meta['kept_mask'] = kept_mask

            # remove contours from layer.data and update roi_map accordingly
            keep_contours = [i for i in range(n_contours) if i not in selected_safe]
            layer.data = [layer.data[i] for i in keep_contours]
            layer.selected_data.clear()

            # update roi_map to reflect only remaining contours
            new_roi_map = roi_map[keep_contours] if len(roi_map) >= len(keep_contours) else roi_map[:len(keep_contours)]
            meta['roi_map'] = np.array(new_roi_map, dtype=int)
            cache_layer_base_edge_color(layer)
            refresh_layer_match_colors(layer)

            # create a roi_masks_view (not altering original roi_masks) for convenience: masks for orig ids that are kept
            if roi_masks is not None:
                try:
                    kept_orig_idxs = np.where(meta.get('kept_mask', np.ones(1, dtype=bool)))[0]
                    new_masks = np.array([roi_masks[int(i)] for i in kept_orig_idxs])
                    meta['roi_masks_view'] = new_masks
                except Exception:
                    meta['roi_masks_view'] = None
            else:
                meta['roi_masks_view'] = None

            # create views for C/A if present
            if C_full_orig is not None:
                try:
                    kept_orig_idxs = np.where(meta.get('kept_mask', np.ones(1, dtype=bool)))[0]
                    meta['C_view'] = C_full_orig[kept_orig_idxs, :]
                except Exception:
                    meta['C_view'] = None
            if A_full_orig is not None:
                try:
                    kept_orig_idxs = np.where(meta.get('kept_mask', np.ones(1, dtype=bool)))[0]
                    meta['A_view'] = A_full_orig[:, kept_orig_idxs]
                except Exception:
                    meta['A_view'] = None

            _update_layer_store(layer, meta)

            for actual_id in removed_actual_ids:
                self._remove_match_entry(layer, actual_id, update_counterpart=True)

            self._apply_style_to_layer(layer)
            print(f"✅ 已删除 {len(selected_safe)} 个 contour； 标记 {len(removed_orig_ids)} 个原始 ROI 为已删除（kept_mask updated）")
        self.refresh_all_match_colors()

    # manual recolor
    def recolor_rois(self):
        for layer in self.shapes_layers:
            if not isinstance(layer, napari.layers.Shapes):
                continue
            n_shapes = len(layer.data)
            if n_shapes == 0:
                continue
            self._apply_style_to_layer(layer)
        print("🎨 手动重排颜色已完成。")
        self.refresh_all_match_colors()

    # export current layer data (uses kept_mask to select subset from original matrices)
    def save_current_layer_data(self):
        active_layer = self.viewer.layers.selection.active
        if active_layer is None or not isinstance(active_layer, napari.layers.Shapes):
            print("⚠️ 请先选中一个 Shapes 层")
            return
        meta = _get_layer_store(active_layer)
        C_full_orig = meta.get('C_full_orig', None)
        A_full_orig = meta.get('A_full_orig', None)
        dims = meta.get('dims', None)
        kept_mask = meta.get('kept_mask', None)
        if kept_mask is None:
            print("⚠️ 当前 layer 没有 kept_mask，导出全部原始数据")
            kept_idxs = np.arange(C_full_orig.shape[0]) if C_full_orig is not None else np.arange(A_full_orig.shape[1])
        else:
            kept_idxs = np.where(kept_mask)[0]

        C_export = C_full_orig[kept_idxs, :] if C_full_orig is not None else None
        A_export = A_full_orig[:, kept_idxs] if A_full_orig is not None else None

        save_path, _ = QFileDialog.getSaveFileName(None, "保存 ROI 数据", "", "MAT 文件 (*.mat)")
        if not save_path:
            return
        sio.savemat(save_path, {"C": C_export, "A": A_export, "kept_orig_ids": kept_idxs, "dims": dims})
        print(f"💾 已保存当前层 {len(kept_idxs)} 个 ROI 数据到: {save_path}")

    # save PNG of current shapes layer (visual)
    def save_current_image(self):
        selected_layers = [l for l in self.viewer.layers if isinstance(l, napari.layers.Shapes) and l.visible]
        if not selected_layers:
            print("⚠️ 没有可见 Shapes 层")
            return
        shapes_layer = selected_layers[0]
        path, _ = QFileDialog.getSaveFileName(None, "保存 ROI 图像", "", "PNG 文件 (*.png)")
        if not path:
            return

        meta = _get_layer_store(shapes_layer)
        roi_masks_view = meta.get('roi_masks_view', None)
        if roi_masks_view is not None and len(roi_masks_view) > 0:
            H, W = roi_masks_view[0].shape
        else:
            try:
                img_layer = next(l for l in self.viewer.layers if l.ndim == 2)
                H, W = img_layer.data.shape[-2:]
            except Exception:
                H, W = 512, 512

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        face_colors = np.array(shapes_layer.face_color)
        edge_colors = np.array(shapes_layer.edge_color)

        for i, verts in enumerate(shapes_layer.data):
            ys, xs = verts[:, 0], verts[:, 1]
            rr, cc = polygon(ys, xs, (H, W))
            if i < len(face_colors):
                fc = face_colors[i]
                if len(fc) == 4 and fc[3] > 0:
                    fc_uint8 = tuple(int(max(0, min(1, c)) * 255) for c in fc)
                    rgba[rr, cc, :3] = fc_uint8[:3]
                    rgba[rr, cc, 3] = fc_uint8[3]
            if i < len(edge_colors):
                ec = edge_colors[i]
                ec_uint8 = tuple(int(max(0, min(1, c)) * 255) for c in ec)
                rr_e, cc_e = polygon_perimeter(ys, xs, (H, W))
                rgba[rr_e, cc_e, :3] = ec_uint8[:3]
                rgba[rr_e, cc_e, 3] = 255

        Image.fromarray(rgba).save(path)
        print(f"✅ 保存成功: {path}")

    # import dialogs
    def dialog_load_h5(self):
        paths, _ = QFileDialog.getOpenFileNames(None, "选择 H5 文件", "", "HDF5 Files (*.h5 *.hdf5)")
        if not paths:
            return
        for p in paths:
            try:
                layer = load_h5_to_viewer(p, self.viewer)
                self.shapes_layers.append(layer)
                self.register_shapes_layer(layer)
            except Exception as e:
                print("导入失败:", p, e)

    def dialog_load_mat(self):
        paths, _ = QFileDialog.getOpenFileNames(None, "选择 MAT 文件", "", "MAT Files (*.mat)")
        if not paths:
            return
        for p in paths:
            try:
                data = sio.loadmat(p)
                roi_masks = []
                A = data.get('A', None)
                C = data.get('C', None)
                dims = data.get('dims', None)
                kept_orig_ids = data.get('kept_orig_ids', None)
                if A is None or C is None or dims is None or kept_orig_ids is None:
                    print(f"⚠️ {p} 缺少必要字段，跳过")
                    continue
                n_rois = C.shape[0]
                print(f"{p}: C.shape={C.shape}, A.shape={A.shape}, dims={dims}, n_rois={n_rois}")

                H, W = int(dims[0][0]), int(dims[0][1])

                roi_contours = []
                roi_colors = []
                roi_map = []
                cmap_local = cm.get_cmap('plasma')

                for i in range(n_rois):
                    mask = A[:, i].reshape((H, W), order="F")
                    roi_masks.append(mask)
                    mask_norm = mask / (mask.max() if mask.max() > 0 else 1)
                    binary_mask = mask_norm > 0.5
                    binary_mask = binary_closing(binary_mask, iterations=2)
                    binary_mask = binary_dilation(binary_mask, iterations=1)
                    contours = mask_to_contours(binary_mask)
                    if len(contours) == 0:
                        ys, xs = np.where(binary_mask)
                        if len(ys) > 3:
                            pts = np.column_stack([ys, xs])
                            roi_contours.append(pts.astype(float))
                            roi_colors.append(cmap_local(i / max(1, n_rois)))
                            roi_map.append(i)
                    else:
                        for c in contours:
                            roi_contours.append(c)
                            roi_colors.append(cmap_local(i / max(1, n_rois)))
                            roi_map.append(i)

                # create shapes layer
                shapes_layer = viewer.add_shapes(
                    roi_contours,
                    shape_type="polygon",
                    edge_color=np.array(roi_colors) if len(roi_colors) else np.array([[0.5, 0.5, 0.5, 1]]),
                    face_color=np.array([[0, 0, 0, 0]] * max(1, len(roi_contours))),
                    edge_width=2,
                    name=os.path.basename(p),
                    opacity=1.0,
                )
                shapes_layer.mode = 'select'
                shapes_layer.editable = True

                # metadata: preserve original full matrices and orig IDs; kept_mask marks which orig IDs are kept
                orig_ids = np.array(kept_orig_ids).flatten()
                _update_layer_store(
                    shapes_layer,
                    {
                        'roi_masks': np.array(roi_masks),   # indexed by original id
                        'roi_map': np.array(roi_map, dtype=int),  # contour -> original id
                        'orig_ids': orig_ids,               # original ids (permanent)
                        'C_full_orig': np.array(C),         # DO NOT modify this in place
                        'A_full_orig': np.array(A),         # DO NOT modify
                        'kept_mask': np.ones(n_rois, dtype=bool),  # which original ids still exist
                        'dims': np.array(dims)
                    },
                )
                try:
                    shapes_layer.metadata = {}
                except Exception:
                    pass

                print(f"Loaded {os.path.basename(p)}: n_rois={n_rois}, contours={len(roi_contours)}, C.shape={C.shape}")
                self.shapes_layers.append(shapes_layer)
                self.register_shapes_layer(shapes_layer)

            except Exception as e:
                print("导入 MAT 失败:", p, e)

    def dialog_load_png(self):
        folder = QFileDialog.getExistingDirectory(None, "选择 PNG 文件夹")
        if not folder:
            return
        pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
        for i, p in enumerate(pngs):
            try:
                img = np.array(Image.open(p).convert("L"))
                self.viewer.add_image(img, name=f"Background {os.path.basename(p)}", blending="additive", opacity=0.8)
            except Exception:
                pass
        print(f"✅ 导入背景图 {len(pngs)} 张")

    # apply colors to original IDs
    def apply_color_to_indices(self):
        idx_text = self.input_roi_indices.text().strip()
        rgb_text = self.input_rgb.text().strip()
        if not idx_text or not rgb_text:
            print("⚠️ 请同时输入 ROI 索引 和 RGB 值")
            return
        try:
            indices = [int(x.strip()) for x in idx_text.split(",") if x.strip() != ""]
        except Exception:
            print("⚠️ ROI 索引解析失败，请使用逗号分隔整数，例如 0,3,7")
            return
        # parse rgb: accept 'r,g,b' or '#RRGGBB'
        try:
            if rgb_text.startswith("#"):
                rgb = tuple(int(rgb_text[i:i + 2], 16) for i in (1, 3, 5))
            else:
                parts = [int(x.strip()) for x in rgb_text.split(",")]
                if len(parts) != 3:
                    raise ValueError
                rgb = tuple(parts)
            rgb = np.array(rgb, dtype=np.uint8)
        except Exception:
            print("⚠️ RGB 解析失败，请输入 3 个 0-255 的整数或 #RRGGBB")
            return
        color_norm = (rgb / 255.0).tolist() + [1.0]
        # choose target layer
        if isinstance(self.viewer.layers.selection.active, napari.layers.Shapes):
            target = self.viewer.layers.selection.active
        elif self.shapes_layers:
            target = self.shapes_layers[-1]
        else:
            target = None
        if target is None:
            print("⚠️ 未找到目标 Shapes 层")
            return
        roi_map = _get_roi_map_array(target)  # contour -> orig id
        if roi_map is None or len(roi_map) == 0:
            print("⚠️ 当前 layer 没有 roi_map，无法按原始 id 修改颜色")
            return
        # ensure edge_color length equals contours
        try:
            ec = np.array(target.edge_color)
            if ec.shape[0] != len(target.data):
                ec = np.tile(np.array([0.5, 0.5, 0.5, 1.0]), (len(target.data), 1))
        except Exception:
            ec = np.tile(np.array([0.5, 0.5, 0.5, 1.0]), (len(target.data), 1))
        meta = _get_layer_store(target)
        orig_ids = np.array(meta.get('orig_ids', []), dtype=int)
        for ci, idx_in_file in enumerate(roi_map):
            if idx_in_file < len(orig_ids):
                orig_id = int(orig_ids[idx_in_file])
            else:
                orig_id = int(idx_in_file)
            if orig_id in indices:
                ec[ci] = color_norm

        target.edge_color = ec
        cache_layer_base_edge_color(target)
        self.refresh_all_match_colors()
        print(f"✅ 已将原始 ROI id {indices} 的 contour 改色为 {rgb.tolist()}")

    # segments update
    def update_segment_list(self):
        text = self.input_segments.text().strip()
        self.combo_segments.clear()
        self.segment_list = []
        if not text:
            self.combo_segments.addItem("全段(0-end)")
            self.segment_list = [(0, None)]
            return
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        for p in parts:
            if "-" in p:
                try:
                    s, e = p.split("-", 1)
                    s_i = int(s.strip())
                    e_i = int(e.strip())
                    if s_i < 0:
                        s_i = 0
                    if e_i < s_i:
                        e_i = s_i
                    self.segment_list.append((s_i, e_i))
                    self.combo_segments.addItem(f"{s_i}-{e_i}")
                except Exception:
                    continue
        if not self.segment_list:
            self.segment_list = [(0, None)]
            self.combo_segments.addItem("全段(0-end)")

    # import tiff and overlay (choose frame by segment selection or ask)
    def import_tiff_and_overlay(self):
        tiff_path, _ = QFileDialog.getOpenFileName(None, "选择 TIFF 视频", "", "TIFF Files (*.tif *.tiff)")
        if not tiff_path:
            return
        # choose target shapes layer
        shape_layers = [l for l in self.viewer.layers if isinstance(l, napari.layers.Shapes)]
        if not shape_layers:
            print("⚠️ 未发现 Shapes 层")
            return
        if isinstance(self.viewer.layers.selection.active, napari.layers.Shapes):
            target = self.viewer.layers.selection.active
        else:
            target = shape_layers[-1]

        seg_idx = self.combo_segments.currentIndex()
        if hasattr(self, 'segment_list') and 0 <= seg_idx < len(self.segment_list):
            s, e = self.segment_list[seg_idx]
            if e is None:
                frame_idx, ok = QInputDialog.getInt(self, "选择帧", "输入要叠加的帧索引 (0-based):", 0, 0, 100000000, 1)
                if not ok:
                    return
                frame_to_overlay = frame_idx
            else:
                frame_to_overlay = (s + e) // 2
        else:
            frame_to_overlay = 0

        save_path, _ = QFileDialog.getSaveFileName(None, "保存叠加结果", "", "TIFF Files (*.tif *.tiff)")
        if not save_path:
            return

        frames = tifffile.imread(tiff_path)
        if frames.ndim == 2:
            frames = frames[np.newaxis, ...]
        elif frames.ndim == 3 and frames.shape[0] < frames.shape[-1]:
            if frames.shape[-1] < frames.shape[0]:
                frames = np.moveaxis(frames, -1, 0)
        n_frames, Hf, Wf = frames.shape
        frame_to_overlay = max(0, min(n_frames - 1, frame_to_overlay))

        meta = _get_layer_store(target)
        roi_masks_view = meta.get('roi_masks_view', None)
        if roi_masks_view is not None:
            rm = np.array(roi_masks_view)
            if rm.ndim == 3:
                mask_total = np.any(rm > 0, axis=0).astype(np.uint8)
            else:
                mask_total = np.zeros((Hf, Wf), dtype=np.uint8)
                for m in roi_masks_view:
                    mm = np.array(m)
                    if mm.shape == mask_total.shape:
                        mask_total |= (mm > 0).astype(np.uint8)
        else:
            mask_total = np.zeros((Hf, Wf), dtype=np.uint8)
            for cont in target.data:
                rr = np.round(cont[:, 0]).astype(int)
                cc = np.round(cont[:, 1]).astype(int)
                rr = np.clip(rr, 0, Hf - 1)
                cc = np.clip(cc, 0, Wf - 1)
                mask_total[rr, cc] = 1

        out_stack = []
        for fi in range(n_frames):
            frame = frames[fi]
            if frame.dtype != np.uint8:
                fmin, fmax = float(frame.min()), float(frame.max())
                if fmax > fmin:
                    frame_u8 = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
                else:
                    frame_u8 = np.zeros_like(frame, dtype=np.uint8)
            else:
                frame_u8 = frame.copy()
            img_rgb = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
            if fi == frame_to_overlay:
                mask_idx = mask_total.astype(bool)
                overlay_color = np.array([200, 200, 200], dtype=np.uint8)
                alpha = 0.45
                img_rgb[mask_idx] = (1 - alpha) * img_rgb[mask_idx] + alpha * overlay_color
            out_stack.append(img_rgb.astype(np.uint8))

        out_stack = np.stack(out_stack, axis=0)
        tifffile.imwrite(save_path, out_stack, photometric='rgb')
        print(f"✅ 已保存叠加视频到 {save_path}（叠加帧: {frame_to_overlay}）")

    def run_auto_match(self):
        if len(self.shapes_layers) < 2:
            print("⚠️ 至少需要两个 session 才能自动配准")
            return
        ref_idx = self.combo_ref_layer.currentIndex()
        tgt_idx = self.combo_target_layer.currentIndex()
        if ref_idx < 0 or tgt_idx < 0:
            print("⚠️ 请选择参考和目标 session")
            return
        if ref_idx == tgt_idx:
            print("⚠️ 参考 session 与目标 session 不能相同")
            return
        try:
            max_dist = float(self.input_max_dist.text()) if self.input_max_dist.text().strip() else 45.0
        except ValueError:
            print("⚠️ 最大距离输入无效，使用默认 45 像素")
            max_dist = 45.0
        try:
            min_score = float(self.input_min_score.text()) if self.input_min_score.text().strip() else 0.3
        except ValueError:
            print("⚠️ 最小得分输入无效，使用默认 0.3")
            min_score = 0.3
        weights = parse_weight_text(self.input_weights.text())

        ref_layer = self.shapes_layers[ref_idx]
        tgt_layer = self.shapes_layers[tgt_idx]
        if not isinstance(ref_layer, napari.layers.Shapes) or not isinstance(tgt_layer, napari.layers.Shapes):
            print("⚠️ 选中的 layer 不是 Shapes 类型")
            return

        features_ref = compute_layer_features(ref_layer)
        features_tgt = compute_layer_features(tgt_layer)
        if features_ref is None:
            self._report_missing_data(ref_layer)
        if features_tgt is None:
            self._report_missing_data(tgt_layer)
        if features_ref is None or features_tgt is None:
            return

        matches, score_matrix = compute_auto_matches(features_ref, features_tgt, max_dist=max_dist, min_score=min_score, weights=weights)
        if not matches:
            print("⚠️ 未找到满足阈值的匹配对，请尝试放宽参数")
            return
        filtered_matches = []
        skipped_manual = 0
        for match in matches:
            orig_a = int(features_ref['orig_ids'][match['idx_a']])
            orig_b = int(features_tgt['orig_ids'][match['idx_b']])
            info_a = self._get_match_info(ref_layer, orig_a)
            info_b = self._get_match_info(tgt_layer, orig_b)
            if (info_a is not None and info_a.get('source') == 'manual') or (info_b is not None and info_b.get('source') == 'manual'):
                skipped_manual += 1
                continue
            filtered_matches.append(match)
        if skipped_manual:
            print(f"ℹ️ 有 {skipped_manual} 对候选匹配被手动配准结果保留而跳过")
        if not filtered_matches:
            print("⚠️ 自动配准结果全部被手动指定的匹配覆盖，未进行更新")
            return
        matches = filtered_matches
        color_seq = [self._next_match_color() for _ in range(len(matches))]
        apply_matches(ref_layer, tgt_layer, features_ref, features_tgt, matches, colors=color_seq, source='auto')
        self.refresh_all_match_colors()
        print(f"🤖 自动配准完成：共匹配 {len(matches)} 对 ROI（{ref_layer.name} ↔ {tgt_layer.name}）")
        for match in matches[:10]:
            orig_a = int(features_ref['orig_ids'][match['idx_a']])
            orig_b = int(features_tgt['orig_ids'][match['idx_b']])
            print(f"  · ROI {orig_a} ↔ ROI {orig_b} | score={match['score']:.3f}, spatial={match['spatial']:.3f}, temporal={match['temporal']:.3f}, dist={match['distance']:.2f}")


# -------------------------
control_panel = ROIControlPanel(viewer, shapes_layers)
viewer.window.add_dock_widget(control_panel, area='right')

# -------------------------
# on_click: show trace for selected segment (use orig id -> index into C_full_orig)
current_text_dialog, current_plot_fig = None, None

def on_click(layer, event):
    global current_text_dialog, current_plot_fig
    click_y, click_x = layer.world_to_data(event.position)
    click_xy = (click_x, click_y)

    roi_map = _get_roi_map_array(layer)  # contour -> original id
    meta = _get_layer_store(layer)
    C_full_orig = meta.get('C_full_orig', None)
    kept_mask = meta.get('kept_mask', None)
    orig_ids = meta.get('orig_ids', None)

    found_idx = None
    for shape_idx, verts in enumerate(layer.data):
        path = Path(np.column_stack((verts[:, 1], verts[:, 0])))
        if path.contains_point(click_xy):
            found_idx = shape_idx
            break
    if found_idx is None:
        return

    if found_idx >= len(roi_map):
        print(
            f"⚠️ {layer.name}: 找不到对应的 roi_map 索引 (contour={found_idx}, roi_map_len={len(roi_map)})，尝试自动修复。"
        )
        roi_map = _get_roi_map_array(layer)
        if found_idx >= len(roi_map):
            meta_keys = list(meta.keys()) if isinstance(meta, dict) else []
            print(
                f"❌ 仍然无法找到 roi_map 条目。contours={len(layer.data)}, roi_map_len={len(roi_map)}, metadata_keys={meta_keys}"
            )
            return

    # 当前导入文件内的 ROI 索引
    roi_idx_in_file = int(roi_map[found_idx])
    if roi_idx_in_file < 0:
        print("⚠️ 该 ROI 未关联原始索引，可能是手绘轮廓。")
        return

    # 获取原始 ID（导入时保存的 kept_orig_ids）
    if orig_ids is not None and roi_idx_in_file < len(orig_ids):
        orig_roi_id = int(orig_ids[roi_idx_in_file])
    else:
        orig_roi_id = roi_idx_in_file  # fallback

    # if marked deleted
    if kept_mask is not None and (orig_roi_id < 0 or orig_roi_id >= len(kept_mask) or not kept_mask[orig_roi_id]):
        print(f"⚠️ 原始 ROI {orig_roi_id} 已被删除或不可用。")
        return

    if C_full_orig is None:
        print("⚠️ 没有 C_full_orig，无 trace 可显示。")
        return

    # get trace by orig id (C_full_orig is indexed by orig id)
    try:
        trace = C_full_orig[roi_idx_in_file]
    except Exception:
        C_view = meta.get('C_view', None)
        if C_view is not None:
            kept_idxs = np.where(kept_mask)[0] if kept_mask is not None else None
            if kept_idxs is None:
                print("⚠️ 无法索引 trace")
                return
            try:
                idx = list(kept_idxs).index(orig_roi_id)
                trace = C_view[idx]
            except Exception:
                print("⚠️ 无法定位 trace")
                return
        else:
            print("⚠️ 无法读取 trace")
            return

    # use segment from control panel selection
    seg_idx = control_panel.combo_segments.currentIndex()
    if hasattr(control_panel, 'segment_list') and 0 <= seg_idx < len(control_panel.segment_list):
        s, e = control_panel.segment_list[seg_idx]
        start_frame = s if s is not None else 0
        end_frame = e if e is not None else len(trace)
    else:
        start_frame, end_frame = 0, len(trace)

    # sanitize segment
    start_frame = max(0, int(start_frame))
    end_frame = min(len(trace), int(end_frame)) if end_frame is not None else len(trace)
    if start_frame >= end_frame:
        start_frame, end_frame = 0, len(trace)
    segment = trace[start_frame:end_frame]

    # show text
    if control_panel.checkbox_data.isChecked():
        if current_text_dialog:
            current_text_dialog.close()
        dialog = QDialog()
        dialog.setWindowTitle(f"ROI {orig_roi_id} 数据 [{start_frame}:{end_frame}]")
        layout = QVBoxLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n".join(map(str, segment)))
        layout.addWidget(text)
        btn = QPushButton("关闭")
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.resize(400, 400)
        dialog.show()
        current_text_dialog = dialog

    # plot
    if control_panel.checkbox_plot.isChecked():
        if current_plot_fig:
            plt.close(current_plot_fig)
        fig, ax = plt.subplots()
        ax.plot(range(start_frame, end_frame), segment, color='b')
        ax.set_title(f"ROI {orig_roi_id} Trace [{start_frame}:{end_frame}]")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Fluorescence")
        fig.show()
        current_plot_fig = fig


# bind callbacks to existing shapes layers and keep dynamic binding for future ones
def bind_layer_callbacks(layer):
    if layer not in shapes_layers:
        shapes_layers.append(layer)
    if on_click not in [cb for cb in layer.mouse_drag_callbacks]:
        layer.mouse_drag_callbacks.append(on_click)
    cache_layer_base_edge_color(layer)
    try:
        control_panel.register_shapes_layer(layer)
    except Exception:
        pass


for layer in shapes_layers:
    bind_layer_callbacks(layer)


# when new layers are added manually via imports, user code sets shapes_layers; for safety, attach event on viewer.layers.events.inserted
def on_layer_insert(event):
    layer = event.value
    try:
        if isinstance(layer, napari.layers.Shapes):
            bind_layer_callbacks(layer)
    except Exception:
        pass


def on_layer_remove(event):
    layer = event.value
    try:
        if isinstance(layer, napari.layers.Shapes):
            if layer in shapes_layers:
                shapes_layers.remove(layer)
            control_panel.unregister_shapes_layer(layer)
    except Exception:
        pass


viewer.layers.events.inserted.connect(on_layer_insert)
viewer.layers.events.removed.connect(on_layer_remove)

# run
napari.run()
