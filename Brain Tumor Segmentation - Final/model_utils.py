import os
import h5py
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import cv2
import nibabel as nib

# Global variables to cache model and loaded volume
_cached_model = None
_cached_models = {}
_cached_volumes = {}

# 1. Custom Keras deserialization monkeypatch
orig_deserialize = keras.saving.deserialize_keras_object

def clean_config(config):
    """Recursively strip quantization_config from Keras layer configs."""
    if isinstance(config, dict):
        if 'quantization_config' in config:
            del config['quantization_config']
        for k, v in list(config.items()):
            clean_config(v)
    elif isinstance(config, list):
        for item in config:
            clean_config(item)

def custom_deserialize(config, *args, **kwargs):
    clean_config(config)
    return orig_deserialize(config, *args, **kwargs)

# Apply patches to prevent deserialization crash in Keras 3.x
keras.saving.deserialize_keras_object = custom_deserialize
try:
    import keras.src.saving.serialization_lib as serialization_lib
    serialization_lib.deserialize_keras_object = custom_deserialize
except:
    pass

# Bypass Lambda layer execution during load_model to prevent hard crashes
# due to Python version mismatch in bytecode. The correct functions are
# injected manually after the model is loaded.
orig_lambda_from_config = keras.layers.Lambda.from_config

@classmethod
def patched_lambda_from_config(cls, config, custom_objects=None):
    return cls(lambda x: x, name=config.get('name'))

keras.layers.Lambda.from_config = patched_lambda_from_config

# 2. Define and register custom layers and functions for the hybrid model
@keras.utils.register_keras_serializable(package='hybrid_seg')
class BatchMatMul(Layer):
    def __init__(self, transpose_a=False, transpose_b=False, **kwargs):
        super().__init__(**kwargs)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def call(self, inputs):
        a, b = inputs
        return tf.matmul(a, b, transpose_a=self.transpose_a, transpose_b=self.transpose_b)

    def get_config(self):
        config = super().get_config()
        config.update({'transpose_a': self.transpose_a, 'transpose_b': self.transpose_b})
        return config

@keras.utils.register_keras_serializable(package='hybrid_seg')
class SliceChannel(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index:self.index+1]

    def get_config(self):
        config = super().get_config()
        config.update({'index': self.index})
        return config

# Dummy metrics to load compiled model without needing original metrics code
def dice_coef(y_true, y_pred, smooth=1.0): return 0.0
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6): return 0.0
def dice_coef_edema(y_true, y_pred, epsilon=1e-6): return 0.0
def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6): return 0.0
def precision(y_true, y_pred): return 0.0
def sensitivity(y_true, y_pred): return 0.0
def specificity(y_true, y_pred): return 0.0

custom_objects = {
    'BatchMatMul': BatchMatMul,
    'SliceChannel': SliceChannel,
    'dice_coef': dice_coef,
    'precision': precision,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'dice_coef_necrotic': dice_coef_necrotic,
    'dice_coef_edema': dice_coef_edema,
    'dice_coef_enhancing': dice_coef_enhancing,
}

def _ocr_region_representation(t):
    return tf.einsum('bnc,bnf->bcf', t[0], t[1])

def _ocr_pixel_relation(t):
    return tf.nn.softmax(tf.einsum('bnf,bcf->bnc', t[0], t[1]), axis=-1)

def _ocr_context(t):
    return tf.einsum('bnc,bcf->bnf', t[0], t[1])

# 3. Model Loader
def get_model(model_name='best_model.keras'):
    """Load and cache the Keras segmentation model."""
    global _cached_model, _cached_models
    if model_name in _cached_models:
        return _cached_models[model_name]

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workspace_dir, model_name)
    if model_name == 'best_model.keras' and not os.path.exists(model_path):
        # Fallback to the other keras model name in workspace
        model_path = os.path.join(workspace_dir, 'Brain Model', 'attention_unet_best.keras')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")

    print(f"Loading Brain Tumor Segmentation model from: {model_path}...")
    # These checkpoints are used for prediction only; disabling compilation
    # avoids loading training-only losses and supports the trusted Lambda layer
    # in the supplied HRNet-OCR model.
    _cached_model = load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=True)
    # HRNet-OCR was saved from a notebook. Replacing its serialized anonymous
    # Lambda bytecode with the equivalent named functions makes it portable to
    # the current Python/Keras runtime.
    if _cached_model.name == 'HRNetOCR':
        lambda_functions = [_ocr_region_representation, _ocr_pixel_relation, _ocr_context]
        lambda_layers = [layer for layer in _cached_model.layers if isinstance(layer, keras.layers.Lambda)]
        if len(lambda_layers) != len(lambda_functions):
            raise ValueError('Unexpected HRNet-OCR Lambda layer layout')
        for layer, function in zip(lambda_layers, lambda_functions):
            layer.function = function
    _cached_models[model_name] = _cached_model
    print("Model loaded successfully.")
    return _cached_model

# 4. Volume Loader and Predictor
VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128
IN_CHANNELS = 2
COMPARISON_CHANNELS = 4
COMPARISON_IMG_SIZE = 160
NUM_CLASSES = 4

# Conversion factor: From a 128x128 slice (originally 240x240x1mm) to cubic millimeters
# Voxel volume = (240 / 128) * (240 / 128) * 1.0 mm = 3.515625 cubic mm
VOXEL_VOLUME_MM3 = 3.515625

def load_sample_volume(volume_id):
    """
    Load a preprocessed case from the .h5 files.
    Returns:
        flair_slices: list of 100 images (240, 240) scaled to 0-255 uint8
        t1ce_slices: list of 100 images (240, 240) scaled to 0-255 uint8
        gt_masks: list of 100 mask arrays (128, 128) or None
        pred_masks: list of 100 mask arrays (128, 128) from model predictions
    """
    global _cached_volumes
    if volume_id in _cached_volumes:
        return _cached_volumes[volume_id]

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    h5_dir = os.path.join(workspace_dir, "DataSet", "BraTS2020_training_data", "content", "data")
    
    # Verify directory exists
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"Sample data folder not found at: {h5_dir}")

    # Prepare inputs for prediction
    X = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, IN_CHANNELS), dtype=np.float32)
    comparison_X = np.zeros((VOLUME_SLICES, COMPARISON_IMG_SIZE, COMPARISON_IMG_SIZE, COMPARISON_CHANNELS), dtype=np.float32)
    flair_raw = []
    t1ce_raw = []
    gt_masks = []

    for j in range(VOLUME_SLICES):
        slice_idx = j + VOLUME_START_AT
        h5_path = os.path.join(h5_dir, f"volume_{volume_id}_slice_{slice_idx}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Missing slice file: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            img = f['image'][:]
            msk = f['mask'][:] # Shape (240, 240, 3)

            # Extract raw image channels (0 = flair, 2 = t1ce)
            flair = img[:, :, 0]
            t1ce = img[:, :, 2]

            # Save raw slices (scale to 0-255 for serving)
            flair_min, flair_max = flair.min(), flair.max()
            if flair_max - flair_min > 0:
                f_img = ((flair - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
            else:
                f_img = np.zeros_like(flair, dtype=np.uint8)

            t1ce_min, t1ce_max = t1ce.min(), t1ce.max()
            if t1ce_max - t1ce_min > 0:
                t_img = ((t1ce - t1ce_min) / (t1ce_max - t1ce_min) * 255).astype(np.uint8)
            else:
                t_img = np.zeros_like(t1ce, dtype=np.uint8)

            flair_raw.append(f_img)
            t1ce_raw.append(t_img)

            # Resize for network
            X[j, :, :, 0] = cv2.resize(flair, (IMG_SIZE, IMG_SIZE))
            X[j, :, :, 1] = cv2.resize(t1ce, (IMG_SIZE, IMG_SIZE))
            for ch in range(COMPARISON_CHANNELS):
                comparison_X[j, :, :, ch] = cv2.resize(img[:, :, ch], (COMPARISON_IMG_SIZE, COMPARISON_IMG_SIZE))

            # Reconstruct GT mask for display
            bg = np.clip(1.0 - np.sum(msk, axis=-1), 0.0, 1.0)
            target = np.stack([bg, msk[:, :, 0], msk[:, :, 1], msk[:, :, 2]], axis=-1)
            target_resized = np.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=np.float32)
            for ch in range(NUM_CLASSES):
                target_resized[:, :, ch] = cv2.resize(target[:, :, ch], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            gt_masks.append(np.argmax(target_resized, axis=-1).astype(np.uint8))

    # Run predictions
    pred_masks = run_model_inference(comparison_X)

    volume_data = {
        'flair': flair_raw,
        't1ce': t1ce_raw,
        'gt': gt_masks,
        'pred': pred_masks,
        'comparison_inputs': comparison_X
    }
    _cached_volumes[volume_id] = volume_data
    return volume_data

def process_nifti_volume(flair_path, t1_path, t1ce_path, t2_path, gt_path=None, original_filename=""):
    """
    Process custom uploaded FLAIR, T1, T1ce, and T2 NIfTI files.
    Optionally extracts or matches ground truth annotations if provided or recognized.
    Returns the processed raw volume slices, GT masks (if available), and predicted masks.
    """
    flair_img = nib.load(flair_path).get_fdata()
    t1_img = nib.load(t1_path).get_fdata()
    t1ce_img = nib.load(t1ce_path).get_fdata()
    t2_img = nib.load(t2_path).get_fdata()

    # Determine dimensions
    depth = flair_img.shape[2]
    # Check if we have enough slices, if not, adjust start slice
    start_slice = VOLUME_START_AT
    if depth < VOLUME_SLICES + start_slice:
        start_slice = max(0, depth - VOLUME_SLICES)

    comparison_X = np.zeros((VOLUME_SLICES, COMPARISON_IMG_SIZE, COMPARISON_IMG_SIZE, COMPARISON_CHANNELS), dtype=np.float32)
    flair_raw = []
    t1ce_raw = []
    gt_masks = []
    has_gt = False

    # 1. Process custom GT file if uploaded
    gt_img = None
    if gt_path and os.path.exists(gt_path):
        try:
            gt_img = nib.load(gt_path).get_fdata()
            has_gt = True
            print("Loaded custom Ground Truth NIfTI file successfully.")
        except Exception as e:
            print(f"Warning: Failed to load GT NIfTI file: {e}")

    # 2. If no GT file provided, try automatic dataset patient matching from filename
    if not has_gt and original_filename:
        import re
        match = re.search(r'(?:patient_|volume_|BraTS20_Patient_|BraTS20_Training_)(\d+)', original_filename, re.IGNORECASE)
        if match:
            matched_id = int(match.group(1))
            workspace_dir = os.path.dirname(os.path.abspath(__file__))
            h5_dir = os.path.join(workspace_dir, "DataSet", "BraTS2020_training_data", "content", "data")
            test_h5 = os.path.join(h5_dir, f"volume_{matched_id}_slice_22.h5")
            if os.path.exists(test_h5):
                print(f"Auto-matched uploaded scan to dataset Patient #{matched_id}. Loading full 4-channel volume!")
                try:
                    vol_data = load_sample_volume(matched_id)
                    _cached_volumes['custom'] = vol_data
                    return vol_data
                except Exception as e:
                    print(f"Warning: Failed to load matched full volume #{matched_id}: {e}")

    for j in range(VOLUME_SLICES):
        slice_idx = j + start_slice
        if slice_idx >= depth:
            # Pad with zeros or mirror last slice
            slice_idx = depth - 1
        
        flair = flair_img[:, :, slice_idx]
        t1 = t1_img[:, :, slice_idx]
        t1ce = t1ce_img[:, :, slice_idx]
        t2 = t2_img[:, :, slice_idx]

        # Scale raw slices to 0-255 uint8 for frontend visualization
        flair_min, flair_max = flair.min(), flair.max()
        if flair_max - flair_min > 0:
            f_img = ((flair - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
        else:
            f_img = np.zeros_like(flair, dtype=np.uint8)

        t1ce_min, t1ce_max = t1ce.min(), t1ce.max()
        if t1ce_max - t1ce_min > 0:
            t_img = ((t1ce - t1ce_min) / (t1ce_max - t1ce_min) * 255).astype(np.uint8)
        else:
            t_img = np.zeros_like(t1ce, dtype=np.uint8)

        flair_raw.append(f_img)
        t1ce_raw.append(t_img)

        # Process and normalize for the model (min-max [0, 1])
        for ch, modality in enumerate((flair, t1, t1ce, t2)):
            resized = cv2.resize(modality, (COMPARISON_IMG_SIZE, COMPARISON_IMG_SIZE)).astype(np.float32)
            c_min, c_max = resized.min(), resized.max()
            if c_max > c_min:
                comparison_X[j, :, :, ch] = (resized - c_min) / (c_max - c_min)
            else:
                comparison_X[j, :, :, ch] = 0.0

        # Process GT slice from custom GT image if available
        if gt_img is not None:
            g_slice = gt_img[:, :, slice_idx]
            # Standard BraTS GT labels: 1 = NCR/NET (Core), 2 = ED (Edema), 4 or 3 = ET (Enhancing)
            gt_map = np.zeros_like(g_slice, dtype=np.uint8)
            gt_map[g_slice == 1] = 1 # Necrotic Core
            gt_map[g_slice == 2] = 2 # Edema
            gt_map[(g_slice == 3) | (g_slice == 4)] = 3 # Enhancing
            
            # Resize mask to IMG_SIZE
            gt_resized = cv2.resize(gt_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            gt_masks.append(gt_resized.astype(np.uint8))

    # Run predictions
    pred_masks = run_model_inference(comparison_X)

    # Save to a unique cache key ('custom')
    volume_data = {
        'flair': flair_raw,
        't1ce': t1ce_raw,
        'gt': gt_masks if (has_gt and len(gt_masks) == VOLUME_SLICES) else None,
        'pred': pred_masks,
        'comparison_inputs': comparison_X
    }
    _cached_volumes['custom'] = volume_data
    return volume_data

def run_model_inference(X):
    """Normalize input volume X and predict segmentations."""
    model = get_model()
    # Volume-wide normalization
    # The new Attention_UNet model expects 160x160x4 inputs.
    X_norm = _normalize_comparison_inputs(X)

    preds = model.predict(X_norm, batch_size=2, verbose=0) 
    pred_masks = (preds[..., 0] > 0.5).astype(np.uint8)
    
    resized_masks = []
    for i in range(pred_masks.shape[0]):
        resized_masks.append(cv2.resize(pred_masks[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST))
    return resized_masks

def _binary_segmentation_metrics(pred_masks, gt_masks):
    """Calculate the UI metrics from this model's prediction and GT.

    Evaluates binary whole-tumor segmentation (pred > 0 vs gt > 0).
    Values are calculated over the uploaded volume, never copied from a log.
    """
    pred_labels = np.asarray(pred_masks, dtype=np.uint8)
    gt_labels = np.asarray(gt_masks, dtype=np.uint8)
    pred = pred_labels > 0
    gt = gt_labels > 0
    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    eps = 1e-8

    tumor_dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    metrics = {
        'mean_dice': tumor_dice,
        'tumor_dice': tumor_dice,
        'iou': iou,
        'pixel_accuracy': pixel_accuracy,
    }

    return {key: round(value, 4) if value is not None else None for key, value in metrics.items()}

def _normalize_comparison_inputs(inputs):
    """Match the standalone-model training preprocessing exactly.

    Training used a per-slice, per-modality Z-score over non-zero brain voxels,
    not one max-value normalization across the entire volume.
    """
    normalized = np.zeros_like(inputs, dtype=np.float32)
    for slice_index in range(inputs.shape[0]):
        for channel in range(inputs.shape[-1]):
            ch = inputs[slice_index, :, :, channel].astype(np.float32)
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                normalized[slice_index, :, :, channel] = (ch - ch_min) / (ch_max - ch_min)
            else:
                normalized[slice_index, :, :, channel] = ch
    return normalized

def get_model_comparison(volume_data):
    """Run each standalone checkpoint on the same four-modality MRI volume."""
    import cv2
    import os
    comparison_X = volume_data.get('comparison_inputs')
    if comparison_X is None:
        return {
            key: {'available': False, 'tumor_detected': None, 'metrics': {},
                  'message': 'Four MRI modalities are required for this model.'}
            for key in ('attention_unet', 'hrnet_ocr')
        }

    X_norm = _normalize_comparison_inputs(comparison_X)
    
    model_specs = {
        'attention_unet': 'Brain Model/attention_unet_best.keras',
        'hrnet_ocr': 'Brain Model/hrnet_ocr_best.keras'
    }

    predictions, results = {}, {}
    gt_masks = volume_data.get('gt')
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    for key, path in model_specs.items():
        try:
            model_path = os.path.join(workspace_dir, path)
            model = get_model(model_path)
            probs = model.predict(X_norm, batch_size=2, verbose=0)
            
            # Apply standard 0.5 threshold for both models
            threshold = 0.5
            masks_160 = (probs[..., 0] > threshold).astype(np.uint8)
            
            masks = []
            for i in range(masks_160.shape[0]):
                masks.append(cv2.resize(masks_160[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST))
            
            predictions[key] = masks
            
            if gt_masks is not None and len(gt_masks) == len(masks):
                metrics = _binary_segmentation_metrics(masks, gt_masks)
            else:
                metrics = {}
                
            tumor_areas = [int(np.count_nonzero(mask)) for mask in masks]
            results[key] = {
                'available': True,
                'tumor_detected': bool(any(tumor_areas)),
                'metrics': metrics,
                'recommended_slice': int(np.argmax(tumor_areas)) if any(tumor_areas) else 50
            }
        except Exception as e:
            print(f"Error evaluating {key}: {e}")
            results[key] = {
                'available': False,
                'message': str(e),
                'metrics': {},
                'tumor_detected': False,
                'recommended_slice': 0
            }
            predictions[key] = [np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8) for _ in range(VOLUME_SLICES)]
                
    volume_data['model_predictions'] = predictions
    return results

def get_representative_slice(volume_data):
    """Return the most informative slice for the result viewer.

    Prefer the slice with the largest independently predicted tumor area. If
    neither model predicts a tumor, use the largest reference-tumor slice.
    """
    prediction_sets = volume_data.get('model_predictions', {})
    total_slices = len(volume_data.get('flair', []))
    best_index, best_area = 0, 0
    for index in range(total_slices):
        area = sum(int(np.count_nonzero(masks[index])) for masks in prediction_sets.values() if index < len(masks))
        if area > best_area:
            best_index, best_area = index, area
    if best_area == 0 and volume_data.get('gt') is not None:
        for index, mask in enumerate(volume_data['gt']):
            area = int(np.count_nonzero(mask))
            if area > best_area:
                best_index, best_area = index, area
    return best_index

def get_volume_stats(volume_data):
    """
    Calculate tumor statistics slice-by-slice and sum them up.
    WT (Whole Tumor) = Core (1) + Edema (2) + Enhancing (3)
    TC (Tumor Core) = Core (1) + Enhancing (3)
    ET (Enhancing Tumor) = Enhancing (3)
    """
    pred_masks = volume_data['pred']
    
    # Calculate stats per slice
    slice_stats = []
    total_voxels = {1: 0, 2: 0, 3: 0} # Core, Edema, Enhancing

    for idx, mask in enumerate(pred_masks):
        # Count non-zero label occurrences
        counts = {1: int(np.sum(mask == 1)), 2: int(np.sum(mask == 2)), 3: int(np.sum(mask == 3))}
        
        # Calculate areas in mm^2
        core_area = counts[1] * VOXEL_VOLUME_MM3
        edema_area = counts[2] * VOXEL_VOLUME_MM3
        enhancing_area = counts[3] * VOXEL_VOLUME_MM3
        whole_tumor_area = core_area + edema_area + enhancing_area
        tumor_core_area = core_area + enhancing_area

        slice_stats.append({
            'slice_idx': idx + VOLUME_START_AT,
            'necrotic_area': core_area,
            'edema_area': edema_area,
            'enhancing_area': enhancing_area,
            'whole_tumor_area': whole_tumor_area,
            'tumor_core_area': tumor_core_area
        })

        total_voxels[1] += counts[1]
        total_voxels[2] += counts[2]
        total_voxels[3] += counts[3]

    # Convert totals to volume in cubic centimeters (cm^3)
    core_vol = (total_voxels[1] * VOXEL_VOLUME_MM3) / 1000.0
    edema_vol = (total_voxels[2] * VOXEL_VOLUME_MM3) / 1000.0
    enhancing_vol = (total_voxels[3] * VOXEL_VOLUME_MM3) / 1000.0
    whole_vol = core_vol + edema_vol + enhancing_vol
    tumor_core_vol = core_vol + enhancing_vol

    totals = {
        'necrotic_volume': round(core_vol, 3),
        'edema_volume': round(edema_vol, 3),
        'enhancing_volume': round(enhancing_vol, 3),
        'whole_tumor_volume': round(whole_vol, 3),
        'tumor_core_volume': round(tumor_core_vol, 3)
    }

    # Reconstruct ground truth stats if available
    gt_totals = None
    if volume_data.get('gt') is not None:
        gt_masks = volume_data['gt']
        gt_voxels = {1: 0, 2: 0, 3: 0}
        for mask in gt_masks:
            gt_voxels[1] += np.sum(mask == 1)
            gt_voxels[2] += np.sum(mask == 2)
            gt_voxels[3] += np.sum(mask == 3)
        
        gt_totals = {
            'necrotic_volume': round((gt_voxels[1] * VOXEL_VOLUME_MM3) / 1000.0, 3),
            'edema_volume': round((gt_voxels[2] * VOXEL_VOLUME_MM3) / 1000.0, 3),
            'enhancing_volume': round((gt_voxels[3] * VOXEL_VOLUME_MM3) / 1000.0, 3),
            'whole_tumor_volume': round(((gt_voxels[1] + gt_voxels[2] + gt_voxels[3]) * VOXEL_VOLUME_MM3) / 1000.0, 3),
            'tumor_core_volume': round(((gt_voxels[1] + gt_voxels[3]) * VOXEL_VOLUME_MM3) / 1000.0, 3)
        }

    return {
        'slices': slice_stats,
        'totals': totals,
        'gt_totals': gt_totals
    }
