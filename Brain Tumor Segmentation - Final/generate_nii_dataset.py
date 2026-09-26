import os
import glob
import h5py
import numpy as np
import nibabel as nib
import time

def generate_nii_files(num_samples=10):
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    h5_dir = os.path.join(workspace_dir, "DataSet", "BraTS2020_training_data", "content", "data")
    
    flair_dir = os.path.join(workspace_dir, "sample_nifti_data", "flair_scans")
    t1_dir = os.path.join(workspace_dir, "sample_nifti_data", "t1_scans")
    t1ce_dir = os.path.join(workspace_dir, "sample_nifti_data", "t1ce_scans")
    t2_dir = os.path.join(workspace_dir, "sample_nifti_data", "t2_scans")
    
    os.makedirs(flair_dir, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t1ce_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    
    print(f"Target directory for FLAIR scans: {flair_dir}")
    print(f"Target directory for T1 scans: {t1_dir}")
    print(f"Target directory for T1ce scans: {t1ce_dir}")
    print(f"Target directory for T2 scans: {t2_dir}")
    
    # Get available volume IDs
    slice_files = glob.glob(os.path.join(h5_dir, "volume_*_slice_*.h5"))
    vol_ids = sorted(list(set([int(os.path.basename(f).split('volume_')[1].split('_slice')[0]) for f in slice_files])))
    
    selected_vols = vol_ids[:num_samples]
    print(f"Found {len(vol_ids)} total volumes. Processing first {len(selected_vols)} volumes...")
    
    start_time = time.time()
    
    for idx, vol_id in enumerate(selected_vols, 1):
        # Find all slice files for this volume, ordered by slice index
        vol_slice_files = glob.glob(os.path.join(h5_dir, f"volume_{vol_id}_slice_*.h5"))
        vol_slice_files.sort(key=lambda x: int(os.path.basename(x).split('_slice_')[1].split('.h5')[0]))
        
        flair_slices = []
        t1_slices = []
        t1ce_slices = []
        t2_slices = []
        
        for sf in vol_slice_files:
            try:
                with h5py.File(sf, 'r') as f:
                    img = f['image'][:] # shape (240, 240, 4)
                    flair_slices.append(img[:, :, 0])
                    t1_slices.append(img[:, :, 1])
                    t1ce_slices.append(img[:, :, 2])
                    t2_slices.append(img[:, :, 3])
            except Exception as e:
                print(f"Warning: Error reading {sf}: {e}")
                continue
        
        if not flair_slices:
            continue
            
        # Stack slices into 3D volume (240, 240, Z)
        flair_3d = np.stack(flair_slices, axis=-1).astype(np.float32)
        t1_3d = np.stack(t1_slices, axis=-1).astype(np.float32)
        t1ce_3d = np.stack(t1ce_slices, axis=-1).astype(np.float32)
        t2_3d = np.stack(t2_slices, axis=-1).astype(np.float32)
        
        # Define affine (standard 1mm x 1mm x 1mm voxel spacing)
        affine = np.eye(4)
        
        flair_nii = nib.Nifti1Image(flair_3d, affine)
        t1_nii = nib.Nifti1Image(t1_3d, affine)
        t1ce_nii = nib.Nifti1Image(t1ce_3d, affine)
        t2_nii = nib.Nifti1Image(t2_3d, affine)
        
        flair_path = os.path.join(flair_dir, f"patient_{vol_id:03d}_flair.nii.gz")
        t1_path = os.path.join(t1_dir, f"patient_{vol_id:03d}_t1.nii.gz")
        t1ce_path = os.path.join(t1ce_dir, f"patient_{vol_id:03d}_t1ce.nii.gz")
        t2_path = os.path.join(t2_dir, f"patient_{vol_id:03d}_t2.nii.gz")
        
        nib.save(flair_nii, flair_path)
        nib.save(t1_nii, t1_path)
        nib.save(t1ce_nii, t1ce_path)
        nib.save(t2_nii, t2_path)
        
        if idx % 10 == 0 or idx == len(selected_vols):
            elapsed = time.time() - start_time
            print(f"[{idx}/{len(selected_vols)}] Saved patient_{vol_id:03d} scans. ({elapsed:.1f}s elapsed)")

    print(f"\nSUCCESS: Generated {len(selected_vols)} matching pairs of .nii.gz files!")
    print(f"FLAIR scans path: {flair_dir}")
    print(f"T1ce scans path: {t1ce_dir}")

if __name__ == '__main__':
    generate_nii_files(100)
