import os
import sys
import argparse
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session, flash
import model_utils
import db

# Create upload directory inside workspace
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('SECRET_KEY', 'neuroseg_ai_secret_key_2026_brain_tumor_segmentation_key')

# Initialize database tables
db.init_db()

# Cache active patient volume data in memory
active_volume = {
    'id': None,
    'data': None
}

@app.route('/')
def index():
    """Render the dashboard UI if logged in, else redirect to landing page."""
    if 'user_id' not in session:
        return redirect(url_for('landing'))
    return render_template('index.html', user_name=session.get('user_name', 'Doctor / Researcher'))

@app.route('/landing')
def landing():
    """Render the landing page with Sign Up, Login, and About sections."""
    if 'user_id' in session:
        return redirect(url_for('index'))
    section = request.args.get('section', 'home')
    return render_template('landing.html', active_section=section)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration."""
    if request.method == 'GET':
        return redirect(url_for('landing', section='signup'))
    
    # Handle POST submission (supports both JSON AJAX and standard Form)
    is_json = request.is_json
    data = request.get_json() if is_json else request.form
    
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')
    
    if not full_name or not email or not password:
        msg = "All fields (Full Name, Email, Password) are required."
        if is_json:
            return jsonify({'success': False, 'error': msg}), 400
        flash(msg, 'danger')
        return render_template('landing.html', active_section='signup', error=msg)
        
    if password != confirm_password:
        msg = "Passwords do not match. Please re-type your password."
        if is_json:
            return jsonify({'success': False, 'error': msg}), 400
        flash(msg, 'danger')
        return render_template('landing.html', active_section='signup', error=msg)

    success, result = db.create_user(full_name, email, password)
    if success:
        # Auto-login after successful registration
        session['user_id'] = result['id']
        session['user_name'] = result['full_name']
        session['user_email'] = result['email']
        
        if is_json:
            return jsonify({'success': True, 'message': 'Account created successfully!', 'redirect': url_for('index')})
        flash(f"Welcome to NeuroSeg AI, {result['full_name']}!", 'success')
        return redirect(url_for('index'))
    else:
        if is_json:
            return jsonify({'success': False, 'error': result}), 400
        flash(result, 'danger')
        return render_template('landing.html', active_section='signup', error=result)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'GET':
        return redirect(url_for('landing', section='login'))
        
    is_json = request.is_json
    data = request.get_json() if is_json else request.form
    
    email = data.get('email', '').strip()
    password = data.get('password', '')
    
    if not email or not password:
        msg = "Please enter both email and password."
        if is_json:
            return jsonify({'success': False, 'error': msg}), 400
        flash(msg, 'danger')
        return render_template('landing.html', active_section='login', error=msg)

    user, error = db.verify_user(email, password)
    if user:
        session['user_id'] = user['id']
        session['user_name'] = user['full_name']
        session['user_email'] = user['email']
        
        if is_json:
            return jsonify({'success': True, 'message': 'Logged in successfully!', 'redirect': url_for('index')})
        flash(f"Welcome back, {user['full_name']}!", 'success')
        return redirect(url_for('index'))
    else:
        if is_json:
            return jsonify({'success': False, 'error': error}), 400
        flash(error, 'danger')
        return render_template('landing.html', active_section='login', error=error)

@app.route('/logout')
def logout():
    """Clear session and logout user."""
    session.clear()
    flash('You have been logged out safely.', 'info')
    return redirect(url_for('landing'))

@app.route('/about')
def about():
    """Direct route to About section on landing page."""
    return redirect(url_for('landing', section='about'))


@app.route('/api/load_sample/<int:volume_id>', methods=['GET'])
def load_sample(volume_id):
    """Load a sample case and cache it."""
    global active_volume
    try:
        data = model_utils.load_sample_volume(volume_id)
        active_volume['id'] = str(volume_id)
        active_volume['data'] = data
        
        stats = model_utils.get_volume_stats(data)
        
        model_results = model_utils.get_model_comparison(data)
        return jsonify({
            'success': True,
            'volume_id': volume_id,
            'slices_count': model_utils.VOLUME_SLICES,
            'start_slice': model_utils.VOLUME_START_AT,
            'stats': stats,
            'model_results': model_results,
            'recommended_slice': model_utils.get_representative_slice(data)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle raw NIfTI scans uploading and run model predictions."""
    global active_volume
    required_modalities = ('flair', 't1', 't1ce', 't2')
    if any(name not in request.files for name in required_modalities):
        return jsonify({'success': False, 'error': 'FLAIR, T1, T1ce, and T2 files are required'}), 400

    flair_file = request.files['flair']
    t1_file = request.files['t1']
    t1ce_file = request.files['t1ce']
    t2_file = request.files['t2']
    gt_file = request.files.get('gt')

    if any(not file.filename for file in (flair_file, t1_file, t1ce_file, t2_file)):
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    try:
        # Save temporary files
        flair_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_flair.nii')
        t1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_t1.nii')
        t1ce_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_t1ce.nii')
        t2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_t2.nii')
        gt_path = None

        # Handle NIfTI compressed files (.nii.gz)
        if flair_file.filename.endswith('.gz'): flair_path += '.gz'
        if t1_file.filename.endswith('.gz'): t1_path += '.gz'
        if t1ce_file.filename.endswith('.gz'): t1ce_path += '.gz'
        if t2_file.filename.endswith('.gz'): t2_path += '.gz'

        flair_file.save(flair_path)
        t1_file.save(t1_path)
        t1ce_file.save(t1ce_path)
        t2_file.save(t2_path)

        if gt_file and gt_file.filename != '':
            gt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_gt.nii')
            if gt_file.filename.endswith('.gz'):
                gt_path += '.gz'
            gt_file.save(gt_path)
            print(f"Uploaded Ground Truth file: {gt_path}")

        print(f"Uploaded custom MRI. FLAIR: {flair_path}, T1: {t1_path}, T1ce: {t1ce_path}, T2: {t2_path}")
        
        # Process and predict
        data = model_utils.process_nifti_volume(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            gt_path=gt_path, 
            original_filename=flair_file.filename
        )
        
        # Clean up files
        for p in [flair_path, t1_path, t1ce_path, t2_path, gt_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

        active_volume['id'] = 'custom'
        active_volume['data'] = data

        stats = model_utils.get_volume_stats(data)

        model_results = model_utils.get_model_comparison(data)
        return jsonify({
            'success': True,
            'volume_id': 'custom',
            'slices_count': model_utils.VOLUME_SLICES,
            'start_slice': model_utils.VOLUME_START_AT,
            'has_gt': data['gt'] is not None,
            'stats': stats,
            'model_results': model_results,
            'recommended_slice': model_utils.get_representative_slice(data)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/slice', methods=['GET'])
def get_slice():
    """
    Serve a slice or mask overlay as a PNG image dynamically.
    Query parameters:
        volume_id: the current active volume id
        slice_idx: axial slice index (0 to 99)
        type: flair, t1ce, gt, or pred
    """
    volume_id = request.args.get('volume_id')
    slice_idx = request.args.get('slice_idx', type=int)
    slice_type = request.args.get('type')

    if active_volume['data'] is None or active_volume['id'] != volume_id:
        return 'Volume not loaded', 404

    if slice_idx is None or slice_idx < 0 or slice_idx >= model_utils.VOLUME_SLICES:
        return 'Invalid slice index', 400

    vol_data = active_volume['data']

    if slice_type == 'flair':
        img = vol_data['flair'][slice_idx]
        _, buffer = cv2.imencode('.png', img)
        return send_file(io.BytesIO(buffer), mimetype='image/png')

    elif slice_type == 't1ce':
        img = vol_data['t1ce'][slice_idx]
        _, buffer = cv2.imencode('.png', img)
        return send_file(io.BytesIO(buffer), mimetype='image/png')

    elif slice_type in ['gt', 'pred']:
        # A model query is required for the new comparison panels. This keeps a
        # legacy/fused prediction from ever being served as Attention U-Net or
        # HRNet-OCR output. Once dedicated inference is wired in, store masks as
        # volume_data['model_predictions'][model_key].
        model_key = request.args.get('model')
        if slice_type == 'pred' and model_key:
            model_predictions = vol_data.get('model_predictions', {})
            mask_list = model_predictions.get(model_key)
        else:
            mask_list = vol_data.get(slice_type)
        if mask_list is None:
            # Create a fully transparent dummy mask if GT is requested but unavailable
            dummy_rgba = np.zeros((240, 240, 4), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', dummy_rgba)
            return send_file(io.BytesIO(buffer), mimetype='image/png')

        mask_128 = mask_list[slice_idx] # (128, 128)
        # Upscale to (240, 240) using nearest-neighbor to align with raw scan dimensions
        mask_240 = cv2.resize(mask_128, (240, 240), interpolation=cv2.INTER_NEAREST)

        # Create RGBA colored image
        rgba = np.zeros((240, 240, 4), dtype=np.uint8)

        # Define modern color-codes (RGBA) for each sub-region
        # 1: Necrotic Core (Red) -> (R, G, B, A)
        # 2: Peritumoral Edema (Yellow)
        # 3: Enhancing Tumor (Blue)
        rgba[mask_240 == 1] = [68, 68, 239, 150]    # OpenCV uses BGRA format! Red (239, 68, 68) -> (68, 68, 239)
        rgba[mask_240 == 2] = [8, 179, 234, 120]    # Yellow (234, 179, 8) -> (8, 179, 234)
        rgba[mask_240 == 3] = [246, 130, 59, 170]   # Blue (59, 130, 246) -> (246, 130, 59)

        _, buffer = cv2.imencode('.png', rgba)
        return send_file(io.BytesIO(buffer), mimetype='image/png')

    else:
        return 'Invalid slice type', 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation Flask Frontend")
    parser.add_argument('--test-load', action='store_true', help="Perform model loading check and exit")
    parser.add_argument('--port', type=int, default=5001, help="Server port (default 5001)")
    args = parser.parse_args()

    if args.test_load:
        print("Starting model loading test check...")
        try:
            model_utils.get_model()
            print("Model loaded successfully. Verification complete!")
            sys.exit(0)
        except Exception as e:
            print("Model loading test verification FAILED:", e)
            sys.exit(1)

    print("Pre-warming hybrid model...")
    try:
        model_utils.get_model()
    except Exception as e:
        print("Warning: Model failed to pre-warm. It will be loaded on demand:", e)

    app.run(host='127.0.0.1', port=args.port, debug=False)
