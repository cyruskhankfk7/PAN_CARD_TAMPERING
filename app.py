# app.py
import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity
import imutils
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}
RESULT_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_tampering(original_path, tampered_path):
    """
    Detect tampering between original and tampered images
    Returns: dict with results and images
    """
    try:
        # Read images
        original = cv2.imread(original_path)
        tampered = cv2.imread(tampered_path)

        if original is None or tampered is None:
            return {'error': 'Could not read one or both images'}

        # Resize images to same dimensions
        original = cv2.resize(original, (250, 160))
        tampered = cv2.resize(tampered, (250, 160))

        # Convert to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold and find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Create copies for marking
        original_marked = original.copy()
        tampered_marked = tampered.copy()

        # Draw rectangles around tampered areas
        tamper_count = 0
        for c in cnts:
            if cv2.contourArea(c) > 40:  # Filter small noise
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(original_marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(tampered_marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
                tamper_count += 1

        # Determine tampering status
        threshold = 0.9  # SSIM threshold for tampering detection
        is_tampered = score < threshold

        # Convert images to base64 for displaying in HTML
        def image_to_base64(img):
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        return {
            'ssim_score': score,
            'is_tampered': is_tampered,
            'tamper_count': tamper_count,
            'original_marked': image_to_base64(original_marked),
            'tampered_marked': image_to_base64(tampered_marked),
            'diff_image': image_to_base64(diff),
            'thresh_image': image_to_base64(thresh),
            'confidence': abs(score - 0.5) * 200  # Simple confidence calculation
        }

    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    if 'original' not in request.files or 'tampered' not in request.files:
        flash('Please upload both images')
        return redirect(url_for('index'))

    original_file = request.files['original']
    tampered_file = request.files['tampered']

    if original_file.filename == '' or tampered_file.filename == '':
        flash('Please select both images')
        return redirect(url_for('index'))

    if original_file and allowed_file(original_file.filename) and \
            tampered_file and allowed_file(tampered_file.filename):

        # Save uploaded files
        original_filename = secure_filename(original_file.filename)
        tampered_filename = secure_filename(tampered_file.filename)

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + original_filename)
        tampered_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tampered_' + tampered_filename)

        original_file.save(original_path)
        tampered_file.save(tampered_path)

        # Detect tampering
        result = detect_tampering(original_path, tampered_path)

        if 'error' in result:
            flash(f'Error: {result["error"]}')
            return redirect(url_for('index'))

        return render_template('result.html', result=result)

    flash('Invalid file type. Please upload PNG, JPG, or JPEG files')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)