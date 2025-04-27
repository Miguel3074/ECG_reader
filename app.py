import os
import logging
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import ecg_analyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_for_ecg_analyzer")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if a file was uploaded
    if 'ecg_image' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['ecg_image']
    
    # Check if user submitted an empty form
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    
    # Check file type
    if not allowed_file(file.filename):
        flash('Only PNG files are allowed', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Save file with a secure and unique filename
        unique_filename = str(uuid.uuid4()) + '.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the ECG image
        results = ecg_analyzer.analyze_ecg(filepath)
        
        # Remove the temporary file after processing
        os.remove(filepath)
        
        # Store results in session for display
        session['analysis_results'] = results
        
        return redirect(url_for('results'))
    
    except Exception as e:
        logging.error(f"Error processing ECG: {str(e)}")
        flash(f'Error processing ECG: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    # Get results from session
    results = session.get('analysis_results')
    if not results:
        flash('No analysis results found. Please upload an ECG image.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=results)

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error. Please try again later.', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
