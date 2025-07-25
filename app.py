
"""
Healthcare Medical Image Analysis - Main Flask Application
Production-ready web application for chest X-ray analysis
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import sqlite3
import json

from src.models.cnn_model import ChestXRayModel
from src.preprocessing.image_processor import ImageProcessor
from src.utils.report_generator import ReportGenerator
from src.api.endpoints import api_bp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Initialize components
model = ChestXRayModel()
processor = ImageProcessor()
report_gen = ReportGenerator()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
def init_db():
    """Initialize SQLite database for logging results"""
    conn = sqlite3.connect('medical_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            processing_time REAL NOT NULL,
            report_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or DICOM files'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process and analyze image
        start_time = datetime.now()
        
        # Preprocess image
        processed_image = processor.preprocess_image(filepath)
        
        # Make prediction
        prediction, confidence = model.predict(processed_image)
        
        # Generate detailed report
        report_data = report_gen.generate_report(
            image_path=filepath,
            prediction=prediction,
            confidence=confidence,
            processed_image=processed_image
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log to database
        log_analysis(filename, prediction, confidence, processing_time, report_data)
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'prediction': prediction,
            'confidence': float(confidence),
            'processing_time': processing_time,
            'report': report_data,
            'image_url': f'/uploads/{filename}'
        }
        
        logger.info(f"Analysis completed for {filename}: {prediction} ({confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def analysis_history():
    """Display analysis history"""
    try:
        conn = sqlite3.connect('medical_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename, timestamp, prediction, confidence, processing_time
            FROM analyses
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        history = cursor.fetchall()
        conn.close()
        
        return render_template('history.html', history=history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return render_template('history.html', history=[], error=str(e))

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    try:
        conn = sqlite3.connect('medical_analysis.db')
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('SELECT COUNT(*) FROM analyses')
        total_analyses = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(processing_time) FROM analyses')
        avg_processing_time = cursor.fetchone()[0] or 0
        
        cursor.execute('''
            SELECT prediction, COUNT(*) as count
            FROM analyses
            GROUP BY prediction
        ''')
        prediction_stats = cursor.fetchall()
        
        conn.close()
        
        stats = {
            'total_analyses': total_analyses,
            'avg_processing_time': round(avg_processing_time, 3),
            'prediction_distribution': dict(prediction_stats)
        }
        
        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return render_template('dashboard.html', stats={}, error=str(e))

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_analysis(filename, prediction, confidence, processing_time, report_data):
    """Log analysis results to database"""
    try:
        conn = sqlite3.connect('medical_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analyses (filename, prediction, confidence, processing_time, report_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, prediction, confidence, processing_time, json.dumps(report_data)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging analysis: {str(e)}")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    init_db()
    
    # Load model on startup
    try:
        model.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
    
    # Run application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
