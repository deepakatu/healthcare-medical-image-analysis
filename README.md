
# Healthcare Medical Image Analysis

A production-ready web application for medical image analysis using deep learning. Features automated diagnosis of chest X-rays with confidence scoring and detailed reporting.

## Features

- **Deep Learning Model**: Custom CNN architecture trained on chest X-ray datasets
- **Web Interface**: Flask-based web application with file upload
- **Image Processing**: Advanced preprocessing with OpenCV and PIL
- **Confidence Scoring**: Probabilistic outputs with uncertainty quantification
- **Report Generation**: Automated medical reports with visualizations
- **API Endpoints**: RESTful API for integration with hospital systems

## Tech Stack

- **Backend**: Flask, TensorFlow/Keras, OpenCV
- **Frontend**: HTML5, Bootstrap, JavaScript
- **ML**: CNN, Transfer Learning (ResNet50)
- **Database**: SQLite for logging and results
- **Deployment**: Docker, Gunicorn

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd healthcare-medical-image-analysis
pip install -r requirements.txt

# Run application
python app.py

# Access at http://localhost:5000
```

## API Usage

```python
import requests

# Upload image for analysis
files = {'image': open('chest_xray.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/analyze', files=files)
result = response.json()
```

## Model Performance

- **Accuracy**: 94.2% on validation set
- **Precision**: 93.8% (Pneumonia detection)
- **Recall**: 95.1% (Pneumonia detection)
- **F1-Score**: 94.4%

## Project Structure

```
├── src/
│   ├── models/          # ML models and training
│   ├── preprocessing/   # Image processing utilities
│   ├── api/            # Flask API endpoints
│   └── utils/          # Helper functions
├── templates/          # HTML templates
├── static/            # CSS, JS, images
├── tests/             # Unit tests
├── data/              # Sample data
└── models/            # Trained model files
```

## License

MIT License - See LICENSE file for details.
