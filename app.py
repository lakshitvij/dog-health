from flask import Flask, render_template, request, jsonify
import os
import json
from utils.symptom_predictor import SymptomPredictor
from utils.image_predictor import ImagePredictor
from utils.disease_info import get_disease_info
from utils.supabase_client import (
    save_prediction, get_user_predictions, add_favorite,
    remove_favorite, get_user_favorites, save_user_profile, get_user_profile
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

symptom_predictor = None
image_predictor = None

def initialize_models():
    """Initialize ML models"""
    global symptom_predictor, image_predictor

    print("Initializing models...")
    symptom_predictor = SymptomPredictor()
    image_predictor = ImagePredictor()
    print("Models initialized successfully!")

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/predict-symptom', methods=['POST'])
def predict_symptom():
    """Predict disease based on symptoms"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        user_id = data.get('user_id')

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        prediction = symptom_predictor.predict(symptoms)
        disease_name = prediction['disease']
        confidence = prediction['confidence']

        disease_data = get_disease_info(disease_name)

        result = {
            'success': True,
            'disease': disease_name,
            'confidence': confidence,
            'description': disease_data['description'],
            'care_tips': disease_data['care_tips'],
            'severity': disease_data['severity']
        }

        if user_id:
            pred_data = {
                'type': 'symptom',
                'input_text': symptoms,
                'disease': disease_name,
                'confidence': confidence,
                'description': disease_data['description'],
                'severity': disease_data['severity'],
                'care_tips': disease_data['care_tips']
            }
            saved = save_prediction(user_id, pred_data)
            if saved:
                result['prediction_id'] = saved.get('id')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Predict condition based on image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        user_id = request.form.get('user_id')

        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower()

        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type'}), 400

        prediction = image_predictor.predict(image_file)
        disease_name = prediction['disease']
        confidence = prediction['confidence']

        disease_data = get_disease_info(disease_name)

        result = {
            'success': True,
            'disease': disease_name,
            'confidence': confidence,
            'description': disease_data['description'],
            'care_tips': disease_data['care_tips'],
            'severity': disease_data['severity']
        }

        if user_id:
            pred_data = {
                'type': 'image',
                'disease': disease_name,
                'confidence': confidence,
                'description': disease_data['description'],
                'severity': disease_data['severity'],
                'care_tips': disease_data['care_tips']
            }
            saved = save_prediction(user_id, pred_data)
            if saved:
                result['prediction_id'] = saved.get('id')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<user_id>', methods=['GET'])
def get_predictions(user_id):
    """Get user's prediction history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        predictions = get_user_predictions(user_id, limit)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/favorites', methods=['POST'])
def toggle_favorite():
    """Add or remove from favorites"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        prediction_id = data.get('prediction_id')
        action = data.get('action', 'add')

        if action == 'add':
            result = add_favorite(user_id, prediction_id)
            return jsonify({'success': bool(result)})
        else:
            result = remove_favorite(user_id, prediction_id)
            return jsonify({'success': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/favorites/<user_id>', methods=['GET'])
def get_favorites(user_id):
    """Get user's favorite predictions"""
    try:
        favorites = get_user_favorites(user_id)
        return jsonify({'success': True, 'favorites': favorites})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['POST', 'GET'])
def user_profile():
    """Manage user profile"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            profile = get_user_profile(user_id)
            return jsonify({'success': True, 'profile': profile})
        else:
            data = request.get_json()
            user_id = data.get('user_id')
            profile = save_user_profile(user_id, data)
            return jsonify({'success': bool(profile), 'profile': profile})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
