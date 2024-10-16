import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from flask import Flask, request, render_template, flash, redirect, url_for
import inspect
from huggingface_model_utils import load_model_from_local_path
from datetime import datetime

# Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
aligner = load_model_from_local_path('model/minchul/cvlface_DFA_mobilenet').to(device)
fr_model = load_model_from_local_path('model/minchul/cvlface_adaface_vit_base_webface4m').to(device)

database_path = './face_db.csv'

# Create or reset CSV
def reset_csv(database_path):
    columns = ['id', 'name', 'feat', 'date_registered']
    df = pd.DataFrame(columns=columns)
    df.to_csv(database_path, index=False)

# Image normalization and alignment function
def pil_to_input(pil_image, device):
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return trans(pil_image).unsqueeze(0).to(device)

def get_feat(input_tensor, aligner, fr_model, device):
    input_tensor = input_tensor.to(device)
    aligned_x, _, aligned_ldmks, _, _, _ = aligner(input_tensor)
    input_signature = inspect.signature(fr_model.model.net.forward)
    if input_signature.parameters.get('keypoints') is not None:
        feat = fr_model(aligned_x, aligned_ldmks)
    else:
        feat = fr_model(aligned_x)
    return feat

# Function to get the ID based on image or feature
def get_id(input_image_or_feat, database_path, aligner, fr_model, device, threshold=0.3):
    if not os.path.exists(database_path):
        return None
    db = pd.read_csv(database_path)
    if isinstance(input_image_or_feat, Image.Image):
        input_tensor = pil_to_input(input_image_or_feat, device)
        feat_input = get_feat(input_tensor, aligner, fr_model, device)
    else:
        feat_input = input_image_or_feat
    max_sim = -1
    matched_id = None
    for _, row in db.iterrows():
        feat_db = torch.tensor(eval(row['feat']), device=device)
        cossim = torch.nn.functional.cosine_similarity(feat_input, feat_db).item()
        if cossim > threshold and cossim > max_sim:
            max_sim = cossim
            matched_id = row['id']
    return matched_id

# Save a new face to the database
def save_to_db(feat, name, database_path):
    if not os.path.exists(database_path) or os.stat(database_path).st_size == 0:
        next_id = 1
        db = pd.DataFrame(columns=['id', 'name', 'feat', 'date_registered'])
    else:
        db = pd.read_csv(database_path)
        if pd.to_numeric(db['id'], errors='coerce').isna().all():
            next_id = 1
        else:
            next_id = db['id'].max() + 1
    new_row = pd.DataFrame({
        'id': [next_id],
        'name': [name],
        'feat': [feat.squeeze().cpu().detach().numpy().tolist()],
        'date_registered': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    db = pd.concat([db, new_row], ignore_index=True)
    db.to_csv(database_path, index=False)
    return next_id

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        if 'file' not in request.files or not name:
            flash('Please provide a name and select a file.')
            return redirect(url_for('register'))  # Redirect back to register page for missing details
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file.')
            return redirect(url_for('register'))  # Redirect back to register page for missing file
        
        pil_image = Image.open(file)
        input_tensor = pil_to_input(pil_image, device)
        feat = get_feat(input_tensor, aligner, fr_model, device)
        
        # Check if the face is already registered
        current_id = get_id(feat, database_path, aligner, fr_model, device)
        
        if current_id is None:
            # Save new face
            new_id = save_to_db(feat, name, database_path)
            flash(f'Registration successful with ID: {new_id}')
        else:
            flash(f'The face is already registered with ID: {current_id}')
        
        # Redirect to the 'view-registered' page to display the registration status
        return redirect(url_for('view_registered'))
    return render_template('register.html')

# Recognition page
@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part.')
            return render_template('recognize.html')
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file.')
            return render_template('recognize.html')
        
        pil_image = Image.open(file)
        input_tensor = pil_to_input(pil_image, device)
        feat = get_feat(input_tensor, aligner, fr_model, device)
        id = get_id(feat, database_path, aligner, fr_model, device)
        
        if id is None:
            flash('Face not registered.')
        else:
            flash(f'Face recognized with ID: {id}')
        
        return render_template('recognize.html')
    return render_template('recognize.html')

# View registered faces
# View registered faces with flash messages handling
@app.route('/view-registered')
def view_registered():
    if os.path.exists(database_path):
        db = pd.read_csv(database_path)
        return render_template('registered_face.html', faces=db.to_dict(orient='records'))
    else:
        flash('No registered faces found.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(database_path):
        reset_csv(database_path)
    app.run(debug=True)
