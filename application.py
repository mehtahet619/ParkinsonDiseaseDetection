from flask import Flask, request, app, render_template, flash
from flask import Response
import pickle
import numpy as np
import pandas as pd
import os
import librosa
import parselmouth
import parselmouth.praat as praat
import soundfile as sf
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

application = Flask(__name__)
app = application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'  # Required for flashing messages

model = pickle.load(open("Parkinson_disease.pkl", "rb"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    # Load the audio file
    sound = parselmouth.Sound(audio_path)
    
    # Extract pitch features
    pitch = praat.call(sound, "To Pitch", 0.0, 75, 600)
    
    # Get basic frequency measures
    MDVP_Fo_Hz = praat.call(pitch, "Get mean", 0, 0)
    MDVP_Fhi_Hz = praat.call(pitch, "Get maximum", 0, 0)
    MDVP_Flo_Hz = praat.call(pitch, "Get minimum", 0, 0)
    
    # Get jitter measurements
    pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    MDVP_Jitter_percent = praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    MDVP_Jitter_Abs = praat.call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    MDVP_RAP = praat.call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    MDVP_PPQ = praat.call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    Jitter_DDP = MDVP_RAP * 3  # As per standard calculation
    
    # Get shimmer measurements
    MDVP_Shimmer = praat.call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    MDVP_Shimmer_dB = praat.call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    Shimmer_APQ3 = praat.call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    Shimmer_APQ5 = praat.call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    MDVP_APQ = Shimmer_APQ5  # Using APQ5 as MDVP_APQ
    Shimmer_DDA = Shimmer_APQ3 * 3  # As per standard calculation
    
    # Get harmonicity measurements
    harmonicity = praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    NHR = 1 / praat.call(harmonicity, "Get mean", 0, 0)  # Noise-to-Harmonics ratio
    HNR = praat.call(harmonicity, "Get mean", 0, 0)  # Harmonics-to-Noise ratio
    
    # Load audio file using librosa for additional features
    y, sr = librosa.load(audio_path)
    
    # Calculate RPDE (Recurrence Period Density Entropy)
    RPDE = np.mean(librosa.feature.rms(y=y))  # Using RMS as approximation
    
    # Calculate DFA (Detrended Fluctuation Analysis)
    DFA = np.mean(librosa.feature.zero_crossing_rate(y))  # Using zero crossing rate as approximation
    
    # Calculate spread measures
    spread1 = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    spread2 = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # Calculate D2 (Correlation Dimension)
    D2 = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))  # Using spectral rolloff as approximation
    
    # Calculate PPE (Pitch Period Entropy)
    PPE = np.mean(librosa.feature.spectral_flatness(y=y))  # Using spectral flatness as approximation
    
    features = {
        'MDVP_Fo_Hz': MDVP_Fo_Hz,
        'MDVP_Fhi_Hz': MDVP_Fhi_Hz,
        'MDVP_Flo_Hz': MDVP_Flo_Hz,
        'MDVP_Jitter_percent': MDVP_Jitter_percent,
        'MDVP_Jitter_Abs': MDVP_Jitter_Abs,
        'MDVP_RAP': MDVP_RAP,
        'MDVP_PPQ': MDVP_PPQ,
        'Jitter_DDP': Jitter_DDP,
        'MDVP_Shimmer': MDVP_Shimmer,
        'MDVP_Shimmer_dB': MDVP_Shimmer_dB,
        'Shimmer_APQ3': Shimmer_APQ3,
        'Shimmer_APQ5': Shimmer_APQ5,
        'MDVP_APQ': MDVP_APQ,
        'Shimmer_DDA': Shimmer_DDA,
        'NHR': NHR,
        'HNR': HNR,
        'RPDE': RPDE,
        'DFA': DFA,
        'spread1': spread1,
        'spread2': spread2,
        'D2': D2,
        'PPE': PPE
    }
    
    return features

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""
    features = None

    if request.method == 'POST':
        if 'audio_file' in request.files:
            file = request.files['audio_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    features = extract_features(filepath)
                    
                    predict_data = [[
                        features['MDVP_Fo_Hz'], features['MDVP_Fhi_Hz'], features['MDVP_Flo_Hz'],
                        features['MDVP_Jitter_percent'], features['MDVP_Jitter_Abs'],
                        features['MDVP_RAP'], features['MDVP_PPQ'], features['Jitter_DDP'],
                        features['MDVP_Shimmer'], features['MDVP_Shimmer_dB'],
                        features['Shimmer_APQ3'], features['Shimmer_APQ5'], features['MDVP_APQ'],
                        features['Shimmer_DDA'], features['NHR'], features['HNR'],
                        features['RPDE'], features['DFA'], features['spread1'],
                        features['spread2'], features['D2'], features['PPE']
                    ]]
                    
                    predict = model.predict(predict_data)
                    
                    if predict[0] == 1:
                        result = 'Person Has No Parkinson Disease'
                    else:
                        result = 'Person Has Parkinson Disease'
                    
                    os.remove(filepath)  # Clean up the uploaded file
                    
                except Exception as e:
                    result = f"Error processing audio file: {str(e)}"
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                result = "Invalid file format. Please upload a WAV or MP3 file."
        else:
            # Handle manual input case (existing code)
            try:
                MDVP_Fo_Hz = float(request.form.get("MDVP_Fo_Hz"))
                MDVP_Fhi_Hz = float(request.form.get("MDVP_Fhi_Hz"))
                MDVP_Flo_Hz = float(request.form.get("MDVP_Flo_Hz"))
                MDVP_Jitter_percent = float(request.form.get("MDVP_Jitter_percent"))
                MDVP_Jitter_Abs = float(request.form.get("MDVP_Jitter_Abs"))
                MDVP_RAP = float(request.form.get("MDVP_RAP"))
                MDVP_PPQ = float(request.form.get("MDVP_PPQ"))
                Jitter_DDP = float(request.form.get("Jitter_DDP"))
                MDVP_Shimmer = float(request.form.get("MDVP_Shimmer"))
                MDVP_Shimmer_dB = float(request.form.get("MDVP_Shimmer_dB"))
                Shimmer_APQ3 = float(request.form.get("Shimmer_APQ3"))
                Shimmer_APQ5 = float(request.form.get("Shimmer_APQ5"))
                MDVP_APQ = float(request.form.get("MDVP_APQ"))
                Shimmer_DDA = float(request.form.get("Shimmer_DDA"))
                NHR = float(request.form.get("NHR"))
                HNR = float(request.form.get("HNR"))
                RPDE = float(request.form.get("RPDE"))
                DFA = float(request.form.get("DFA"))
                spread1 = float(request.form.get("spread1"))
                spread2 = float(request.form.get("spread2"))
                D2 = float(request.form.get("D2"))
                PPE = float(request.form.get("PPE"))
                
                predict_data = [[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent,
                               MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
                               MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,
                               Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
                
                predict = model.predict(predict_data)
                
                if predict[0] == 1:
                    result = 'Person Has No Parkinson Disease'
                else:
                    result = 'Person Has Parkinson Disease'
            except Exception as e:
                result = f"Error processing form data: {str(e)}"

    return render_template('home.html', result=result, features=features)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)