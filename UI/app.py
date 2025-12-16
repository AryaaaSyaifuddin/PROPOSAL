# app.py - ONE PAGE SYSTEM
import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, send_file, flash
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'elm-prediksi-9011'
app.config['DATABASE_FILE'] = 'data/database_pasien.xlsx'

# Load model sekali saat startup
print("🔄 Memuat model ELM...")
try:
    # Load model
    with open('model/elm_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Load preprocessor
    with open('model/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Create model class
    class ELMPredictor:
        def __init__(self, model_data, preprocessor):
            self.W = model_data['W']
            self.b = model_data['b']
            self.beta = model_data['beta']
            self.threshold = model_data['threshold']
            self.activation = model_data['activation']
            self.preprocessor = preprocessor
            self.feature_names = [
                'usia', 'bmi', 'sistolik', 'diastolik', 'proteinuria', 
                'diabetes', 'hb', 'berat_janin', 'cairan_ketuban', 
                'riwayat_hipertensi', 'riwayat_keluarga', 'primigravida', 
                'kehamilan_kembar'
            ]
        
        def predict_proba(self, X):
            H = np.dot(X, self.W) + self.b
            if self.activation == 'sigmoid':
                H = 1 / (1 + np.exp(-np.clip(H, -250, 250)))
            
            y_pred = H @ self.beta
            y_pred_proba = 1 / (1 + np.exp(-np.clip(y_pred, -250, 250)))
            return y_pred_proba
        
        def predict(self, X, threshold=None):
            if threshold is None:
                threshold = self.threshold
            y_pred_proba = self.predict_proba(X)
            return (y_pred_proba >= threshold).astype(int)
    
    model = ELMPredictor(model_data, preprocessor)
    print("✅ Model berhasil dimuat!")
    print(f"   Threshold: {model.threshold:.4f}")
    
except Exception as e:
    print(f"❌ Error memuat model: {str(e)}")
    model = None

def initialize_database():
    """Buat file Excel jika belum ada"""
    if not os.path.exists(app.config['DATABASE_FILE']):
        # Buat DataFrame dengan kolom-kolom yang diperlukan
        columns = model.feature_names + [
            'diagnosis', 'probabilitas_pre_eklampsia', 
            'prediksi_numerik', 'threshold_decision',
            'waktu_prediksi', 'id_pasien'
        ]
        
        df = pd.DataFrame(columns=columns)
        
        # Buat folder data jika belum ada
        os.makedirs('data', exist_ok=True)
        
        # Simpan ke Excel
        df.to_excel(app.config['DATABASE_FILE'], index=False)
        print(f"📁 File database dibuat: {app.config['DATABASE_FILE']}")

def save_to_database(patient_data, diagnosis, probability, prediction):
    """Simpan data ke file Excel"""
    try:
        # Baca file yang sudah ada
        if os.path.exists(app.config['DATABASE_FILE']):
            df_existing = pd.read_excel(app.config['DATABASE_FILE'])
        else:
            initialize_database()
            df_existing = pd.read_excel(app.config['DATABASE_FILE'])
        
        # Buat ID pasien (auto increment)
        if len(df_existing) == 0:
            patient_id = 1
        else:
            patient_id = df_existing['id_pasien'].max() + 1
        
        # Siapkan data baru
        new_data = patient_data.copy()
        new_data['diagnosis'] = diagnosis
        new_data['probabilitas_pre_eklampsia'] = probability
        new_data['prediksi_numerik'] = prediction
        new_data['threshold_decision'] = model.threshold
        new_data['waktu_prediksi'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data['id_pasien'] = patient_id
        
        # Convert ke DataFrame
        df_new = pd.DataFrame([new_data])
        
        # Gabung dengan data existing
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Simpan ke Excel
        df_updated.to_excel(app.config['DATABASE_FILE'], index=False)
        
        return patient_id, len(df_updated)
        
    except Exception as e:
        print(f"❌ Error menyimpan ke database: {e}")
        return None, 0

def preprocess_input(form_data):
    """Proses input dari form"""
    patient_data = {}
    
    # Numerical features
    num_features = ['usia', 'bmi', 'sistolik', 'diastolik', 'hb', 'berat_janin', 'cairan_ketuban']
    for feature in num_features:
        value = form_data.get(feature, '').strip()
        if value:
            value = value.replace(',', '.')
            try:
                patient_data[feature] = float(value)
            except:
                patient_data[feature] = 0.0
        else:
            patient_data[feature] = 0.0
    
    # Categorical features
    cat_mapping = {
        'proteinuria': ['negatif', '+', '++', '+++'],
        'diabetes': ['tidak', 'iya'],
        'riwayat_hipertensi': ['tidak', 'iya'],
        'riwayat_keluarga': ['tidak', 'iya'],
        'primigravida': ['tidak', 'iya'],
        'kehamilan_kembar': ['tidak', 'iya']
    }
    
    for feature, options in cat_mapping.items():
        value = form_data.get(feature, options[0]).strip().lower()
        
        # Standardize values
        if feature == 'proteinuria':
            if value not in options:
                value = 'negatif'
        else:
            if value in ['ya', 'yes', 'y', '1', 'true']:
                value = 'iya'
            elif value in ['tidak', 'no', 'n', '0', 'false']:
                value = 'tidak'
            elif value not in options:
                value = options[0]
        
        patient_data[feature] = value
    
    return patient_data

@app.route('/', methods=['GET', 'POST'])
def index():
    """SATU HALAMAN: Form input + Hasil prediksi"""
    
    # Initialize database jika belum ada
    initialize_database()
    
    # Variabel untuk hasil prediksi
    prediction_result = None
    form_data = {}
    
    # Jika POST request (form submitted)
    if request.method == 'POST' and model is not None:
        try:
            # Simpan data form untuk ditampilkan kembali
            form_data = request.form.to_dict()
            
            # Process input data
            patient_data = preprocess_input(request.form)
            
            # Convert to DataFrame for preprocessing
            df_input = pd.DataFrame([patient_data])
            
            # Apply preprocessor
            X_processed = model.preprocessor.transform(df_input)
            
            # Make prediction
            probability = model.predict_proba(X_processed)[0]
            prediction = int(model.predict(X_processed)[0])
            
            # Diagnosis
            diagnosis = "PRE-EKLAMPSIA" if prediction == 1 else "TIDAK PRE-EKLAMPSIA"
            diagnosis_class = "danger" if prediction == 1 else "success"
            
            # Save to database
            patient_id, total_pasien = save_to_database(patient_data, diagnosis, probability, prediction)
            
            # Prepare result
            prediction_result = {
                'diagnosis': diagnosis,
                'diagnosis_class': diagnosis_class,
                'probabilitas': f"{probability*100:.2f}%",
                'probabilitas_raw': probability,
                'threshold': f"{model.threshold:.4f}",
                'prediksi_numerik': prediction,
                'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id_pasien': patient_id,
                'total_pasien': total_pasien,
                'input_data': patient_data,
                'form_data': form_data  # Untuk populate form kembali
            }
            
            flash(f'✅ Prediksi berhasil! ID Pasien: {patient_id}', 'success')
            
        except Exception as e:
            flash(f'❌ Error: {str(e)}', 'danger')
    
    # Hitung jumlah pasien yang sudah ada
    total_pasien = 0
    if os.path.exists(app.config['DATABASE_FILE']):
        try:
            df = pd.read_excel(app.config['DATABASE_FILE'])
            total_pasien = len(df)
        except:
            pass
    
    return render_template('index.html',
                         model_loaded=model is not None,
                         threshold=model.threshold if model else 0.6243,
                         total_pasien=total_pasien,
                         prediction_result=prediction_result,
                         form_data=form_data)

@app.route('/download')
def download():
    """Download database Excel"""
    try:
        if os.path.exists(app.config['DATABASE_FILE']):
            # Baca file untuk cek jumlah data
            df = pd.read_excel(app.config['DATABASE_FILE'])
            filename = f"database_pasien_{len(df)}_data_{datetime.now().strftime('%Y%m%d')}.xlsx"
            
            return send_file(
                app.config['DATABASE_FILE'],
                as_attachment=True,
                download_name=filename
            )
        else:
            flash('Database belum ada!', 'warning')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    
    return render_template('index.html',
                         model_loaded=model is not None,
                         threshold=model.threshold if model else 0.6243)

@app.route('/clear_database')
def clear_database():
    """Hapus semua data (reset database)"""
    try:
        if os.path.exists(app.config['DATABASE_FILE']):
            # Backup nama file
            backup_name = f"data/backup_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            os.rename(app.config['DATABASE_FILE'], backup_name)
            flash(f'Database berhasil direset! Backup disimpan sebagai: {backup_name}', 'success')
        else:
            flash('Database tidak ditemukan!', 'warning')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    
    return render_template('index.html',
                         model_loaded=model is not None,
                         threshold=model.threshold if model else 0.6243)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Initialize database
    initialize_database()
    
    print("\n" + "="*60)
    print("🚀 SISTEM PREDIKSI PRE-EKLAMPSIA - ONE PAGE")
    print("="*60)
    print("📁 Model: " + ("✅ Loaded" if model else "❌ Not loaded"))
    print(f"💾 Database: {app.config['DATABASE_FILE']}")
    print(f"🌐 Server: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)