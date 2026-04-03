# app.py - ONE PAGE SYSTEM with RM & Nama Pasien
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
    """Buat file Excel dengan kolom lengkap jika belum ada"""
    if not os.path.exists(app.config['DATABASE_FILE']):
        # Kolom database: identitas + data medis + hasil
        columns = ['No_RM', 'Nama_Pasien'] + model.feature_names + [
            'diagnosis', 'probabilitas_pre_eklampsia', 
            'prediksi_numerik', 'threshold_decision',
            'waktu_prediksi'
        ]
        
        df = pd.DataFrame(columns=columns)
        
        # Buat folder data jika belum ada
        os.makedirs('data', exist_ok=True)
        
        # Simpan ke Excel
        df.to_excel(app.config['DATABASE_FILE'], index=False)
        print(f"📁 File database dibuat: {app.config['DATABASE_FILE']}")

def save_to_database(patient_data, diagnosis, probability, prediction, no_rm, nama_pasien):
    """Simpan data ke file Excel dengan identitas"""
    try:
        # Baca file yang sudah ada
        if os.path.exists(app.config['DATABASE_FILE']):
            df_existing = pd.read_excel(app.config['DATABASE_FILE'])
        else:
            initialize_database()
            df_existing = pd.read_excel(app.config['DATABASE_FILE'])
        
        # Siapkan data baru
        new_data = {
            'No_RM': no_rm,                     # Format "RM-12345"
            'Nama_Pasien': nama_pasien,
            **patient_data,                      # data medis
            'diagnosis': diagnosis,
            'probabilitas_pre_eklampsia': probability,
            'prediksi_numerik': prediction,
            'threshold_decision': model.threshold,
            'waktu_prediksi': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert ke DataFrame
        df_new = pd.DataFrame([new_data])
        
        # Gabung dengan data existing
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Simpan ke Excel
        df_updated.to_excel(app.config['DATABASE_FILE'], index=False)
        
        # Total pasien setelah simpan
        total_pasien = len(df_updated)
        
        return total_pasien
        
    except Exception as e:
        print(f"❌ Error menyimpan ke database: {e}")
        return 0

def preprocess_input(form_data):
    """Proses input dari form (hanya data medis)"""
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
            
            # Ambil identitas pasien
            no_rm_raw = request.form.get('no_rm', '').strip()
            nama_pasien = request.form.get('nama_pasien', '').strip()
            
            # Validasi identitas wajib
            if not no_rm_raw or not nama_pasien:
                flash('Nomor RM dan Nama Pasien harus diisi!', 'danger')
                return render_template('index.html', model_loaded=True, 
                                     threshold=model.threshold,
                                     total_pasien=get_total_pasien(),
                                     prediction_result=None,
                                     form_data=form_data)
            
            # Format nomor RM dengan prefix RM-
            no_rm = f"RM-{no_rm_raw}"
            
            # Process input data medis
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
            
            # Save to database (dengan identitas)
            total_pasien = save_to_database(patient_data, diagnosis, probability, prediction, no_rm, nama_pasien)
            
            # Prepare result
            prediction_result = {
                'diagnosis': diagnosis,
                'diagnosis_class': diagnosis_class,
                'probabilitas': f"{probability*100:.2f}%",
                'probabilitas_raw': probability,
                'threshold': f"{model.threshold:.4f}",
                'prediksi_numerik': prediction,
                'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_pasien': total_pasien,
                'input_data': patient_data,
                'no_rm': no_rm,               # Tambahkan untuk ditampilkan
                'nama_pasien': nama_pasien,    # Tambahkan untuk ditampilkan
                'form_data': form_data
            }
            
            flash(f'✅ Prediksi berhasil! Data pasien {nama_pasien} (RM: {no_rm}) tersimpan.', 'success')
            
        except Exception as e:
            flash(f'❌ Error: {str(e)}', 'danger')
    
    total_pasien = get_total_pasien()
    
    return render_template('index.html',
                         model_loaded=model is not None,
                         threshold=model.threshold if model else 0.6243,
                         total_pasien=total_pasien,
                         prediction_result=prediction_result,
                         form_data=form_data)

def get_total_pasien():
    """Helper untuk mendapatkan jumlah pasien di database"""
    if os.path.exists(app.config['DATABASE_FILE']):
        try:
            df = pd.read_excel(app.config['DATABASE_FILE'])
            return len(df)
        except:
            return 0
    return 0

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
                         threshold=model.threshold if model else 0.6243,
                         total_pasien=get_total_pasien())

@app.route('/clear_database')
def clear_database():
    """Hapus semua data (reset database)"""
    try:
        if os.path.exists(app.config['DATABASE_FILE']):
            # Backup nama file
            backup_name = f"data/backup_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            os.rename(app.config['DATABASE_FILE'], backup_name)
            flash(f'Database berhasil direset! Backup disimpan sebagai: {backup_name}', 'success')
            
            # Buat ulang database dengan struktur yang benar
            initialize_database()
        else:
            flash('Database tidak ditemukan!', 'warning')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    
    return render_template('index.html',
                         model_loaded=model is not None,
                         threshold=model.threshold if model else 0.6243,
                         total_pasien=get_total_pasien())

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