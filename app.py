from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import pandas as pd
import joblib
import json
from werkzeug.utils import secure_filename
from model_utils.auditor import ModelAuditor
from model_utils.report_generator import ReportGenerator
from model_utils.compatibility import check_model_compatibility, get_compatibility_guidance
from model_utils.universal_loader import UniversalModelLoader, UniversalDatasetLoader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model file extensions
MODEL_EXTENSIONS = {'pkl', 'joblib', 'model', 'json', 'ubj', 'onnx', 'cbm', 'txt', 'dill', 'pickle'}
# Dataset file extensions
DATASET_EXTENSIONS = {'csv', 'tsv', 'json', 'jsonl', 'xlsx', 'xls', 'parquet', 'feather', 'pkl', 'h5', 'hdf5'}
# Combined allowed extensions
ALLOWED_EXTENSIONS = MODEL_EXTENSIONS | DATASET_EXTENSIONS

def allowed_file(filename, file_type=None):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'model':
        return ext in MODEL_EXTENSIONS
    elif file_type == 'dataset':
        return ext in DATASET_EXTENSIONS
    else:
        return ext in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main upload page"""
    print("[DEBUG] Index page requested")
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    print(f"[DEBUG] Test endpoint called with method: {request.method}")
    return jsonify({'status': 'ok', 'method': request.method})

@app.route('/test_upload')
def test_upload_page():
    return send_from_directory('.', 'test_upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and trigger ML audit"""
    try:
        print("Starting file upload process...")
        
        # Check if files are present
        if 'model_file' not in request.files or 'dataset_file' not in request.files:
            return jsonify({'error': 'Both model and dataset files are required'}), 400
        
        model_file = request.files['model_file']
        dataset_file = request.files['dataset_file']
        target_column = request.form.get('target_column')
        sensitive_features = request.form.getlist('sensitive_features')
        
        # Validate inputs
        if not target_column:
            return jsonify({'error': 'Target column is required'}), 400
        
        if not sensitive_features:
            return jsonify({'error': 'At least one sensitive feature is required'}), 400
        
        # Validate files
        if model_file.filename == '' or dataset_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        print(f"Files received: {model_file.filename}, {dataset_file.filename}")
        
        if not allowed_file(model_file.filename, 'model'):
            return jsonify({'error': f'Invalid model file type. Supported formats: {", ".join(sorted(MODEL_EXTENSIONS))}'}), 400
        
        if not allowed_file(dataset_file.filename, 'dataset'):
            return jsonify({'error': f'Invalid dataset file type. Supported formats: {", ".join(sorted(DATASET_EXTENSIONS))}'}), 400
        
        # Save files with original extensions
        model_filename = secure_filename(model_file.filename)
        dataset_filename = secure_filename(dataset_file.filename)
        
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
        
        model_file.save(model_path)
        dataset_file.save(dataset_path)
        print("Files saved successfully")
        
        # Load and validate data using universal loaders
        try:
            print("Loading model with universal loader...")
            model, model_metadata = UniversalModelLoader.load_model(model_path)
            print(f"Model loaded successfully using {model_metadata['loader_used']}")
            print(f"Model type: {model_metadata['model_type']} ({model_metadata['framework']})")
            
            # Display any warnings
            if model_metadata['warnings']:
                print("Model loading warnings:")
                for warning in model_metadata['warnings']:
                    print(f"  - {warning}")
            
            print("Loading dataset with universal loader...")
            dataset, dataset_metadata = UniversalDatasetLoader.load_dataset(dataset_path)
            print(f"Dataset loaded successfully using {dataset_metadata['loader_used']}")
            print(f"Dataset shape: {dataset_metadata['shape']}")
            
            # Display any dataset warnings
            if dataset_metadata['warnings']:
                print("Dataset loading warnings:")
                for warning in dataset_metadata['warnings']:
                    print(f"  - {warning}")
                    
        except Exception as e:
            print(f"Error loading files: {str(e)}")
            return jsonify({'error': f'Error loading files: {str(e)}'}), 400
        
        # Validate columns exist
        if target_column not in dataset.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in dataset'}), 400
        
        missing_features = [f for f in sensitive_features if f not in dataset.columns]
        if missing_features:
            return jsonify({'error': f'Sensitive features not found: {missing_features}'}), 400
        
        # Run ML audit with timeout protection
        try:
            print("Starting ML audit...")
            auditor = ModelAuditor(model, dataset, target_column, sensitive_features)
            
            # Add timeout handling using threading
            import threading
            import time
            
            audit_results = None
            error_occurred = None
            
            def run_audit_with_timeout():
                nonlocal audit_results, error_occurred
                try:
                    audit_results = auditor.run_audit()
                except Exception as e:
                    error_occurred = str(e)
            
            # Start audit in separate thread
            audit_thread = threading.Thread(target=run_audit_with_timeout)
            audit_thread.daemon = True
            audit_thread.start()
            
            # Wait for completion with timeout (5 minutes)
            audit_thread.join(timeout=300)
            
            if audit_thread.is_alive():
                print("Audit timed out after 5 minutes")
                return jsonify({'error': 'Analysis timed out. Please try with a smaller dataset or simpler model.'}), 408
            
            if error_occurred:
                print(f"Audit error: {error_occurred}")
                return jsonify({'error': f'Error during analysis: {error_occurred}'}), 500
            
            if audit_results is None:
                return jsonify({'error': 'Analysis failed to complete'}), 500
            
            print("Audit completed successfully")
            
            # Generate report
            print("Generating report...")
            report_generator = ReportGenerator(audit_results)
            report_path = report_generator.generate_html_report()
            
            # Cache the results for the results page
            try:
                results_cache_path = os.path.join('reports', 'latest_results.json')
                with open(results_cache_path, 'w') as f:
                    json.dump(audit_results, f, indent=2, default=str)
                print("Results cached successfully")
            except Exception as e:
                print(f"Error caching results: {e}")
            
            print("Upload and analysis completed successfully")
            return jsonify({
                'success': True,
                'audit_results': audit_results,
                'report_path': report_path
            })
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return jsonify({'error': f'Error during analysis: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/results')
def results():
    """Results dashboard page"""
    # Check if there's a recent audit report available
    report_path = os.path.join('reports', 'audit_report.html')
    has_results = os.path.exists(report_path)
    
    # Try to load the most recent audit results if available
    audit_results = None
    if has_results:
        try:
            # Check if there's a cached results file
            results_cache_path = os.path.join('reports', 'latest_results.json')
            if os.path.exists(results_cache_path):
                with open(results_cache_path, 'r') as f:
                    audit_results = json.load(f)
        except Exception as e:
            print(f"Error loading cached results: {e}")
    
    return render_template('results.html', 
                         has_results=has_results, 
                         audit_results=audit_results)

@app.route('/download_report')
def download_report():
    """Download the generated audit report"""
    try:
        report_path = os.path.join('reports', 'audit_report.html')
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True, download_name='ml_audit_report.html')
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading report: {str(e)}'}), 500

@app.route('/get_columns', methods=['POST'])
def get_columns():
    print("[DEBUG] ===== GET_COLUMNS ENDPOINT CALLED =====")
    print(f"[DEBUG] Request method: {request.method}")
    print(f"[DEBUG] Request files: {list(request.files.keys())}")
    print(f"[DEBUG] Request form: {dict(request.form)}")
    print(f"[DEBUG] Request headers: {dict(request.headers)}")
    print(f"[DEBUG] Content type: {request.content_type}")
    
    try:
        if 'dataset_file' not in request.files:
            print("[DEBUG] No dataset_file in request.files")
            return jsonify({'success': False, 'error': 'No dataset file provided'})
        
        file = request.files['dataset_file']
        print(f"[DEBUG] File received: {file.filename}, size: {file.content_length}")
        print(f"[DEBUG] File object: {file}")
        print(f"[DEBUG] File stream: {file.stream}")
        
        if file.filename == '':
            print("[DEBUG] Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read the file content
        file_content = file.read()
        print(f"[DEBUG] File content length: {len(file_content)}")
        print(f"[DEBUG] First 100 chars: {file_content[:100]}")
        
        # Reset file pointer and read as DataFrame
        file.seek(0)
        df = pd.read_csv(file)
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        
        columns = df.columns.tolist()
        print(f"[DEBUG] Columns found: {columns}")
        
        return jsonify({
            'success': True,
            'columns': columns
        })
        
    except Exception as e:
        print(f"[DEBUG] Error in get_columns: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded model and dataset"""
    try:
        print("[DEBUG] ===== ANALYZE ENDPOINT CALLED =====")
        print(f"[DEBUG] Request method: {request.method}")
        print(f"[DEBUG] Request files: {list(request.files.keys())}")
        print(f"[DEBUG] Request form: {dict(request.form)}")
        
        # Check for required files
        if 'model_file' not in request.files or 'dataset_file' not in request.files:
            return jsonify({'success': False, 'error': 'Both model and dataset files are required'})
        
        model_file = request.files['model_file']
        dataset_file = request.files['dataset_file']
        target_column = request.form.get('target_column')
        sensitive_features = request.form.getlist('sensitive_features')
        
        print(f"[DEBUG] Model file: {model_file.filename}")
        print(f"[DEBUG] Dataset file: {dataset_file.filename}")
        print(f"[DEBUG] Target column: {target_column}")
        print(f"[DEBUG] Sensitive features: {sensitive_features}")
        
        if not target_column:
            return jsonify({'success': False, 'error': 'Target column is required'})
        
        if not sensitive_features:
            return jsonify({'success': False, 'error': 'At least one sensitive feature is required'})
        
        # Save uploaded files
        model_path = os.path.join('uploads', model_file.filename)
        dataset_path = os.path.join('uploads', dataset_file.filename)
        
        model_file.save(model_path)
        dataset_file.save(dataset_path)
        
        # Load model and dataset using universal loaders
        model, model_metadata = UniversalModelLoader.load_model(model_path)
        dataset, dataset_metadata = UniversalDatasetLoader.load_dataset(dataset_path)
        
        print(f"[DEBUG] Model loaded with {model_metadata['loader_used']}, type: {model_metadata['model_type']}")
        print(f"[DEBUG] Dataset loaded with {dataset_metadata['loader_used']}, shape: {dataset_metadata['shape']}")
        
        print(f"[DEBUG] Dataset shape: {dataset.shape}")
        print(f"[DEBUG] Dataset columns: {dataset.columns.tolist()}")
        
        # Enhanced feature compatibility handling
        print("[DEBUG] Checking feature compatibility...")
        
        # Try to detect if the model expects encoded features
        try:
            # Test prediction with a small sample to detect feature mismatch
            test_features = dataset.drop(columns=[target_column]).head(1)
            test_pred = model.predict(test_features)
            print("[DEBUG] Direct prediction successful - no encoding needed")
        except Exception as feature_error:
            print(f"[DEBUG] Direct prediction failed: {str(feature_error)}")
            
            # Check if this is a feature mismatch error
            if "feature" in str(feature_error).lower() and "mismatch" in str(feature_error).lower():
                print("[DEBUG] Feature mismatch detected - attempting automatic encoding")
                
                # Try to automatically encode categorical features
                dataset_encoded = dataset.copy()
                
                # Encode sensitive features that might need one-hot encoding
                for sensitive_feature in sensitive_features:
                    if sensitive_feature in dataset.columns:
                        # Check if this feature is categorical
                        if dataset[sensitive_feature].dtype == 'object' or dataset[sensitive_feature].dtype.name == 'category':
                            print(f"[DEBUG] Encoding categorical feature: {sensitive_feature}")
                            
                            # Create one-hot encoded features
                            encoded_features = pd.get_dummies(dataset[sensitive_feature], prefix=sensitive_feature)
                            
                            # Add encoded features to dataset
                            for col in encoded_features.columns:
                                dataset_encoded[col] = encoded_features[col]
                            
                            print(f"[DEBUG] Added encoded features: {encoded_features.columns.tolist()}")
                
                # Test again with encoded dataset
                try:
                    test_features_encoded = dataset_encoded.drop(columns=[target_column]).head(1)
                    test_pred = model.predict(test_features_encoded)
                    print("[DEBUG] Prediction with encoding successful")
                    dataset = dataset_encoded  # Use the encoded version
                except Exception as encoding_error:
                    print(f"[DEBUG] Encoding attempt failed: {str(encoding_error)}")
                    # Continue with original dataset and let the auditor handle it
        
        print(f"[DEBUG] Final dataset shape: {dataset.shape}")
        print(f"[DEBUG] Final dataset columns: {dataset.columns.tolist()}")
        
        # Run ML audit
        auditor = ModelAuditor(model, dataset, target_column, sensitive_features)
        audit_results = auditor.run_audit()
        
        # Generate HTML report
        report_generator = ReportGenerator(audit_results)
        report_path = report_generator.generate_html_report()
        
        # Cache the results
        results_cache_path = os.path.join('reports', 'latest_results.json')
        with open(results_cache_path, 'w') as f:
            json.dump(audit_results, f, indent=2, default=str)
        
        return jsonify({
            'success': True,
            'audit_results': audit_results,
            'report_path': report_path
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[DEBUG] Analysis error: {str(e)}")
        print(f"[DEBUG] Full traceback: {error_details}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/demo', methods=['POST'])
def demo():
    """Run demo analysis - simplified endpoint name"""
    return demo_analysis()

@app.route('/demo_analysis', methods=['POST'])
def demo_analysis():
    """Run demo analysis with pre-built model and dataset"""
    try:
        print("Starting demo analysis...")
        
        # Load pre-built model and dataset
        print("Loading model and dataset...")
        model = joblib.load('demo_model.pkl')
        dataset = pd.read_csv('demo_dataset.csv')
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset columns: {dataset.columns.tolist()}")
        
        # Encode categorical features to match model training
        dataset_encoded = dataset.copy()
        
        # Check if model has label encoders stored
        print("Checking model encoders...")
        if hasattr(model, 'label_encoders'):
            print(f"Model has encoders: {list(model.label_encoders.keys())}")
            # Use the same encoders that were used during training
            if 'gender' in model.label_encoders:
                dataset_encoded['gender_encoded'] = model.label_encoders['gender'].transform(dataset['gender'])
                print("Gender encoded successfully")
            if 'race' in model.label_encoders:
                dataset_encoded['race_encoded'] = model.label_encoders['race'].transform(dataset['race'])
                print("Race encoded successfully")
        else:
            print("No encoders found, creating new ones...")
            # Fallback: create new encoders (this shouldn't happen with our test model)
            from sklearn.preprocessing import LabelEncoder
            le_gender = LabelEncoder()
            le_race = LabelEncoder()
            dataset_encoded['gender_encoded'] = le_gender.fit_transform(dataset['gender'])
            dataset_encoded['race_encoded'] = le_race.fit_transform(dataset['race'])
        
        # Get the feature columns the model was trained on
        print("Getting model features...")
        if hasattr(model, 'feature_columns'):
            model_features = model.feature_columns
            print(f"Model features from stored: {model_features}")
        else:
            # Fallback: assume standard features
            model_features = ['age', 'education_years', 'experience_years', 'previous_salary', 
                            'interview_score', 'technical_score', 'gender_encoded', 'race_encoded']
            print(f"Model features fallback: {model_features}")
        
        # Define target and sensitive features for demo
        target_column = 'hired'
        sensitive_features = ['gender', 'race']  # Keep original names for analysis
        print(f"Target: {target_column}, Sensitive features: {sensitive_features}")
        
        # Create final dataset with model features + target + original sensitive features for analysis
        print("Creating final dataset...")
        required_columns = model_features + [target_column, 'gender', 'race']
        print(f"Required columns: {required_columns}")
        print(f"Available columns: {dataset_encoded.columns.tolist()}")
        
        missing_columns = [col for col in required_columns if col not in dataset_encoded.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        final_dataset = dataset_encoded[required_columns].copy()
        print(f"Final dataset shape: {final_dataset.shape}")
        
        # Run ML audit with properly formatted dataset
        print("Running ML audit...")
        auditor = ModelAuditor(model, final_dataset, target_column, sensitive_features)
        audit_results = auditor.run_audit()
        print("Audit completed successfully")
        
        # Generate HTML report
        print("Generating report...")
        report_generator = ReportGenerator(audit_results)
        report_path = report_generator.generate_html_report()
        print(f"Report generated: {report_path}")
        
        # Cache the results for the results page
        try:
            results_cache_path = os.path.join('reports', 'latest_results.json')
            with open(results_cache_path, 'w') as f:
                json.dump(audit_results, f, indent=2, default=str)
            print("Results cached successfully")
        except Exception as e:
            print(f"Error caching results: {e}")
        
        return jsonify({
            'success': True,
            'audit_results': audit_results,
            'report_path': report_path
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Demo analysis error: {str(e)}")
        print(f"Full traceback: {error_details}")
        return jsonify({'error': f'Demo analysis failed: {str(e)}', 'details': error_details}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)