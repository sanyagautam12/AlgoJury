// AlgoJury - Enhanced Debug Script
console.log('=== SCRIPT LOADED ===');
console.log('Current URL:', window.location.href);
console.log('Document ready state:', document.readyState);

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded');
    
    // Get elements
    const datasetFileInput = document.getElementById('datasetFile');
    const modelFileInput = document.getElementById('modelFile');
    const targetColumnSelect = document.getElementById('targetColumn');
    const sensitiveFeaturesSelect = document.getElementById('sensitiveFeatures');
    
    console.log('Elements:', {
        datasetFileInput: !!datasetFileInput,
        modelFileInput: !!modelFileInput,
        targetColumnSelect: !!targetColumnSelect,
        sensitiveFeaturesSelect: !!sensitiveFeaturesSelect
    });
    
    // Model file upload handler
    if (modelFileInput) {
        console.log('=== ADDING EVENT LISTENER TO MODEL FILE INPUT ===');
        modelFileInput.addEventListener('change', function(event) {
            console.log('=== MODEL FILE INPUT CHANGED ===');
            const file = event.target.files[0];
            const uploadArea = document.getElementById('modelUploadArea');
            
            if (!file) {
                // Reset UI
                if (uploadArea) {
                    const placeholder = uploadArea.querySelector('.upload-placeholder');
                    if (placeholder) {
                        placeholder.innerHTML = `
                            <i class="fas fa-file-upload fa-3x text-muted mb-3"></i>
                            <p class="text-muted">Click to select your trained model file (.pkl)</p>
                            <small class="text-muted">Supported: Scikit-learn, XGBoost, LightGBM models</small>
                        `;
                    }
                }
                return;
            }
            
            console.log('Model file selected:', file.name, file.size, file.type);
            
            // Show file selection feedback
            if (uploadArea) {
                const placeholder = uploadArea.querySelector('.upload-placeholder');
                if (placeholder) {
                    placeholder.innerHTML = `
                        <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                        <p class="text-success"><strong>Model Selected:</strong> ${file.name}</p>
                        <small class="text-muted">${(file.size / 1024).toFixed(1)} KB</small>
                    `;
                }
            }
            
            // Validation
            const modelExtensions = ['.pkl', '.joblib', '.model', '.json', '.ubj', '.onnx', '.cbm', '.txt', '.dill', '.pickle'];
        const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        if (!modelExtensions.includes(fileExt)) {
            alert(`Please select a valid model file. Supported formats: ${modelExtensions.join(', ')}`);
            return;
        }
        });
    } else {
        console.log('Model file input not found');
    }
    
    if (datasetFileInput) {
        console.log('=== ADDING EVENT LISTENER TO DATASET FILE INPUT ===');
        console.log('Dataset file input element:', datasetFileInput);
        console.log('Input type:', datasetFileInput.type);
        console.log('Input accept:', datasetFileInput.accept);
        
        datasetFileInput.addEventListener('change', function(event) {
            console.log('=== FILE INPUT CHANGED ===');
            console.log('Event:', event);
            console.log('Target:', event.target);
            console.log('Files:', event.target.files);
            console.log('Files length:', event.target.files.length);
            
            const file = event.target.files[0];
            if (!file) {
                console.log('No file selected');
                // Reset UI
                if (targetColumnSelect) {
                    targetColumnSelect.innerHTML = '<option value="">Upload dataset first to see columns...</option>';
                    targetColumnSelect.disabled = true;
                }
                if (sensitiveFeaturesSelect) {
                    sensitiveFeaturesSelect.innerHTML = '<option value="">Upload dataset first to see columns...</option>';
                    sensitiveFeaturesSelect.disabled = true;
                }
                return;
            }
            
            console.log('File selected:', file.name, file.size, file.type);
            
            // Show file selection feedback
            const uploadArea = document.getElementById('datasetUploadArea');
            if (uploadArea) {
                const placeholder = uploadArea.querySelector('.upload-placeholder');
                if (placeholder) {
                    placeholder.innerHTML = `
                        <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                        <p class="text-success"><strong>File Selected:</strong> ${file.name}</p>
                        <small class="text-muted">${(file.size / 1024).toFixed(1)} KB</small>
                    `;
                }
            }
            
            // Simple validation
            const datasetExtensions = ['.csv', '.tsv', '.json', '.jsonl', '.xlsx', '.xls', '.parquet', '.feather', '.pkl', '.h5', '.hdf5'];
        const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        if (!datasetExtensions.includes(fileExt)) {
            alert(`Please select a valid dataset file. Supported formats: ${datasetExtensions.join(', ')}`);
            return;
        }
            
            // Show loading state
            if (targetColumnSelect) {
                targetColumnSelect.innerHTML = '<option value="">Loading columns...</option>';
                targetColumnSelect.disabled = true;
            }
            if (sensitiveFeaturesSelect) {
                sensitiveFeaturesSelect.innerHTML = '<option value="">Loading columns...</option>';
                sensitiveFeaturesSelect.disabled = true;
            }
            
            // Create FormData and send to server
            const formData = new FormData();
            formData.append('dataset_file', file);
            
            console.log('FormData created:', formData);
            console.log('FormData entries:');
            for (let [key, value] of formData.entries()) {
                console.log(key, value);
            }
            
            fetch('/get_columns', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('Response received:', response.status, response.statusText);
                return response.json();
            })
            .then(data => {
                console.log('Data received:', data);
                
                if (data.success && data.columns) {
                    console.log('Populating columns:', data.columns);
                    
                    // Populate target column
                    if (targetColumnSelect) {
                        targetColumnSelect.innerHTML = '<option value="">Select target column...</option>';
                        data.columns.forEach(col => {
                            const option = document.createElement('option');
                            option.value = col;
                            option.textContent = col;
                            targetColumnSelect.appendChild(option);
                        });
                        targetColumnSelect.disabled = false;
                    }
                    
                    // Populate sensitive features
                    if (sensitiveFeaturesSelect) {
                        sensitiveFeaturesSelect.innerHTML = '<option value="">Select sensitive features...</option>';
                        data.columns.forEach(col => {
                            const option = document.createElement('option');
                            option.value = col;
                            option.textContent = col;
                            sensitiveFeaturesSelect.appendChild(option);
                        });
                        sensitiveFeaturesSelect.disabled = false;
                    }
                    
                    console.log('Columns populated successfully');
                } else {
                    console.error('Error in response:', data.error || 'Unknown error');
                    alert('Error loading columns: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Fetch error:', error);
                alert('Error loading columns: ' + error.message);
                
                // Reset UI
                if (targetColumnSelect) {
                    targetColumnSelect.innerHTML = '<option>Error loading columns</option>';
                    targetColumnSelect.disabled = true;
                }
                if (sensitiveFeaturesSelect) {
                    sensitiveFeaturesSelect.innerHTML = '<option>Error loading columns</option>';
                    sensitiveFeaturesSelect.disabled = true;
                }
            });
        });
    } else {
        console.error('Dataset file input not found!');
    }
    
    // Demo button handler
    const runDemoBtn = document.getElementById('runDemoBtn');
    if (runDemoBtn) {
        console.log('Adding demo button event listener');
        runDemoBtn.addEventListener('click', function() {
            console.log('Demo button clicked');
            
            const demoText = document.getElementById('demoText');
            const demoSpinner = document.getElementById('demoSpinner');
            
            // Update button state
            if (demoText) demoText.textContent = 'Running Demo...';
            if (demoSpinner) demoSpinner.classList.remove('d-none');
            runDemoBtn.disabled = true;
            
            fetch('/demo', {
                method: 'POST'
            })
            .then(response => {
                console.log('Demo response:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Demo completed:', data);
                if (data.success) {
                    window.location.href = '/results';
                } else {
                    alert('Demo failed: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Demo error:', error);
                alert('Demo failed: ' + error.message);
            })
            .finally(() => {
                // Reset button state
                if (demoText) demoText.textContent = 'Run Demo Analysis';
                if (demoSpinner) demoSpinner.classList.add('d-none');
                runDemoBtn.disabled = false;
            });
        });
    } else {
        console.log('Demo button not found');
    }
    
    // Form submission handler
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        console.log('Adding form submission event listener');
        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            console.log('Form submitted');
            
            const formData = new FormData(uploadForm);
            const submitBtn = document.getElementById('submitBtn');
            const submitText = document.getElementById('submitText');
            const submitSpinner = document.getElementById('submitSpinner');
            
            // Update submit button
            if (submitText) submitText.textContent = 'Analyzing...';
            if (submitSpinner) submitSpinner.classList.remove('d-none');
            if (submitBtn) submitBtn.disabled = true;
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('Analysis response:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Analysis completed:', data);
                if (data.success) {
                    window.location.href = '/results';
                } else {
                    alert('Analysis failed: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Analysis error:', error);
                alert('Analysis failed: ' + error.message);
            })
            .finally(() => {
                // Reset submit button
                if (submitText) submitText.textContent = 'Analyze Model';
                if (submitSpinner) submitSpinner.classList.add('d-none');
                if (submitBtn) submitBtn.disabled = false;
            });
        });
    } else {
        console.log('Upload form not found');
    }
    
    console.log('Initialization complete');
});