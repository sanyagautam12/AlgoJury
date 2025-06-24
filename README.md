# AlgoJury - ML Ethics Audit Platform

🔍 **AlgoJury** is a comprehensive web platform for auditing machine learning models for fairness, bias, and ethical concerns. Upload your model and dataset to get detailed analysis, visualizations, and actionable recommendations.

![AlgoJury Platform](https://img.shields.io/badge/Platform-ML%20Ethics%20Audit-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### 🎯 **Core Capabilities**
- **Model Upload**: Support for multiple formats (.pkl, .joblib, .model, .json, .onnx, .cbm, .txt, .dill, .pickle)
- **Dataset Analysis**: Multi-format dataset processing (.csv, .tsv, .json, .jsonl, .xlsx, .xls, .parquet, .feather, .pkl, .h5)
- **Bias Detection**: Comprehensive fairness metrics analysis
- **Explainability**: SHAP-based feature importance and interpretability
- **Visual Reports**: Interactive charts and bias visualizations
- **Audit Reports**: Downloadable HTML reports with findings

### 📊 **Analysis Modules**
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Fairness Metrics**: Demographic Parity, Equalized Odds
- **Group Analysis**: Performance comparison across sensitive groups
- **Feature Importance**: SHAP values and feature impact analysis
- **Bias Visualization**: Heatmaps and comparison charts
- **Recommendations**: Actionable steps for bias mitigation

### 🎨 **User Interface**
- **Modern Design**: Clean, responsive Bootstrap 5 interface
- **Drag & Drop**: Easy file upload with validation
- **Interactive Dashboard**: Collapsible sections and dynamic content
- **Real-time Feedback**: Loading states and error handling
- **Mobile Friendly**: Responsive design for all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd AlgoJury
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```
   
   **Note**: The application will automatically create necessary directories:
   - `uploads/` - For temporary file storage
   - `reports/` - For generated audit reports
   - `static/plots/` - For visualization images

4. **Open your browser**:
   Navigate to `http://localhost:5000`

### 🧪 Testing with Demo Data

1. **Try the built-in demo**:
   - Click "Run Demo Analysis" on the homepage
   - This uses a pre-generated biased hiring model
   - Demonstrates bias detection across gender and race

2. **Generate your own test files**:
   ```bash
   python create_demo_data.py
   ```
   This creates:
   - `demo_model.pkl` - A biased hiring model
   - `demo_dataset.csv` - Sample hiring dataset

3. **Upload and analyze**:
   - Upload `demo_model.pkl` as your model
   - Upload `demo_dataset.csv` as your dataset
   - Select `hired` as target column
   - Select `gender` and `race` as sensitive features
   - Click "Analyze Model"

## 📁 Project Structure

```
AlgoJury/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── create_demo_data.py         # Demo data generator
├── demo_dataset.csv            # Demo dataset (generated)
├── demo_model.pkl              # Demo model (generated)
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── templates/
│   ├── index.html             # Upload page
│   └── results.html           # Results dashboard
├── static/
│   ├── styles.css             # Custom CSS styles
│   ├── script.js              # Frontend JavaScript
│   ├── results.js             # Results page JavaScript
│   └── plots/                 # Generated plot images (auto-created)
├── model_utils/
│   ├── __init__.py            # Package initialization
│   ├── auditor.py             # ML audit logic
│   ├── compatibility.py       # Model compatibility handling
│   ├── model_wrapper.py       # Model wrapper utilities
│   └── report_generator.py    # Report generation
├── uploads/                   # Uploaded files storage (auto-created)
└── reports/                   # Generated reports (auto-created)
```

## 🔧 Usage Guide

### Step 1: Prepare Your Files

**Model Requirements**:
- **Formats**: `.pkl`, `.joblib`, `.model`, `.json`, `.onnx`, `.cbm`, `.txt`, `.dill`, `.pickle`
- **Frameworks**: Scikit-learn, XGBoost, LightGBM, CatBoost, ONNX
- **Compatibility**: Automatic version handling and fallback methods
- Must have `predict()` method (and `predict_proba()` if available)

**Dataset Requirements**:
- **Formats**: `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.pkl`, `.h5`
- Must contain the target column
- Must contain sensitive feature columns
- Should include feature columns used by the model
- Automatic format detection and loading

### Step 2: Upload and Configure

1. **Upload Model**: Drag and drop or click to select your model file (supports multiple formats)
2. **Upload Dataset**: Drag and drop or click to select your dataset file (supports multiple formats)
3. **Select Target Column**: Choose the column your model predicts
4. **Select Sensitive Features**: Choose columns representing protected attributes (e.g., gender, race, age)

### Step 3: Analyze and Review

1. **Submit Analysis**: Click "Analyze Model" to start the audit
2. **Review Results**: Examine the comprehensive dashboard with:
   - Executive summary and risk assessment
   - Overall model performance metrics
   - Fairness analysis across groups
   - SHAP-based feature importance
   - Bias visualizations and heatmaps
   - Actionable recommendations

3. **Download Report**: Get a complete HTML report for documentation

## 📊 Understanding the Results

### Risk Assessment
- **Low Risk (0-40%)**: Model shows minimal bias
- **Medium Risk (40-70%)**: Some bias detected, review recommended
- **High Risk (70%+)**: Significant bias, immediate action required

### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal true positive and false positive rates across groups
- **Selection Rate**: Percentage of positive predictions per group

### SHAP Analysis
- **Feature Importance**: Which features most influence predictions
- **Summary Plots**: Feature impact distribution
- **Waterfall Plots**: Individual prediction explanations

## 🛠️ Technical Details

### Dependencies
- **Flask**: Web framework
- **scikit-learn**: Machine learning utilities
- **SHAP**: Model explainability
- **Fairlearn**: Fairness metrics
- **Pandas**: Data manipulation
- **Matplotlib/Plotly**: Visualizations
- **Bootstrap 5**: Frontend framework

### Supported Models
- Scikit-learn classifiers
- Models with `predict()` and `predict_proba()` methods
- Binary and multi-class classification

### File Size Limits
- Model files: 50MB maximum
- Dataset files: 100MB maximum

### Supported Formats

**Model Formats**:
- `.pkl` - Pickle/Joblib (Scikit-learn, general Python objects)
- `.joblib` - Joblib format (optimized for NumPy arrays)
- `.model` - XGBoost native format
- `.json` - XGBoost JSON format
- `.onnx` - ONNX (Open Neural Network Exchange)
- `.cbm` - CatBoost model format
- `.txt` - LightGBM text format
- `.dill` - Enhanced pickle format
- `.pickle` - Standard pickle format

**Dataset Formats**:
- `.csv` - Comma-separated values
- `.tsv` - Tab-separated values
- `.json` - JSON format
- `.jsonl` - JSON Lines format
- `.xlsx/.xls` - Excel spreadsheets
- `.parquet` - Apache Parquet format
- `.feather` - Apache Arrow Feather format
- `.pkl` - Pickled pandas DataFrame
- `.h5/.hdf5` - HDF5 format

### Version Compatibility

AlgoJury automatically handles version compatibility issues:
- **Automatic Detection**: Detects model framework and version
- **Fallback Methods**: Uses alternative prediction methods when needed
- **Cross-Version Support**: Works with models trained on different library versions
- **Comprehensive Error Handling**: Provides detailed guidance for compatibility issues

## 🔒 Security & Privacy

- **Local Processing**: All analysis happens on your machine
- **No Data Storage**: Uploaded files are temporarily stored and can be deleted
- **No External Calls**: No data sent to external services
- **Session-based**: Results stored in browser session only

## 🐛 Troubleshooting

### Common Issues

**"Model loading failed"**:
- Ensure your model is saved with `joblib.dump()`
- Check that the model has required methods (`predict`, `predict_proba`)
- Verify the file is not corrupted

**"Dataset columns not found"**:
- Ensure CSV has proper headers
- Check for special characters in column names
- Verify the target column exists in the dataset

**"Analysis failed"**:
- Check that dataset features match model expectations
- Ensure sensitive features exist in the dataset
- Verify there are no missing values in critical columns

**"Memory errors"**:
- Reduce dataset size if very large
- Close other applications to free memory
- Consider using a subset of your data for initial analysis

### Getting Help

1. Check the browser console for JavaScript errors
2. Check the Flask console for Python errors
3. Verify all dependencies are installed correctly
4. Ensure file formats and sizes meet requirements

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: Open an issue with details and reproduction steps
2. **Suggest Features**: Propose new functionality or improvements
3. **Submit Code**: Fork, develop, and submit pull requests
4. **Improve Documentation**: Help make the docs clearer and more comprehensive

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes
6. Test thoroughly
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Fairlearn**: Microsoft's fairness assessment toolkit
- **SHAP**: SHapley Additive exPlanations library
- **Scikit-learn**: Machine learning library
- **Bootstrap**: Frontend framework
- **Flask**: Web framework

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**AlgoJury** - Making AI Fair and Transparent 🤖⚖️

*Built with ❤️ for responsible AI development*