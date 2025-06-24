import sklearn
import sys
import warnings
import pandas as pd

def check_model_compatibility(model):
    """
    Check model compatibility and provide guidance for version issues
    
    Args:
        model: Loaded sklearn model
        
    Returns:
        dict: Compatibility information and recommendations
    """
    compatibility_info = {
        'current_sklearn_version': sklearn.__version__,
        'python_version': sys.version,
        'warnings': [],
        'recommendations': []
    }
    
    # Check if model has version information
    if hasattr(model, '_sklearn_version'):
        model_sklearn_version = model._sklearn_version
        compatibility_info['model_sklearn_version'] = model_sklearn_version
        
        # Compare versions
        current_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        model_version = tuple(map(int, model_sklearn_version.split('.')[:2]))
        
        if current_version != model_version:
            compatibility_info['warnings'].append(
                f"Version mismatch: Model trained with scikit-learn {model_sklearn_version}, "
                f"current environment has {sklearn.__version__}"
            )
            compatibility_info['recommendations'].append(
                "Consider retraining the model with the current scikit-learn version for best compatibility"
            )
    else:
        compatibility_info['warnings'].append(
            "Model does not contain version information - it may be from an older scikit-learn version"
        )
        compatibility_info['recommendations'].append(
            "Retrain the model with a recent scikit-learn version that includes version metadata"
        )
    
    # Check for tree-based models (common source of compatibility issues)
    model_type = type(model).__name__
    if any(tree_type in model_type for tree_type in ['Tree', 'Forest', 'Gradient', 'XGB', 'LGBM']):
        compatibility_info['model_type'] = 'tree_based'
        compatibility_info['recommendations'].append(
            "Tree-based models are particularly sensitive to version changes. "
            "If you encounter 'node array' errors, retraining is strongly recommended."
        )
    else:
        compatibility_info['model_type'] = 'other'
    
    return compatibility_info

def get_compatibility_guidance(error_message):
    """
    Provide specific guidance based on error message
    
    Args:
        error_message: The error message encountered
        
    Returns:
        dict: Specific guidance and solutions
    """
    guidance = {
        'error_type': 'unknown',
        'solutions': [],
        'prevention': []
    }
    
    error_lower = error_message.lower()
    
    if 'node array' in error_lower and 'incompatible dtype' in error_lower:
        guidance['error_type'] = 'tree_structure_incompatibility'
        guidance['solutions'] = [
            "Retrain your model with the current scikit-learn version",
            "Use the same scikit-learn version that was used to train the model",
            "Try loading the model in a virtual environment with the original scikit-learn version"
        ]
        guidance['prevention'] = [
            "Always save the scikit-learn version alongside your model",
            "Use requirements.txt or environment.yml to track dependencies",
            "Consider using model versioning tools like MLflow or DVC"
        ]
    
    elif 'feature' in error_lower:
        guidance['error_type'] = 'feature_mismatch'
        guidance['solutions'] = [
            "Ensure your dataset contains all features the model was trained on",
            "Check feature names and order match the training data",
            "Verify data preprocessing steps are applied consistently"
        ]
        guidance['prevention'] = [
            "Save feature names and preprocessing steps with your model",
            "Use sklearn pipelines to bundle preprocessing and model",
            "Document your feature engineering process"
        ]
    
    elif 'pickle' in error_lower or 'joblib' in error_lower:
        guidance['error_type'] = 'serialization_error'
        guidance['solutions'] = [
            "Try using pickle instead of joblib or vice versa",
            "Check if the file is corrupted",
            "Ensure the model was saved properly"
        ]
        guidance['prevention'] = [
            "Use consistent serialization methods",
            "Verify model files after saving",
            "Keep backups of important models"
        ]
    
    return guidance

def create_compatibility_report(model, error_message=None):
    """
    Create a comprehensive compatibility report
    
    Args:
        model: The loaded model (if available)
        error_message: Any error message encountered
        
    Returns:
        dict: Complete compatibility report
    """
    report = {
        'timestamp': str(pd.Timestamp.now()),
        'environment': {
            'sklearn_version': sklearn.__version__,
            'python_version': sys.version
        }
    }
    
    if model is not None:
        report['model_compatibility'] = check_model_compatibility(model)
    
    if error_message:
        report['error_guidance'] = get_compatibility_guidance(error_message)
    
    return report