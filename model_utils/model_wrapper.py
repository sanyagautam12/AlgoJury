import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

class CompatibleModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper class to handle scikit-learn version compatibility issues
    by providing fallback prediction methods when tree structure is incompatible
    """
    
    def __init__(self, original_model):
        self.original_model = original_model
        self.is_tree_based = self._check_if_tree_based()
        self.prediction_method = 'original'
        
    def _check_if_tree_based(self):
        """Check if the model is tree-based"""
        model_name = type(self.original_model).__name__.lower()
        tree_indicators = ['tree', 'forest', 'gradient', 'xgb', 'lgb', 'catboost']
        return any(indicator in model_name for indicator in tree_indicators)
    
    def _safe_predict(self, X):
        """Attempt prediction with fallback methods"""
        try:
            # Try original prediction first
            return self.original_model.predict(X), 'original'
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'node array' in error_msg and 'incompatible dtype' in error_msg:
                print("Detected tree structure incompatibility, attempting workarounds...")
                
                # Try different approaches for tree-based models
                if self.is_tree_based:
                    return self._tree_fallback_predict(X)
                else:
                    raise e
            else:
                raise e
    
    def _tree_fallback_predict(self, X):
        """Fallback prediction methods for tree-based models"""
        try:
            # Method 1: Try to access decision_function if available
            if hasattr(self.original_model, 'decision_function'):
                print("Trying decision_function approach...")
                decision_scores = self.original_model.decision_function(X)
                predictions = (decision_scores > 0).astype(int)
                return predictions, 'decision_function'
        except Exception as e1:
            print(f"Decision function failed: {e1}")
            
        try:
            # Method 2: Try predict_proba and convert to predictions
            if hasattr(self.original_model, 'predict_proba'):
                print("Trying predict_proba approach...")
                probabilities = self.original_model.predict_proba(X)
                predictions = np.argmax(probabilities, axis=1)
                return predictions, 'predict_proba'
        except Exception as e2:
            print(f"Predict_proba failed: {e2}")
            
        try:
            # Method 3: Try to manually traverse tree (for simple cases)
            if hasattr(self.original_model, 'tree_') and hasattr(self.original_model.tree_, 'feature'):
                print("Trying manual tree traversal...")
                return self._manual_tree_predict(X)
        except Exception as e3:
            print(f"Manual tree traversal failed: {e3}")
            
        # Method 4: Use a simple heuristic based on feature importance
        try:
            print("Trying feature importance heuristic...")
            return self._heuristic_predict(X)
        except Exception as e4:
            print(f"Heuristic prediction failed: {e4}")
            
        # If all methods fail, raise the original error with guidance
        raise ValueError(
            "All prediction fallback methods failed. "
            "The model is incompatible with the current scikit-learn version. "
            "Please retrain your model with the current environment."
        )
    
    def _manual_tree_predict(self, X):
        """Attempt manual tree traversal for simple decision trees"""
        # This is a simplified approach - may not work for all tree types
        tree = self.original_model.tree_
        
        # Get basic tree structure info without accessing problematic arrays
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # For each sample, try to make a prediction based on available info
        for i in range(n_samples):
            # Use a simple majority class prediction as fallback
            if hasattr(tree, 'value') and tree.value is not None:
                try:
                    # Get the root node value and use majority class
                    root_value = tree.value[0]
                    predictions[i] = np.argmax(root_value)
                except:
                    predictions[i] = 0  # Default to class 0
            else:
                predictions[i] = 0
                
        return predictions, 'manual_tree'
    
    def _heuristic_predict(self, X):
        """Simple heuristic prediction based on feature statistics"""
        # This is a very basic fallback - just for demonstration
        # In practice, you might want to implement more sophisticated heuristics
        
        n_samples = X.shape[0]
        
        # Simple heuristic: predict based on feature means
        feature_means = X.mean(axis=1)
        overall_mean = feature_means.mean()
        
        predictions = (feature_means > overall_mean).astype(int)
        
        warnings.warn(
            "Using heuristic prediction due to model compatibility issues. "
            "Results may not be accurate. Please retrain your model.",
            UserWarning
        )
        
        return predictions, 'heuristic'
    
    def predict(self, X):
        """Main predict method with compatibility handling"""
        predictions, method = self._safe_predict(X)
        self.prediction_method = method
        
        if method != 'original':
            print(f"Warning: Used fallback prediction method '{method}' due to compatibility issues.")
            
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities with compatibility handling"""
        try:
            return self.original_model.predict_proba(X)
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'node array' in error_msg and 'incompatible dtype' in error_msg:
                print("Predict_proba failed due to compatibility issues, using fallback...")
                
                # Get predictions and convert to probabilities
                predictions = self.predict(X)
                n_classes = len(np.unique(predictions))
                
                # Create simple probability matrix
                probabilities = np.zeros((len(predictions), max(2, n_classes)))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 0.8  # High confidence for predicted class
                    # Distribute remaining probability among other classes
                    other_prob = 0.2 / (probabilities.shape[1] - 1)
                    for j in range(probabilities.shape[1]):
                        if j != pred:
                            probabilities[i, j] = other_prob
                            
                warnings.warn(
                    "Using fallback probability estimation due to compatibility issues.",
                    UserWarning
                )
                
                return probabilities
            else:
                raise e
    
    def get_prediction_method(self):
        """Return the method used for the last prediction"""
        return self.prediction_method
    
    def __getattr__(self, name):
        """Delegate other attributes to the original model"""
        return getattr(self.original_model, name)

def wrap_model_for_compatibility(model):
    """
    Wrap a model with compatibility handling
    
    Args:
        model: Original sklearn model
        
    Returns:
        CompatibleModelWrapper: Wrapped model with fallback methods
    """
    return CompatibleModelWrapper(model)