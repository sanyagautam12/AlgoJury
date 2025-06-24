import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict, Optional
import warnings
import sys

# Try importing optional dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import dill
    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False

try:
    import pickle5
    PICKLE5_AVAILABLE = True
except ImportError:
    PICKLE5_AVAILABLE = False

class UniversalModelLoader:
    """Universal loader for various ML model formats and versions"""
    
    @staticmethod
    def load_model(file_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from various formats with automatic format detection
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Tuple of (model, metadata) where metadata contains loading info
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            'file_path': file_path,
            'file_extension': file_ext,
            'loader_used': None,
            'warnings': [],
            'model_type': None,
            'framework': None
        }
        
        # Try different loading methods based on file extension and availability
        loaders = [
            ('joblib', UniversalModelLoader._load_with_joblib),
            ('pickle', UniversalModelLoader._load_with_pickle),
        ]
        
        # Add optional loaders if available
        if DILL_AVAILABLE:
            loaders.append(('dill', UniversalModelLoader._load_with_dill))
        if PICKLE5_AVAILABLE:
            loaders.append(('pickle5', UniversalModelLoader._load_with_pickle5))
        if ONNX_AVAILABLE and file_ext == '.onnx':
            loaders.insert(0, ('onnx', UniversalModelLoader._load_with_onnx))
        if XGB_AVAILABLE and file_ext in ['.model', '.json', '.ubj']:
            loaders.insert(0, ('xgboost', UniversalModelLoader._load_with_xgboost))
        if LGB_AVAILABLE and file_ext == '.txt':
            loaders.insert(0, ('lightgbm', UniversalModelLoader._load_with_lightgbm))
        if CATBOOST_AVAILABLE and file_ext == '.cbm':
            loaders.insert(0, ('catboost', UniversalModelLoader._load_with_catboost))
        
        last_error = None
        
        for loader_name, loader_func in loaders:
            try:
                model = loader_func(file_path)
                metadata['loader_used'] = loader_name
                metadata['model_type'] = type(model).__name__
                metadata['framework'] = UniversalModelLoader._detect_framework(model)
                
                # Add version compatibility info
                UniversalModelLoader._add_version_info(model, metadata)
                
                return model, metadata
                
            except Exception as e:
                last_error = e
                metadata['warnings'].append(f"{loader_name} failed: {str(e)}")
                continue
        
        # If all loaders failed, raise the last error with comprehensive info
        error_msg = f"Failed to load model with all available loaders.\n"
        error_msg += f"File: {file_path}\n"
        error_msg += f"Extension: {file_ext}\n"
        error_msg += "Attempted loaders and errors:\n"
        for warning in metadata['warnings']:
            error_msg += f"  - {warning}\n"
        
        raise ValueError(error_msg)
    
    @staticmethod
    def _load_with_joblib(file_path: str) -> Any:
        """Load model using joblib"""
        return joblib.load(file_path)
    
    @staticmethod
    def _load_with_pickle(file_path: str) -> Any:
        """Load model using standard pickle"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _load_with_dill(file_path: str) -> Any:
        """Load model using dill (enhanced pickle)"""
        with open(file_path, 'rb') as f:
            return dill.load(f)
    
    @staticmethod
    def _load_with_pickle5(file_path: str) -> Any:
        """Load model using pickle5 (Python 3.8+ pickle protocol)"""
        with open(file_path, 'rb') as f:
            return pickle5.load(f)
    
    @staticmethod
    def _load_with_onnx(file_path: str) -> Any:
        """Load ONNX model"""
        # Create ONNX runtime session
        session = ort.InferenceSession(file_path)
        return ONNXModelWrapper(session)
    
    @staticmethod
    def _load_with_xgboost(file_path: str) -> Any:
        """Load XGBoost model"""
        model = xgb.Booster()
        model.load_model(file_path)
        return XGBoostModelWrapper(model)
    
    @staticmethod
    def _load_with_lightgbm(file_path: str) -> Any:
        """Load LightGBM model"""
        model = lgb.Booster(model_file=file_path)
        return LightGBMModelWrapper(model)
    
    @staticmethod
    def _load_with_catboost(file_path: str) -> Any:
        """Load CatBoost model"""
        model = cb.CatBoost()
        model.load_model(file_path)
        return CatBoostModelWrapper(model)
    
    @staticmethod
    def _detect_framework(model: Any) -> str:
        """Detect the ML framework used"""
        model_type = type(model).__name__
        module_name = type(model).__module__
        
        if 'sklearn' in module_name:
            return 'scikit-learn'
        elif 'xgboost' in module_name or 'XGB' in model_type:
            return 'xgboost'
        elif 'lightgbm' in module_name or 'LGB' in model_type:
            return 'lightgbm'
        elif 'catboost' in module_name:
            return 'catboost'
        elif 'onnx' in module_name:
            return 'onnx'
        elif hasattr(model, 'predict') and hasattr(model, 'fit'):
            return 'sklearn-compatible'
        else:
            return 'unknown'
    
    @staticmethod
    def _add_version_info(model: Any, metadata: Dict[str, Any]) -> None:
        """Add version compatibility information"""
        # Check for sklearn version info
        if hasattr(model, '_sklearn_version'):
            metadata['model_sklearn_version'] = model._sklearn_version
        
        # Add current environment info
        metadata['environment'] = {
            'python_version': sys.version,
        }
        
        # Add framework-specific version info
        try:
            import sklearn
            metadata['environment']['sklearn_version'] = sklearn.__version__
        except ImportError:
            pass
        
        if XGB_AVAILABLE:
            metadata['environment']['xgboost_version'] = xgb.__version__
        
        if LGB_AVAILABLE:
            metadata['environment']['lightgbm_version'] = lgb.__version__
        
        if CATBOOST_AVAILABLE:
            metadata['environment']['catboost_version'] = cb.__version__


class UniversalDatasetLoader:
    """Universal loader for various dataset formats"""
    
    @staticmethod
    def load_dataset(file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a dataset from various formats with automatic format detection
        
        Args:
            file_path: Path to the dataset file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Tuple of (dataframe, metadata) where metadata contains loading info
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            'file_path': file_path,
            'file_extension': file_ext,
            'loader_used': None,
            'warnings': [],
            'shape': None,
            'columns': None,
            'dtypes': None
        }
        
        # Define loaders based on file extension
        loaders = {
            '.csv': UniversalDatasetLoader._load_csv,
            '.tsv': UniversalDatasetLoader._load_tsv,
            '.json': UniversalDatasetLoader._load_json,
            '.jsonl': UniversalDatasetLoader._load_jsonl,
            '.xlsx': UniversalDatasetLoader._load_excel,
            '.xls': UniversalDatasetLoader._load_excel,
            '.parquet': UniversalDatasetLoader._load_parquet,
            '.feather': UniversalDatasetLoader._load_feather,
            '.pkl': UniversalDatasetLoader._load_pickle_df,
            '.h5': UniversalDatasetLoader._load_hdf5,
        }
        
        # Try the appropriate loader first
        if file_ext in loaders:
            try:
                df = loaders[file_ext](file_path, **kwargs)
                metadata['loader_used'] = file_ext[1:]  # Remove the dot
                UniversalDatasetLoader._add_dataset_metadata(df, metadata)
                return df, metadata
            except Exception as e:
                metadata['warnings'].append(f"{file_ext[1:]} loader failed: {str(e)}")
        
        # If specific loader failed or extension not recognized, try all loaders
        for ext, loader_func in loaders.items():
            if ext == file_ext:  # Skip if already tried
                continue
            try:
                df = loader_func(file_path, **kwargs)
                metadata['loader_used'] = ext[1:]
                metadata['warnings'].append(f"File extension {file_ext} didn't match, but loaded as {ext[1:]}")
                UniversalDatasetLoader._add_dataset_metadata(df, metadata)
                return df, metadata
            except Exception as e:
                metadata['warnings'].append(f"{ext[1:]} loader failed: {str(e)}")
                continue
        
        # If all loaders failed
        error_msg = f"Failed to load dataset with all available loaders.\n"
        error_msg += f"File: {file_path}\n"
        error_msg += f"Extension: {file_ext}\n"
        error_msg += "Attempted loaders and errors:\n"
        for warning in metadata['warnings']:
            error_msg += f"  - {warning}\n"
        
        raise ValueError(error_msg)
    
    @staticmethod
    def _load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def _load_tsv(file_path: str, **kwargs) -> pd.DataFrame:
        """Load TSV file"""
        kwargs.setdefault('sep', '\t')
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def _load_json(file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSON file"""
        return pd.read_json(file_path, **kwargs)
    
    @staticmethod
    def _load_jsonl(file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSONL (JSON Lines) file"""
        kwargs.setdefault('lines', True)
        return pd.read_json(file_path, **kwargs)
    
    @staticmethod
    def _load_excel(file_path: str, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        return pd.read_excel(file_path, **kwargs)
    
    @staticmethod
    def _load_parquet(file_path: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        return pd.read_parquet(file_path, **kwargs)
    
    @staticmethod
    def _load_feather(file_path: str, **kwargs) -> pd.DataFrame:
        """Load Feather file"""
        return pd.read_feather(file_path, **kwargs)
    
    @staticmethod
    def _load_pickle_df(file_path: str, **kwargs) -> pd.DataFrame:
        """Load pickled DataFrame"""
        return pd.read_pickle(file_path, **kwargs)
    
    @staticmethod
    def _load_hdf5(file_path: str, **kwargs) -> pd.DataFrame:
        """Load HDF5 file"""
        key = kwargs.pop('key', None)
        if key is None:
            # Try to find the first available key
            with pd.HDFStore(file_path, 'r') as store:
                keys = list(store.keys())
                if keys:
                    key = keys[0]
                else:
                    raise ValueError("No keys found in HDF5 file")
        return pd.read_hdf(file_path, key=key, **kwargs)
    
    @staticmethod
    def _add_dataset_metadata(df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """Add dataset metadata"""
        metadata['shape'] = df.shape
        metadata['columns'] = df.columns.tolist()
        metadata['dtypes'] = df.dtypes.to_dict()
        metadata['memory_usage'] = df.memory_usage(deep=True).sum()
        metadata['null_counts'] = df.isnull().sum().to_dict()


# Model wrapper classes for non-sklearn models
class ONNXModelWrapper:
    """Wrapper for ONNX models to provide sklearn-like interface"""
    
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
    
    def predict(self, X):
        """Predict using ONNX model"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: X})
        return result[0]
    
    def predict_proba(self, X):
        """Predict probabilities (if supported)"""
        # This is a simplified implementation
        predictions = self.predict(X)
        if len(predictions.shape) == 1:
            # Binary classification
            proba = np.column_stack([1 - predictions, predictions])
        else:
            proba = predictions
        return proba


class XGBoostModelWrapper:
    """Wrapper for XGBoost models to provide sklearn-like interface"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """Predict using XGBoost model"""
        if isinstance(X, pd.DataFrame):
            X = xgb.DMatrix(X)
        elif isinstance(X, np.ndarray):
            X = xgb.DMatrix(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        predictions = self.predict(X)
        if len(predictions.shape) == 1:
            # Binary classification
            proba = np.column_stack([1 - predictions, predictions])
        else:
            proba = predictions
        return proba


class LightGBMModelWrapper:
    """Wrapper for LightGBM models to provide sklearn-like interface"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """Predict using LightGBM model"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        predictions = self.predict(X)
        if len(predictions.shape) == 1:
            # Binary classification
            proba = np.column_stack([1 - predictions, predictions])
        else:
            proba = predictions
        return proba


class CatBoostModelWrapper:
    """Wrapper for CatBoost models to provide sklearn-like interface"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """Predict using CatBoost model"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)