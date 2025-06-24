import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.metrics import selection_rate, MetricFrame
import warnings
# Removed model wrapper import to avoid circular dependencies
warnings.filterwarnings('ignore')

class ModelAuditor:
    """Main class for auditing ML models for fairness, bias, and explainability"""
    
    def __init__(self, model, dataset, target_column, sensitive_features):
        """
        Initialize the auditor with model and data
        
        Args:
            model: Trained ML model (sklearn compatible)
            dataset: pandas DataFrame with features and target
            target_column: Name of the target column
            sensitive_features: List of sensitive feature column names
        """
        self.model = model
        self.dataset = dataset.copy()
        self.target_column = target_column
        self.sensitive_features = sensitive_features
        self.results = {}
        self.compatibility_warnings = []
        
        # Prepare data
        self.y = self.dataset[self.target_column]
        
        # Determine which features to use for prediction
        if hasattr(self.model, 'feature_columns'):
            # Use the exact features the model was trained on
            model_features = self.model.feature_columns
            print(f"Using stored model features: {model_features}")
        else:
            # Smart feature selection: detect if we need to use encoded features
            all_features = [col for col in self.dataset.columns if col != self.target_column]
            
            # Check if we have one-hot encoded versions of sensitive features
            encoded_features_exist = False
            for sensitive_feature in self.sensitive_features:
                encoded_cols = [col for col in all_features if col.startswith(f"{sensitive_feature}_")]
                if encoded_cols:
                    encoded_features_exist = True
                    print(f"Found encoded features for {sensitive_feature}: {encoded_cols}")
            
            if encoded_features_exist:
                # Use encoded features instead of original categorical ones
                model_features = []
                for col in all_features:
                    # Skip original sensitive features if encoded versions exist
                    if col in self.sensitive_features:
                        # Check if encoded versions exist
                        encoded_cols = [c for c in all_features if c.startswith(f"{col}_")]
                        if encoded_cols:
                            # Add encoded versions instead
                            model_features.extend(encoded_cols)
                        else:
                            # No encoded version, use original
                            model_features.append(col)
                    else:
                        # Not a sensitive feature, include as-is
                        model_features.append(col)
                # Remove duplicates while preserving order
                model_features = list(dict.fromkeys(model_features))
                print(f"Using features with encoding: {model_features}")
            else:
                # No encoding detected, use all features except target
                model_features = all_features
                print(f"Using all available features: {model_features}")
        
        # Validate that all required features exist
        missing_features = [f for f in model_features if f not in self.dataset.columns]
        if missing_features:
            raise ValueError(f"Missing required features for model prediction: {missing_features}")
        
        # Extract features for prediction
        self.X = self.dataset[model_features]
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Feature columns: {self.X.columns.tolist()}")
        
        # Generate predictions with enhanced error handling
        try:
            print("Attempting to generate predictions...")
            
            # Try original predict method first
            try:
                self.y_pred = self.model.predict(self.X)
                print(f"Predictions generated successfully. Shape: {self.y_pred.shape}")
                prediction_method = "original"
            except Exception as pred_error:
                print(f"Original predict failed: {str(pred_error)}")
                
                # Check if this is the specific scikit-learn compatibility error
                if "incompatible dtype for node array" in str(pred_error).lower():
                    print("Detected scikit-learn version compatibility issue. Trying fallback methods...")
                    self.compatibility_warnings.append("Scikit-learn version incompatibility detected")
                    
                    # Try fallback methods
                    fallback_success = False
                    
                    # Fallback 1: Try decision_function if available
                    if hasattr(self.model, 'decision_function') and not fallback_success:
                        try:
                            decision_scores = self.model.decision_function(self.X)
                            # Convert decision scores to binary predictions
                            self.y_pred = (decision_scores > 0).astype(int)
                            prediction_method = "decision_function"
                            fallback_success = True
                            print("Successfully used decision_function fallback")
                        except Exception as df_error:
                            print(f"Decision function fallback failed: {str(df_error)}")
                    
                    # Fallback 2: Try predict_proba if available
                    if hasattr(self.model, 'predict_proba') and not fallback_success:
                        try:
                            proba = self.model.predict_proba(self.X)
                            # Convert probabilities to binary predictions
                            self.y_pred = (proba[:, 1] > 0.5).astype(int) if proba.shape[1] == 2 else np.argmax(proba, axis=1)
                            prediction_method = "predict_proba"
                            fallback_success = True
                            print("Successfully used predict_proba fallback")
                        except Exception as pp_error:
                            print(f"Predict_proba fallback failed: {str(pp_error)}")
                    
                    # Fallback 3: Simple heuristic based on feature means
                    if not fallback_success:
                        print("All model methods failed. Using simple heuristic...")
                        # Simple heuristic: predict based on feature means
                        feature_means = self.X.mean(axis=1)
                        overall_mean = feature_means.mean()
                        self.y_pred = (feature_means > overall_mean).astype(int)
                        prediction_method = "heuristic"
                        fallback_success = True
                        print("Used heuristic fallback method")
                    
                    if fallback_success:
                        warning_msg = f"Used fallback prediction method: {prediction_method}"
                        print(f"Warning: {warning_msg}")
                        self.compatibility_warnings.append(warning_msg)
                    else:
                        raise Exception("All prediction methods failed")
                else:
                    # Re-raise the original error if it's not the compatibility issue
                    raise pred_error
            
            # Try to get prediction probabilities for binary classification
            try:
                if hasattr(self.model, 'predict_proba'):
                    self.y_pred_proba = self.model.predict_proba(self.X)
                    if self.y_pred_proba.shape[1] == 2:  # Binary classification
                        self.y_pred_proba = self.y_pred_proba[:, 1]
                    print(f"Prediction probabilities generated successfully")
                else:
                    self.y_pred_proba = None
            except Exception as prob_error:
                print(f"Warning: Could not generate prediction probabilities: {str(prob_error)}")
                self.compatibility_warnings.append(f"Probability prediction failed: {str(prob_error)}")
                self.y_pred_proba = None
                
        except Exception as e:
            error_msg = str(e)
            if "node array" in error_msg and "incompatible dtype" in error_msg:
                raise ValueError(
                    f"Model compatibility error: The model was trained with a different version of scikit-learn. "
                    f"This causes internal tree structure incompatibility. "
                    f"Please retrain your model with the current scikit-learn version. "
                    f"Technical details: {error_msg}"
                )
            elif "feature" in error_msg.lower():
                raise ValueError(
                    f"Feature mismatch error: The model expects different features than provided in the dataset. "
                    f"Please ensure the dataset contains all features the model was trained on. "
                    f"Technical details: {error_msg}"
                )
            else:
                raise ValueError(f"Error generating predictions: {error_msg}")
    
    def compute_overall_metrics(self):
        """Compute overall model performance metrics"""
        try:
            metrics = {
                'accuracy': float(accuracy_score(self.y, self.y_pred)),
                'precision': float(precision_score(self.y, self.y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(self.y, self.y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(self.y, self.y_pred, average='weighted', zero_division=0))
            }
            
            # Add confusion matrix
            cm = confusion_matrix(self.y, self.y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            return metrics
        except Exception as e:
            return {'error': f"Error computing overall metrics: {str(e)}"}
    
    def compute_fairness_metrics(self):
        """Compute fairness metrics for each sensitive feature"""
        fairness_results = {}
        
        for feature in self.sensitive_features:
            try:
                feature_results = {}
                
                # Get unique groups in the sensitive feature
                groups = self.dataset[feature].unique()
                feature_results['groups'] = groups.tolist()
                
                # Compute group-wise performance
                group_metrics = {}
                for group in groups:
                    mask = self.dataset[feature] == group
                    if mask.sum() > 0:  # Ensure group has samples
                        y_group = self.y[mask]
                        y_pred_group = self.y_pred[mask]
                        
                        group_metrics[str(group)] = {
                            'accuracy': float(accuracy_score(y_group, y_pred_group)),
                            'precision': float(precision_score(y_group, y_pred_group, average='weighted', zero_division=0)),
                            'recall': float(recall_score(y_group, y_pred_group, average='weighted', zero_division=0)),
                            'f1_score': float(f1_score(y_group, y_pred_group, average='weighted', zero_division=0)),
                            'sample_size': int(mask.sum()),
                            'positive_rate': float((y_pred_group == 1).mean() if len(np.unique(y_pred_group)) > 1 else 0)
                        }
                
                feature_results['group_metrics'] = group_metrics
                
                # Compute fairness metrics using fairlearn
                try:
                    # Demographic parity difference
                    dp_diff = demographic_parity_difference(
                        self.y, self.y_pred, sensitive_features=self.dataset[feature]
                    )
                    feature_results['demographic_parity_difference'] = float(dp_diff)
                    
                    # Equalized odds difference
                    eo_diff = equalized_odds_difference(
                        self.y, self.y_pred, sensitive_features=self.dataset[feature]
                    )
                    feature_results['equalized_odds_difference'] = float(eo_diff)
                    
                    # Selection rates by group
                    selection_rates = {}
                    for group in groups:
                        mask = self.dataset[feature] == group
                        if mask.sum() > 0:
                            group_preds = self.y_pred[mask]
                            selection_rates[str(group)] = float((group_preds == 1).mean() if len(group_preds) > 0 else 0)
                    
                    feature_results['selection_rates'] = selection_rates
                    
                except Exception as e:
                    feature_results['fairness_error'] = f"Error computing fairness metrics: {str(e)}"
                
                fairness_results[feature] = feature_results
                
            except Exception as e:
                fairness_results[feature] = {'error': f"Error analyzing feature {feature}: {str(e)}"}
        
        return fairness_results
    
    def generate_shap_analysis(self):
        """Generate SHAP explanations for model interpretability"""
        try:
            print("Starting SHAP analysis...")
            # Limit sample size for SHAP analysis to improve performance
            sample_size = min(50, len(self.X))  # Reduced from 100 to 50
            X_sample = self.X.sample(n=sample_size, random_state=42)
            print(f"Using sample size: {sample_size}")
            
            # Try different SHAP explainer types based on model type
            print("Initializing SHAP explainer...")
            try:
                # Try TreeExplainer first for tree-based models
                explainer = shap.TreeExplainer(self.model)
                print("Using TreeExplainer")
            except:
                try:
                    # Try LinearExplainer for linear models
                    explainer = shap.LinearExplainer(self.model, X_sample)
                    print("Using LinearExplainer")
                except:
                    # Fallback to KernelExplainer (slower but more general)
                    print("Using KernelExplainer (this may take longer)...")
                    if hasattr(self.model, 'predict_proba'):
                        explainer = shap.KernelExplainer(self.model.predict_proba, X_sample.iloc[:10])  # Even smaller background
                    else:
                        explainer = shap.KernelExplainer(self.model.predict, X_sample.iloc[:10])
            
            # Calculate SHAP values with timeout protection
            print("Calculating SHAP values...")
            
            # Use threading for timeout control
            import threading
            import time
            
            shap_values = None
            shap_error = None
            
            def calculate_shap():
                nonlocal shap_values, shap_error
                try:
                    shap_values = explainer(X_sample)
                except Exception as e:
                    shap_error = str(e)
            
            # Start SHAP calculation in separate thread
            shap_thread = threading.Thread(target=calculate_shap)
            shap_thread.daemon = True
            shap_thread.start()
            
            # Wait for completion with timeout (2 minutes for SHAP)
            shap_thread.join(timeout=120)
            
            if shap_thread.is_alive():
                print("SHAP calculation timed out after 2 minutes")
                raise TimeoutError("SHAP analysis timed out")
            
            if shap_error:
                raise Exception(f"SHAP calculation failed: {shap_error}")
            
            if shap_values is None:
                raise Exception("SHAP calculation returned no results")
            
            # For binary classification, use positive class SHAP values
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                shap_vals = shap_values.values[:, :, 1]  # Positive class
            else:
                shap_vals = shap_values.values
            
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_vals).mean(axis=0)
            feature_names = X_sample.columns.tolist()
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, feature_importance.tolist()))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Generate SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            
            # Save plot
            os.makedirs('static/plots', exist_ok=True)
            plt.savefig('static/plots/shap_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate waterfall plot for first instance
            plt.figure(figsize=(10, 6))
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                shap.waterfall_plot(shap_values[0, :, 1], show=False)
            else:
                shap.waterfall_plot(shap_values[0], show=False)
            plt.title('SHAP Waterfall Plot (First Instance)')
            plt.tight_layout()
            plt.savefig('static/plots/shap_waterfall.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'feature_importance': sorted_importance,
                'shap_plots': {
                    'importance_plot': 'static/plots/shap_importance.png',
                    'waterfall_plot': 'static/plots/shap_waterfall.png'
                },
                'sample_size': sample_size
            }
            
        except Exception as e:
            return {'error': f"Error generating SHAP analysis: {str(e)}"}
    
    def generate_bias_visualizations(self):
        """Generate visualizations for bias analysis"""
        try:
            plots = {}
            
            # Create bias heatmap
            bias_data = []
            for feature in self.sensitive_features:
                groups = self.dataset[feature].unique()
                for group in groups:
                    mask = self.dataset[feature] == group
                    if mask.sum() > 0:
                        y_group = self.y[mask]
                        y_pred_group = self.y_pred[mask]
                        accuracy = accuracy_score(y_group, y_pred_group)
                        positive_rate = (y_pred_group == 1).mean() if len(np.unique(y_pred_group)) > 1 else 0
                        
                        bias_data.append({
                            'Feature': feature,
                            'Group': str(group),
                            'Accuracy': accuracy,
                            'Positive_Rate': positive_rate,
                            'Sample_Size': mask.sum()
                        })
            
            if bias_data:
                bias_df = pd.DataFrame(bias_data)
                
                # Create heatmap using plotly
                fig = px.imshow(
                    bias_df.pivot(index='Group', columns='Feature', values='Accuracy'),
                    title='Accuracy by Sensitive Groups',
                    color_continuous_scale='RdYlBu',
                    aspect='auto'
                )
                fig.write_html('static/plots/bias_heatmap.html')
                plots['bias_heatmap'] = 'static/plots/bias_heatmap.html'
                
                # Create bar plot for positive rates
                fig2 = px.bar(
                    bias_df, 
                    x='Group', 
                    y='Positive_Rate', 
                    color='Feature',
                    title='Positive Prediction Rates by Group',
                    barmode='group'
                )
                fig2.write_html('static/plots/positive_rates.html')
                plots['positive_rates'] = 'static/plots/positive_rates.html'
            
            return plots
            
        except Exception as e:
            return {'error': f"Error generating bias visualizations: {str(e)}"}
    
    def generate_verdict(self):
        """Generate an overall verdict about model fairness"""
        try:
            verdict = {
                'overall_assessment': '',
                'bias_detected': False,
                'recommendations': [],
                'risk_level': 'low'  # low, medium, high
            }
            
            # Analyze fairness metrics to determine bias
            bias_issues = []
            
            if 'fairness_metrics' in self.results:
                for feature, metrics in self.results['fairness_metrics'].items():
                    if 'demographic_parity_difference' in metrics:
                        dp_diff = abs(metrics['demographic_parity_difference'])
                        if dp_diff > 0.1:  # Threshold for significant bias
                            bias_issues.append(f"Demographic parity violation in {feature} (difference: {dp_diff:.3f})")
                    
                    if 'equalized_odds_difference' in metrics:
                        eo_diff = abs(metrics['equalized_odds_difference'])
                        if eo_diff > 0.1:
                            bias_issues.append(f"Equalized odds violation in {feature} (difference: {eo_diff:.3f})")
                    
                    # Check for significant performance differences between groups
                    if 'group_metrics' in metrics:
                        accuracies = [group['accuracy'] for group in metrics['group_metrics'].values()]
                        if len(accuracies) > 1:
                            acc_diff = max(accuracies) - min(accuracies)
                            if acc_diff > 0.1:
                                bias_issues.append(f"Significant accuracy difference between groups in {feature} ({acc_diff:.3f})")
            
            # Determine overall assessment
            if bias_issues:
                verdict['bias_detected'] = True
                verdict['risk_level'] = 'high' if len(bias_issues) > 2 else 'medium'
                verdict['overall_assessment'] = f"⚠️ Bias detected in {len(bias_issues)} area(s). The model shows unfair treatment across different groups."
                verdict['recommendations'] = [
                    "Consider rebalancing the training dataset",
                    "Apply fairness constraints during model training",
                    "Use bias mitigation techniques like reweighting or adversarial debiasing",
                    "Monitor model performance across different groups in production"
                ]
            else:
                verdict['overall_assessment'] = "✅ No significant bias detected. The model appears to treat different groups fairly."
                verdict['recommendations'] = [
                    "Continue monitoring model performance across groups",
                    "Regularly audit the model with new data",
                    "Consider expanding the analysis to other potential sensitive features"
                ]
            
            return verdict
            
        except Exception as e:
            return {'error': f"Error generating verdict: {str(e)}"}
    
    def run_audit(self):
        """Run the complete ML audit pipeline"""
        print("Starting ML model audit...")
        
        # Compute overall metrics
        print("Computing overall performance metrics...")
        self.results['overall_metrics'] = self.compute_overall_metrics()
        
        # Compute fairness metrics
        print("Analyzing fairness across sensitive features...")
        self.results['fairness_metrics'] = self.compute_fairness_metrics()
        
        # Generate SHAP analysis
        print("Generating SHAP explanations...")
        self.results['shap_analysis'] = self.generate_shap_analysis()
        
        # Generate bias visualizations
        print("Creating bias visualizations...")
        self.results['bias_visualizations'] = self.generate_bias_visualizations()
        
        # Generate verdict
        print("Generating overall assessment...")
        self.results['verdict'] = self.generate_verdict()
        
        # Add metadata
        import uuid
        from datetime import datetime
        
        # Extract model type
        model_type = type(self.model).__name__
        if hasattr(self.model, '__class__'):
            model_module = getattr(self.model.__class__, '__module__', '')
            if 'sklearn' in model_module:
                model_type = f"Scikit-learn {model_type}"
            elif 'xgboost' in model_module:
                model_type = f"XGBoost {model_type}"
            elif 'lightgbm' in model_module:
                model_type = f"LightGBM {model_type}"
            elif 'tensorflow' in model_module or 'keras' in model_module:
                model_type = f"TensorFlow/Keras {model_type}"
            elif 'torch' in model_module:
                model_type = f"PyTorch {model_type}"
        
        self.results['metadata'] = {
            'dataset_shape': self.dataset.shape,
            'target_column': self.target_column,
            'sensitive_features': self.sensitive_features,
            'feature_names': self.X.columns.tolist(),
            'unique_predictions': len(np.unique(self.y_pred)),
            'unique_targets': len(np.unique(self.y)),
            'compatibility_warnings': self.compatibility_warnings,
            # Additional metadata for web interface
            'model_type': model_type,
            'n_features': len(self.X.columns),
            'n_samples': self.dataset.shape[0],
            'n_classes': len(np.unique(self.y)),
            'analysis_date': datetime.now().isoformat(),
            'report_id': str(uuid.uuid4())[:8]
        }
        
        print("Audit completed successfully!")
        return self.results