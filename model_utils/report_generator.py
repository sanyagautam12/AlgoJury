import os
import json
from datetime import datetime
import base64
from jinja2 import Template

class ReportGenerator:
    """Generate HTML and PDF reports from audit results"""
    
    def __init__(self, audit_results):
        """
        Initialize report generator with audit results
        
        Args:
            audit_results: Dictionary containing all audit findings
        """
        self.audit_results = audit_results
        self.report_dir = 'reports'
        os.makedirs(self.report_dir, exist_ok=True)
    
    def _encode_image_to_base64(self, image_path):
        """Convert image file to base64 string for embedding in HTML"""
        try:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
        return None
    
    def _format_metric(self, value, decimal_places=3):
        """Format numeric metrics for display"""
        if isinstance(value, (int, float)):
            return round(value, decimal_places)
        return value
    
    def _get_risk_color(self, risk_level):
        """Get color code for risk level"""
        colors = {
            'low': '#28a745',    # Green
            'medium': '#ffc107', # Yellow
            'high': '#dc3545'    # Red
        }
        return colors.get(risk_level, '#6c757d')
    
    def generate_html_report(self):
        """Generate comprehensive HTML audit report"""
        try:
            # HTML template
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlgoJury - ML Model Audit Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        
        .section h3 {
            color: #34495e;
            margin-bottom: 15px;
            margin-top: 20px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-label {
            color: #7f8c8d;
            text-transform: uppercase;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .verdict {
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .verdict.low {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        
        .verdict.medium {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        
        .verdict.high {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        .group-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .group-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .group-card h4 {
            color: #495057;
            margin-bottom: 10px;
        }
        
        .group-metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .recommendations {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        
        .recommendations ul {
            margin-left: 20px;
        }
        
        .recommendations li {
            margin-bottom: 8px;
        }
        
        .feature-importance {
            margin-bottom: 20px;
        }
        
        .feature-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .feature-name {
            width: 150px;
            font-weight: 500;
        }
        
        .feature-bar-bg {
            flex: 1;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            margin: 0 10px;
            position: relative;
        }
        
        .feature-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2980b9);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        .feature-value {
            width: 80px;
            text-align: right;
            font-weight: 500;
        }
        
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        .timestamp {
            text-align: center;
            color: #6c757d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .group-metrics {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🔍 AlgoJury Audit Report</h1>
            <p>Comprehensive ML Model Fairness & Bias Analysis</p>
        </div>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            {% if verdict %}
            <div class="verdict {{ verdict.risk_level }}">
                <h3>Overall Assessment</h3>
                <p>{{ verdict.overall_assessment }}</p>
                <p><strong>Risk Level:</strong> {{ verdict.risk_level.title() }}</p>
                {% if verdict.bias_detected %}
                <p><strong>Bias Status:</strong> ⚠️ Bias Detected</p>
                {% else %}
                <p><strong>Bias Status:</strong> ✅ No Significant Bias</p>
                {% endif %}
            </div>
            {% endif %}
        </div>
        
        <!-- Overall Performance -->
        <div class="section">
            <h2>📈 Overall Model Performance</h2>
            {% if overall_metrics %}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ format_metric(overall_metrics.accuracy) }}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ format_metric(overall_metrics.precision) }}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ format_metric(overall_metrics.recall) }}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ format_metric(overall_metrics.f1_score) }}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
            {% endif %}
        </div>
        
        <!-- Fairness Analysis -->
        <div class="section">
            <h2>⚖️ Fairness Analysis</h2>
            {% if fairness_metrics %}
            {% for feature, metrics in fairness_metrics.items() %}
            <h3>{{ feature }}</h3>
            
            {% if metrics.demographic_parity_difference is defined %}
            <p><strong>Demographic Parity Difference:</strong> {{ format_metric(metrics.demographic_parity_difference) }}</p>
            {% endif %}
            
            {% if metrics.equalized_odds_difference is defined %}
            <p><strong>Equalized Odds Difference:</strong> {{ format_metric(metrics.equalized_odds_difference) }}</p>
            {% endif %}
            
            {% if metrics.group_metrics %}
            <h4>Group-wise Performance</h4>
            <div class="group-metrics">
                {% for group, group_metrics in metrics.group_metrics.items() %}
                <div class="group-card">
                    <h4>{{ group }}</h4>
                    <div class="group-metric">
                        <span>Accuracy:</span>
                        <span>{{ format_metric(group_metrics.accuracy) }}</span>
                    </div>
                    <div class="group-metric">
                        <span>Precision:</span>
                        <span>{{ format_metric(group_metrics.precision) }}</span>
                    </div>
                    <div class="group-metric">
                        <span>Recall:</span>
                        <span>{{ format_metric(group_metrics.recall) }}</span>
                    </div>
                    <div class="group-metric">
                        <span>F1 Score:</span>
                        <span>{{ format_metric(group_metrics.f1_score) }}</span>
                    </div>
                    <div class="group-metric">
                        <span>Sample Size:</span>
                        <span>{{ group_metrics.sample_size }}</span>
                    </div>
                    <div class="group-metric">
                        <span>Positive Rate:</span>
                        <span>{{ format_metric(group_metrics.positive_rate) }}</span>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endfor %}
            {% endif %}
        </div>
        
        <!-- Feature Importance -->
        <div class="section">
            <h2>🎯 Feature Importance (SHAP Analysis)</h2>
            {% if shap_analysis and shap_analysis.feature_importance %}
            <div class="feature-importance">
                {% for feature, importance in shap_analysis.feature_importance.items() %}
                {% if loop.index <= 10 %}
                <div class="feature-bar">
                    <div class="feature-name">{{ feature }}</div>
                    <div class="feature-bar-bg">
                        <div class="feature-bar-fill" style="width: {{ (importance / shap_analysis.feature_importance.values() | list | max * 100) }}%"></div>
                    </div>
                    <div class="feature-value">{{ format_metric(importance) }}</div>
                </div>
                {% endif %}
                {% endfor %}
            </div>
            
            {% if shap_analysis.shap_plots %}
            <div class="plot-container">
                <h3>SHAP Visualizations</h3>
                {% if shap_importance_img %}
                <img src="data:image/png;base64,{{ shap_importance_img }}" alt="SHAP Feature Importance">
                {% endif %}
                {% if shap_waterfall_img %}
                <img src="data:image/png;base64,{{ shap_waterfall_img }}" alt="SHAP Waterfall Plot">
                {% endif %}
            </div>
            {% endif %}
            {% endif %}
        </div>
        
        <!-- Recommendations -->
        <div class="section">
            <h2>💡 Recommendations</h2>
            {% if verdict and verdict.recommendations %}
            <div class="recommendations">
                <h3>Action Items</h3>
                <ul>
                    {% for recommendation in verdict.recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        
        <!-- Technical Details -->
        <div class="section">
            <h2>🔧 Technical Details</h2>
            {% if metadata %}
            <div class="metadata">
                <p><strong>Dataset Shape:</strong> {{ metadata.dataset_shape }}</p>
                <p><strong>Target Column:</strong> {{ metadata.target_column }}</p>
                <p><strong>Sensitive Features:</strong> {{ metadata.sensitive_features | join(', ') }}</p>
                <p><strong>Number of Features:</strong> {{ metadata.feature_names | length }}</p>
                <p><strong>Unique Predictions:</strong> {{ metadata.unique_predictions }}</p>
                <p><strong>Unique Targets:</strong> {{ metadata.unique_targets }}</p>
            </div>
            {% endif %}
        </div>
        
        <div class="timestamp">
            <p>Report generated on {{ timestamp }}</p>
            <p>Powered by AlgoJury - ML Ethics Audit Platform</p>
        </div>
    </div>
</body>
</html>
            """
            
            # Prepare template data
            template_data = {
                'overall_metrics': self.audit_results.get('overall_metrics', {}),
                'fairness_metrics': self.audit_results.get('fairness_metrics', {}),
                'shap_analysis': self.audit_results.get('shap_analysis', {}),
                'verdict': self.audit_results.get('verdict', {}),
                'metadata': self.audit_results.get('metadata', {}),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'format_metric': self._format_metric
            }
            
            # Encode SHAP plots as base64 if they exist
            shap_analysis = self.audit_results.get('shap_analysis', {})
            if 'shap_plots' in shap_analysis:
                plots = shap_analysis['shap_plots']
                if 'importance_plot' in plots:
                    template_data['shap_importance_img'] = self._encode_image_to_base64(plots['importance_plot'])
                if 'waterfall_plot' in plots:
                    template_data['shap_waterfall_img'] = self._encode_image_to_base64(plots['waterfall_plot'])
            
            # Render template
            template = Template(html_template)
            html_content = template.render(**template_data)
            
            # Save report
            report_path = os.path.join(self.report_dir, 'audit_report.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML report generated: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            return None
    
    def generate_pdf_report(self):
        """Generate PDF report from HTML (optional feature)"""
        try:
            # First generate HTML report
            html_path = self.generate_html_report()
            if not html_path:
                return None
            
            # Try to convert to PDF using weasyprint
            try:
                from weasyprint import HTML
                pdf_path = os.path.join(self.report_dir, 'audit_report.pdf')
                HTML(filename=html_path).write_pdf(pdf_path)
                print(f"PDF report generated: {pdf_path}")
                return pdf_path
            except ImportError:
                print("WeasyPrint not available. Install it for PDF generation: pip install weasyprint")
                return html_path
            except Exception as e:
                print(f"Error generating PDF: {str(e)}. Returning HTML report instead.")
                return html_path
                
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            return None
    
    def save_results_json(self):
        """Save audit results as JSON for further analysis"""
        try:
            json_path = os.path.join(self.report_dir, 'audit_results.json')
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            clean_results = convert_numpy(self.audit_results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
            
            print(f"JSON results saved: {json_path}")
            return json_path
            
        except Exception as e:
            print(f"Error saving JSON results: {str(e)}")
            return None