// AlgoJury Results Page JavaScript

// Global variables
let auditResults = null;
let currentReportId = null;
let debugMode = true; // Enable debug logging

// Debug logging function
function debugLog(message, data = null) {
    if (debugMode) {
        console.log(`[AlgoJury Debug] ${message}`, data || '');
    }
}

// Initialize the results page
document.addEventListener('DOMContentLoaded', function() {
    debugLog('DOM Content Loaded - Initializing results page');
    debugLog('Window hasResults:', window.hasResults);
    debugLog('Window backendAuditResults:', window.backendAuditResults);
    
    let results = null;
    
    // First check for backend-provided results
    if (window.hasResults && window.backendAuditResults) {
        debugLog('Using backend-provided results');
        auditResults = window.backendAuditResults;
        currentReportId = auditResults.report_id || 'latest';
        populateResults();
    } else {
        debugLog('No backend results, checking sessionStorage');
        // Check for results in sessionStorage (set by the upload page)
        loadAuditResults();
    }
    
    initializeEventListeners();
});

// Initialize event listeners
function initializeEventListeners() {
    // Download report button
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadReport);
    }
    
    // Back to upload button
    const backBtn = document.getElementById('backToUploadBtn');
    if (backBtn) {
        backBtn.addEventListener('click', function() {
            window.location.href = '/';
        });
    }
    
    // Collapsible sections
    initializeCollapsibleSections();
}

// Load audit results from sessionStorage
function loadAuditResults() {
    try {
        const storedResults = sessionStorage.getItem('auditResults');
        if (storedResults) {
            auditResults = JSON.parse(storedResults);
            currentReportId = auditResults.report_id || 'latest';
            
            // Clear the stored results
            sessionStorage.removeItem('auditResults');
            
            // Populate the results
            populateResults();
        } else {
            console.log('No stored results found, checking if backend provided results');
            if (!window.hasResults || !window.backendAuditResults) {
                showError('No audit results found. Please upload and analyze a model first.');
            }
        }
    } catch (error) {
        console.error('Error loading audit results:', error);
        showError('Error loading audit results. Please try again.');
    }
}

// Validate audit results data structure
function validateAuditResults(results) {
    if (!results || typeof results !== 'object') {
        return { valid: false, error: 'Invalid or missing audit results data' };
    }
    
    const requiredFields = ['verdict', 'overall_metrics', 'fairness_metrics', 'metadata'];
    const missingFields = requiredFields.filter(field => !results[field]);
    
    if (missingFields.length > 0) {
        console.warn('Missing optional fields:', missingFields);
        // Don't fail validation for missing optional fields, just warn
    }
    
    return { valid: true, error: null };
}

// Populate results on the page
function populateResults() {
    console.log('populateResults called, auditResults:', auditResults);
    
    if (!auditResults) {
        console.error('No audit results available');
        showError('No audit results available.');
        return;
    }
    
    // Validate the audit results structure
    const validation = validateAuditResults(auditResults);
    if (!validation.valid) {
        console.error('Invalid audit results:', validation.error);
        showError(`Invalid audit results: ${validation.error}`);
        return;
    }
    
    try {
        console.log('Hiding loading and showing results');
        hideLoading();
        showResults();
        
        console.log('Populating sections...');
        // Populate each section with error handling
        try { populateExecutiveSummary(); } catch (e) { console.error('Error in executive summary:', e); }
        try { populateOverallPerformance(); } catch (e) { console.error('Error in overall performance:', e); }
        try { populateFairnessAnalysis(); } catch (e) { console.error('Error in fairness analysis:', e); }
        try { populateFeatureImportance(); } catch (e) { console.error('Error in feature importance:', e); }
        try { populateBiasVisualizations(); } catch (e) { console.error('Error in bias visualizations:', e); }
        try { populateRecommendations(); } catch (e) { console.error('Error in recommendations:', e); }
        try { populateTechnicalDetails(); } catch (e) { console.error('Error in technical details:', e); }
        
        // Update verdict badge
        try { updateVerdictBadge(); } catch (e) { console.error('Error updating verdict badge:', e); }
        
        console.log('All sections populated successfully');
        
    } catch (error) {
        console.error('Error populating results:', error);
        showError('Error displaying results. Please try again.');
    }
}

// Update verdict badge
function updateVerdictBadge() {
    const verdictBadge = document.getElementById('verdictBadge');
    if (!verdictBadge) {
        console.error('Verdict badge element not found');
        return;
    }
    
    if (!auditResults.verdict) {
        console.error('No verdict data available for badge');
        verdictBadge.innerHTML = '<i class="fas fa-question-circle"></i> No Data';
        verdictBadge.className = 'badge fs-6 bg-secondary';
        return;
    }
    
    const verdict = auditResults.verdict;
    const riskLevel = verdict.risk_level || 'unknown';
    const biasDetected = verdict.bias_detected !== undefined ? verdict.bias_detected : false;
    
    verdictBadge.textContent = biasDetected ? 'Bias Detected' : 'No Bias';
    verdictBadge.className = `badge fs-6 ${getRiskBadgeClass(riskLevel)}`;
}

// Populate executive summary
function populateExecutiveSummary() {
    console.log('populateExecutiveSummary called');
    const container = document.getElementById('executiveSummary');
    console.log('Executive summary container:', container);
    console.log('Verdict data:', auditResults.verdict);
    if (!container) {
        console.error('Missing executive summary container');
        return;
    }
    
    if (!auditResults.verdict) {
        console.error('Missing verdict data');
        container.innerHTML = '<p class="text-muted">No verdict data available.</p>';
        return;
    }
    
    const verdict = auditResults.verdict;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-8">
                <h5>Overall Assessment</h5>
                <p class="lead">${verdict.overall_assessment}</p>
                <div class="alert ${verdict.bias_detected ? 'alert-warning' : 'alert-success'}">
                    <i class="fas ${verdict.bias_detected ? 'fa-exclamation-triangle' : 'fa-check-circle'} me-2"></i>
                    ${verdict.bias_detected ? 'Bias detected in the model' : 'No significant bias detected'}
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <h6 class="card-title">Risk Level</h6>
                        <div class="display-4 ${getRiskTextClass(verdict.risk_level)}">
                            ${verdict.risk_level.toUpperCase()}
                        </div>
                        <small class="text-muted">Bias Risk Assessment</small>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Populate overall performance metrics
function populateOverallPerformance() {
    const container = document.getElementById('overallMetrics');
    if (!container) {
        console.error('Missing overall metrics container');
        return;
    }
    
    if (!auditResults.overall_metrics) {
        console.error('Missing overall metrics data');
        container.innerHTML = '<p class="text-muted">No performance metrics available.</p>';
        return;
    }
    
    const metrics = auditResults.overall_metrics;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${formatMetric(metrics.accuracy)}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${formatMetric(metrics.precision)}</div>
                    <div class="metric-label">Precision</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${formatMetric(metrics.recall)}</div>
                    <div class="metric-label">Recall</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${formatMetric(metrics.f1_score)}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
        </div>
        
        ${metrics.confusion_matrix_plot ? `
            <div class="mt-4">
                <h6>Confusion Matrix</h6>
                <div class="text-center">
                    <img src="data:image/png;base64,${metrics.confusion_matrix_plot}" 
                         class="img-fluid" alt="Confusion Matrix" style="max-width: 500px;">
                </div>
            </div>
        ` : ''}
    `;
}

// Populate fairness analysis
function populateFairnessAnalysis() {
    const container = document.getElementById('fairnessAnalysis');
    if (!container) {
        console.error('Missing fairness analysis container');
        return;
    }
    
    if (!auditResults.fairness_metrics) {
        console.error('Missing fairness metrics data');
        container.innerHTML = '<p class="text-muted">No fairness analysis data available.</p>';
        return;
    }
    
    const fairness = auditResults.fairness_metrics;
    
    let html = `
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Demographic Parity</h6>
                        <div class="d-flex justify-content-between align-items-center">
                            <span>Difference:</span>
                            <span class="badge ${getFairnessMetricClass(fairness.demographic_parity_difference)}">
                                ${formatMetric(fairness.demographic_parity_difference)}
                            </span>
                        </div>
                        <small class="text-muted">Lower values indicate better fairness</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Equalized Odds</h6>
                        <div class="d-flex justify-content-between align-items-center">
                            <span>Difference:</span>
                            <span class="badge ${getFairnessMetricClass(fairness.equalized_odds_difference)}">
                                ${formatMetric(fairness.equalized_odds_difference)}
                            </span>
                        </div>
                        <small class="text-muted">Lower values indicate better fairness</small>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Group-wise performance
    if (fairness.group_performance) {
        html += `
            <h6>Performance by Group</h6>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Group</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            <th>Selection Rate</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        Object.entries(fairness.group_performance).forEach(([group, metrics]) => {
            html += `
                <tr>
                    <td><strong>${group}</strong></td>
                    <td>${formatMetric(metrics.accuracy)}</td>
                    <td>${formatMetric(metrics.precision)}</td>
                    <td>${formatMetric(metrics.recall)}</td>
                    <td>${formatMetric(metrics.f1_score)}</td>
                    <td>${formatMetric(metrics.selection_rate)}</td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// Populate feature importance (SHAP)
function populateFeatureImportance() {
    const container = document.getElementById('featureImportance');
    if (!container) {
        console.error('Missing feature importance container');
        return;
    }
    
    if (!auditResults.shap_analysis) {
        console.error('Missing SHAP analysis data');
        container.innerHTML = '<p class="text-muted">No SHAP analysis data available.</p>';
        return;
    }
    
    const shap = auditResults.shap_analysis;
    
    let html = '';
    
    // SHAP plots from file paths
    if (shap.shap_plots) {
        if (shap.shap_plots.importance_plot) {
            html += `
                <div class="mb-4">
                    <h6 class="feature-importance-heading">SHAP Feature Importance</h6>
                    <div class="text-center">
                        <img src="${shap.shap_plots.importance_plot}" 
                             class="img-fluid" alt="SHAP Feature Importance" style="max-width: 600px;">
                    </div>
                </div>
            `;
        }
        
        if (shap.shap_plots.waterfall_plot) {
            html += `
                <div class="mb-4">
                    <h6 class="feature-importance-heading">SHAP Waterfall Plot</h6>
                    <div class="text-center">
                        <img src="${shap.shap_plots.waterfall_plot}" 
                             class="img-fluid" alt="SHAP Waterfall Plot" style="max-width: 600px;">
                    </div>
                </div>
            `;
        }
    }
    
    // Feature importance table from feature_importance data
    if (shap.feature_importance) {
        html += `
            <div class="mb-4">
                <h6 class="feature-importance-heading">Feature Importance Rankings</h6>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Feature</th>
                                <th>Importance Score</th>
                                <th>Relative Impact</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        const features = Object.entries(shap.feature_importance);
        const maxImportance = Math.max(...features.map(([_, importance]) => importance));
        
        features.forEach(([feature, importance], index) => {
            const relativeImpact = ((importance / maxImportance) * 100).toFixed(1);
            
            html += `
                <tr>
                    <td><strong>#${index + 1}</strong></td>
                    <td>${feature.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                    <td>${formatMetric(importance)}</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${relativeImpact}%" 
                                 aria-valuenow="${relativeImpact}" aria-valuemin="0" aria-valuemax="100">
                                ${relativeImpact}%
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <small class="text-muted">Sample size: ${shap.sample_size || 'Unknown'} instances analyzed</small>
            </div>
        `;
    }
    
    if (!html) {
        html = '<p class="text-muted">No SHAP analysis data available.</p>';
    }
    
    container.innerHTML = html;
}

// Populate bias visualizations
function populateBiasVisualizations() {
    const container = document.getElementById('biasHeatmap');
    if (!container) {
        console.error('Missing bias visualization container');
        return;
    }
    
    if (!auditResults.bias_visualizations) {
        console.error('Missing bias visualization data');
        container.innerHTML = '<p class="text-muted">No bias visualization data available.</p>';
        return;
    }
    
    const viz = auditResults.bias_visualizations;
    let html = '';
    
    // Bias heatmap (HTML file)
    if (viz.bias_heatmap) {
        html += `
            <div class="mb-4">
                <h6>Bias Heatmap</h6>
                <div class="text-center">
                    <iframe src="${viz.bias_heatmap}" 
                            width="100%" height="500" 
                            frameborder="0" 
                            style="max-width: 800px; border: 1px solid #ddd; border-radius: 5px;">
                    </iframe>
                </div>
                <small class="text-muted d-block text-center mt-2">
                    Interactive heatmap showing accuracy across different groups
                </small>
            </div>
        `;
    }
    
    // Positive rates visualization (HTML file)
    if (viz.positive_rates) {
        html += `
            <div class="mb-4">
                <h6>Positive Prediction Rates by Group</h6>
                <div class="text-center">
                    <iframe src="${viz.positive_rates}" 
                            width="100%" height="500" 
                            frameborder="0" 
                            style="max-width: 800px; border: 1px solid #ddd; border-radius: 5px;">
                    </iframe>
                </div>
                <small class="text-muted d-block text-center mt-2">
                    Comparison of positive prediction rates across different demographic groups
                </small>
            </div>
        `;
    }
    
    if (!html) {
        html = '<p class="text-muted">No bias visualization data available.</p>';
    }
    
    container.innerHTML = html;
}

// Populate recommendations
function populateRecommendations() {
    const container = document.getElementById('recommendations');
    if (!container) {
        console.error('Missing recommendations container');
        return;
    }
    
    if (!auditResults.verdict || !auditResults.verdict.recommendations) {
        console.error('Missing recommendations data');
        container.innerHTML = '<p class="text-muted">No recommendations available.</p>';
        return;
    }
    
    const recommendations = auditResults.verdict.recommendations;
    
    let html = '<div class="list-group list-group-flush">';
    
    recommendations.forEach((rec, index) => {
        // Handle both string and object recommendations
        let title, description, priority, action;
        
        if (typeof rec === 'string') {
            title = `Recommendation ${index + 1}`;
            description = rec;
            priority = 'medium';
            action = null;
        } else {
            title = rec.title || `Recommendation ${index + 1}`;
            description = rec.description || rec;
            priority = rec.priority || 'medium';
            action = rec.action || null;
        }
        
        const priorityClass = {
            'high': 'border-danger',
            'medium': 'border-warning', 
            'low': 'border-info'
        }[priority] || 'border-secondary';
        
        const priorityBadge = {
            'high': 'badge-danger',
            'medium': 'badge-warning',
            'low': 'badge-info'
        }[priority] || 'badge-secondary';
        
        const priorityIcon = {
            'high': 'fa-exclamation-triangle',
            'medium': 'fa-exclamation-circle',
            'low': 'fa-info-circle'
        }[priority] || 'fa-lightbulb';
        
        html += `
            <div class="list-group-item ${priorityClass} border-start border-3">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center mb-2">
                            <i class="fas ${priorityIcon} me-2 text-${priority === 'high' ? 'danger' : priority === 'medium' ? 'warning' : 'info'}"></i>
                            <h6 class="mb-0">${title}</h6>
                        </div>
                        <p class="mb-1">${description}</p>
                        ${action ? `<small class="text-muted"><strong>Action:</strong> ${action}</small>` : ''}
                    </div>
                    <span class="badge ${priorityBadge}">${priority.toUpperCase()}</span>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    if (recommendations.length === 0) {
        html = '<p class="text-muted">No recommendations available.</p>';
    }
    
    container.innerHTML = html;
}

// Populate technical details
function populateTechnicalDetails() {
    const container = document.getElementById('technicalDetails');
    if (!container) {
        console.error('Missing technical details container');
        return;
    }
    
    if (!auditResults.metadata) {
        console.error('Missing metadata');
        container.innerHTML = '<p class="text-muted">No technical details available.</p>';
        return;
    }
    
    const metadata = auditResults.metadata;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Model Information</h6>
                <ul class="list-unstyled">
                    <li><strong>Model Type:</strong> ${metadata.model_type || 'Unknown'}</li>
                    <li><strong>Features:</strong> ${metadata.n_features || 'Unknown'}</li>
                    <li><strong>Target Column:</strong> ${metadata.target_column || 'Unknown'}</li>
                    <li><strong>Sensitive Features:</strong> ${metadata.sensitive_features ? metadata.sensitive_features.join(', ') : 'Unknown'}</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6>Dataset Information</h6>
                <ul class="list-unstyled">
                    <li><strong>Samples:</strong> ${metadata.n_samples || 'Unknown'}</li>
                    <li><strong>Classes:</strong> ${metadata.n_classes || 'Unknown'}</li>
                    <li><strong>Analysis Date:</strong> ${metadata.analysis_date ? new Date(metadata.analysis_date).toLocaleString() : 'Unknown'}</li>
                    <li><strong>Report ID:</strong> ${metadata.report_id || 'Unknown'}</li>
                </ul>
            </div>
        </div>
    `;
}

// Download report
async function downloadReport() {
    if (!currentReportId) {
        showError('No report available for download.');
        return;
    }
    
    try {
        setDownloadLoading(true);
        
        const response = await fetch(`/download_report?report_id=${currentReportId}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `algojury_audit_report_${currentReportId}.html`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const error = await response.json();
            showError(error.error || 'Failed to download report.');
        }
    } catch (error) {
        showError('Error downloading report: ' + error.message);
    } finally {
        setDownloadLoading(false);
    }
}

// Set download loading state
function setDownloadLoading(isLoading) {
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (!downloadBtn) return;
    
    if (isLoading) {
        downloadBtn.disabled = true;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Downloading...';
    } else {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<i class="fas fa-download me-2"></i>Download Full Report';
    }
}

// Initialize collapsible sections
function initializeCollapsibleSections() {
    const collapsibleElements = document.querySelectorAll('[data-bs-toggle="collapse"]');
    collapsibleElements.forEach(element => {
        element.addEventListener('click', function() {
            const icon = this.querySelector('i');
            if (icon) {
                setTimeout(() => {
                    const target = document.querySelector(this.getAttribute('data-bs-target'));
                    if (target && target.classList.contains('show')) {
                        icon.classList.remove('fa-chevron-down');
                        icon.classList.add('fa-chevron-up');
                    } else {
                        icon.classList.remove('fa-chevron-up');
                        icon.classList.add('fa-chevron-down');
                    }
                }, 100);
            }
        });
    });
}

// Utility functions
function formatMetric(value) {
    if (value === null || value === undefined || value === '') return 'N/A';
    if (typeof value === 'number') {
        if (isNaN(value) || !isFinite(value)) return 'N/A';
        return value.toFixed(3);
    }
    if (typeof value === 'string' && value.trim() === '') return 'N/A';
    return value.toString();
}

function determineRiskLevel(riskLevel) {
    return riskLevel || 'low';
}

function getRiskBadgeClass(riskLevel) {
    const classes = {
        'high': 'bg-danger',
        'medium': 'bg-warning text-dark',
        'low': 'bg-success',
        'unknown': 'bg-secondary'
    };
    return classes[riskLevel] || 'bg-secondary';
}

function getRiskTextClass(riskLevel) {
    const classes = {
        'high': 'text-danger',
        'medium': 'text-warning',
        'low': 'text-success'
    };
    return classes[riskLevel] || 'text-secondary';
}

function getFairnessMetricClass(value) {
    if (value === null || value === undefined || isNaN(value)) return 'bg-secondary';
    const absValue = Math.abs(value);
    if (absValue >= 0.2) return 'bg-danger';
    if (absValue >= 0.1) return 'bg-warning text-dark';
    return 'bg-success';
}

// Show/hide functions
function showLoading() {
    debugLog('Showing loading state');
    const loading = document.getElementById('loadingState');
    if (loading) {
        loading.classList.remove('d-none');
    } else {
        console.error('Loading state element not found');
    }
}

function hideLoading() {
    debugLog('Hiding loading state');
    const loading = document.getElementById('loadingState');
    if (loading) {
        loading.classList.add('d-none');
    } else {
        console.error('Loading state element not found');
    }
}

function showResults() {
    debugLog('Showing results content');
    const results = document.getElementById('resultsContent');
    if (results) {
        results.classList.remove('d-none');
    } else {
        console.error('Results content element not found');
    }
}

function showError(message) {
    debugLog('Showing error:', message);
    const errorSection = document.getElementById('errorState');
    const errorMessage = document.getElementById('errorMessage');
    
    if (errorSection && errorMessage) {
        errorMessage.textContent = message;
        errorSection.classList.remove('d-none');
        
        // Add debug information in debug mode
        if (debugMode) {
            const debugInfo = document.createElement('div');
            debugInfo.className = 'mt-3 p-3 bg-light border rounded';
            debugInfo.innerHTML = `
                <h6>Debug Information:</h6>
                <ul class="mb-0">
                    <li>Has backend results: ${window.hasResults || false}</li>
                    <li>Backend audit results available: ${!!window.backendAuditResults}</li>
                    <li>Session storage has results: ${!!sessionStorage.getItem('auditResults')}</li>
                    <li>Current audit results: ${!!auditResults}</li>
                </ul>
            `;
            errorMessage.appendChild(debugInfo);
        }
    } else {
        console.error('Error state elements not found');
    }
    
    hideLoading();
}

// Export functions for global access
window.AlgoJuryResults = {
    downloadReport,
    formatMetric,
    showError
};