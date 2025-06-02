// HETROFL System Enhanced Plotting Functions
// Modern plotting capabilities with theme support

// Enhanced color palette with theme support
const COLORS = {
    global: '#ef4444',       // Red for global model
    xgboost: '#3b82f6',      // Blue for XGBoost
    catboost: '#10b981',     // Green for CatBoost
    random_forest: '#f59e0b', // Orange for Random Forest
    
    // Theme-aware colors
    get background() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? '#1e293b' : '#ffffff';
    },
    get grid() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? '#334155' : '#e2e8f0';
    },
    get text() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? '#f8fafc' : '#0f172a';
    },
    get paper() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? '#0f172a' : '#ffffff';
    }
};

// Enhanced plot configuration
const PLOT_CONFIG = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    toImageButtonOptions: {
        format: 'png',
        filename: 'hetrofl_chart',
        height: 600,
        width: 1000,
        scale: 2
    }
};

// Error handling for plots
function handlePlotError(containerId, errorMessage) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="plot-error">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <h5>Error Loading Plot</h5>
                <p>${errorMessage}</p>
                <button class="btn btn-sm btn-outline-danger mt-2" onclick="regeneratePlot('${containerId}')">
                    <i class="fas fa-sync-alt me-1"></i> Retry
                </button>
            </div>
        `;
    }
}

// Loading indicator for plots
function showPlotLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner mb-3"></div>
                <p>Loading plot data...</p>
            </div>
        `;
    }
}

// Create real-time accuracy comparison chart
function createAccuracyComparisonChart(containerId, metricsHistory) {
    try {
        showPlotLoading(containerId);
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Prepare data
        const traces = [];
        
        // Global model trace
        if (metricsHistory.global && metricsHistory.global.length > 0) {
            const rounds = metricsHistory.global.map((_, i) => i + 1);
            const accuracy = metricsHistory.global.map(m => m.accuracy * 100); // Convert to percentage
            
            traces.push({
                x: rounds,
                y: accuracy,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Global MLP',
                line: {
                    color: COLORS.global,
                    width: 4
                },
                marker: {
                    size: 8,
                    symbol: 'circle'
                }
            });
        }
        
        // Local models traces
        if (metricsHistory.local) {
            Object.entries(metricsHistory.local).forEach(([modelName, metrics]) => {
                if (metrics && metrics.length > 0) {
                    const rounds = metrics.map((_, i) => i + 1);
                    const accuracy = metrics.map(m => m.accuracy * 100); // Convert to percentage
                    
                    traces.push({
                        x: rounds,
                        y: accuracy,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: modelName.charAt(0).toUpperCase() + modelName.slice(1),
                        line: {
                            color: COLORS[modelName] || '#9b59b6', // Default to purple if color not defined
                            width: 3
                        },
                        marker: {
                            size: 7
                        }
                    });
                }
            });
        }
        
        // If no data, add a message
        if (traces.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-chart-line fa-3x mb-3"></i>
                    <h5>No data available</h5>
                    <p>Start training to see model accuracy comparison.</p>
                </div>
            `;
            return;
        }
        
        // Enhanced layout configuration with theme support
        const layout = {
            title: {
                text: 'Model Accuracy Comparison',
                font: {
                    family: 'Inter, sans-serif',
                    size: 18,
                    color: COLORS.text
                }
            },
            xaxis: {
                title: {
                    text: 'Training Round',
                    font: { family: 'Inter, sans-serif', color: COLORS.text }
                },
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false,
                tickfont: { color: COLORS.text }
            },
            yaxis: {
                title: {
                    text: 'Accuracy (%)',
                    font: { family: 'Inter, sans-serif', color: COLORS.text }
                },
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false,
                range: [0, 100],
                tickfont: { color: COLORS.text }
            },
            font: {
                family: 'Inter, sans-serif',
                color: COLORS.text
            },
            legend: {
                orientation: 'h',
                y: -0.15,
                font: { color: COLORS.text }
            },
            margin: { t: 60, b: 80, l: 60, r: 30 },
            plot_bgcolor: COLORS.background,
            paper_bgcolor: COLORS.paper,
            hovermode: 'x unified',
            height: 400,
            showlegend: true
        };
        
        // Create the plot
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error('Error creating accuracy comparison chart:', error);
        handlePlotError(containerId, 'Failed to create accuracy comparison chart.');
    }
}

// Create the F1 score comparison chart
function createF1ScoreComparisonChart(containerId, metricsHistory) {
    try {
        showPlotLoading(containerId);
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Prepare data
        const traces = [];
        
        // Global model trace
        if (metricsHistory.global && metricsHistory.global.length > 0) {
            const rounds = metricsHistory.global.map((_, i) => i + 1);
            const f1Score = metricsHistory.global.map(m => m.f1_score * 100); // Convert to percentage
            
            traces.push({
                x: rounds,
                y: f1Score,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Global MLP',
                line: {
                    color: COLORS.global,
                    width: 4
                },
                marker: {
                    size: 8,
                    symbol: 'circle'
                }
            });
        }
        
        // Local models traces
        if (metricsHistory.local) {
            Object.entries(metricsHistory.local).forEach(([modelName, metrics]) => {
                if (metrics && metrics.length > 0) {
                    const rounds = metrics.map((_, i) => i + 1);
                    const f1Score = metrics.map(m => (m.f1_score || 0) * 100); // Convert to percentage
                    
                    traces.push({
                        x: rounds,
                        y: f1Score,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: modelName.charAt(0).toUpperCase() + modelName.slice(1),
                        line: {
                            color: COLORS[modelName] || '#9b59b6', // Default to purple if color not defined
                            width: 3
                        },
                        marker: {
                            size: 7
                        }
                    });
                }
            });
        }
        
        // If no data, add a message
        if (traces.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-chart-bar fa-3x mb-3"></i>
                    <h5>No data available</h5>
                    <p>Start training to see model F1 score comparison.</p>
                </div>
            `;
            return;
        }
        
        // Layout configuration
        const layout = {
            title: 'Model F1 Score Comparison',
            xaxis: {
                title: 'Training Round',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false
            },
            yaxis: {
                title: 'F1 Score (%)',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false,
                range: [0, 100]
            },
            font: {
                family: 'Segoe UI, sans-serif',
                color: COLORS.text
            },
            legend: {
                orientation: 'h',
                y: -0.2
            },
            margin: { t: 60, b: 80, l: 60, r: 30 },
            plot_bgcolor: COLORS.background,
            paper_bgcolor: 'white',
            hovermode: 'closest',
            height: 450
        };
        
        // Create the plot
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error('Error creating F1 score comparison chart:', error);
        handlePlotError(containerId, 'Failed to create F1 score comparison chart.');
    }
}

// Create improvement percentages chart
function createImprovementChart(containerId, improvements) {
    try {
        showPlotLoading(containerId);
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Prepare data
        const modelNames = [];
        const accuracyImprovements = [];
        const f1Improvements = [];
        
        // Global model improvements
        if (improvements.global) {
            modelNames.push('Global MLP');
            accuracyImprovements.push(improvements.global.accuracy_improvement || 0);
            f1Improvements.push(improvements.global.f1_score_improvement || 0);
        }
        
        // Local models improvements
        if (improvements.local) {
            Object.entries(improvements.local).forEach(([modelName, modelImprovements]) => {
                modelNames.push(modelName.charAt(0).toUpperCase() + modelName.slice(1));
                accuracyImprovements.push(modelImprovements.accuracy_improvement || 0);
                f1Improvements.push(modelImprovements.f1_score_improvement || 0);
            });
        }
        
        // If no data, add a message
        if (modelNames.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-percentage fa-3x mb-3"></i>
                    <h5>No improvement data available</h5>
                    <p>Complete multiple training rounds to see improvements.</p>
                </div>
            `;
            return;
        }
        
        // Create traces for accuracy and F1 score improvements
        const traces = [
            {
                x: modelNames,
                y: accuracyImprovements.map(val => val * 100), // Convert to percentage
                name: 'Accuracy Improvement',
                type: 'bar',
                marker: {
                    color: '#3498db'
                }
            },
            {
                x: modelNames,
                y: f1Improvements.map(val => val * 100), // Convert to percentage
                name: 'F1 Score Improvement',
                type: 'bar',
                marker: {
                    color: '#2ecc71'
                }
            }
        ];
        
        // Layout configuration
        const layout = {
            title: 'Model Improvement Percentages',
            xaxis: {
                title: 'Model',
                gridcolor: COLORS.grid,
                gridwidth: 1
            },
            yaxis: {
                title: 'Improvement (%)',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: true
            },
            barmode: 'group',
            font: {
                family: 'Segoe UI, sans-serif',
                color: COLORS.text
            },
            legend: {
                orientation: 'h',
                y: -0.2
            },
            margin: { t: 60, b: 80, l: 60, r: 30 },
            plot_bgcolor: COLORS.background,
            paper_bgcolor: 'white',
            height: 450
        };
        
        // Create the plot
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error('Error creating improvement chart:', error);
        handlePlotError(containerId, 'Failed to create improvement chart.');
    }
}

// Create training progress chart (loss over time)
function createTrainingProgressChart(containerId, metricsHistory) {
    try {
        showPlotLoading(containerId);
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Check if we have global metrics with loss data
        if (!metricsHistory.global || !metricsHistory.global.length) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-chart-line fa-3x mb-3"></i>
                    <h5>No training data available</h5>
                    <p>Start training to see progress charts.</p>
                </div>
            `;
            return;
        }
        
        // Extract loss and training time data
        const rounds = metricsHistory.global.map((_, i) => i + 1);
        const loss = metricsHistory.global.map(m => m.loss || 0);
        const trainingTime = metricsHistory.global.map(m => m.training_time || 0);
        
        // Create a subplot with two y-axes
        const traces = [
            {
                x: rounds,
                y: loss,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Loss',
                line: {
                    color: '#e74c3c',
                    width: 3
                },
                marker: {
                    size: 7,
                    symbol: 'circle'
                },
                yaxis: 'y'
            },
            {
                x: rounds,
                y: trainingTime,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Training Time (s)',
                line: {
                    color: '#3498db',
                    width: 3
                },
                marker: {
                    size: 7,
                    symbol: 'square'
                },
                yaxis: 'y2'
            }
        ];
        
        // Layout with dual y-axes
        const layout = {
            title: 'Training Progress',
            xaxis: {
                title: 'Round',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false
            },
            yaxis: {
                title: 'Loss',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false,
                side: 'left'
            },
            yaxis2: {
                title: 'Training Time (s)',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false,
                overlaying: 'y',
                side: 'right'
            },
            font: {
                family: 'Segoe UI, sans-serif',
                color: COLORS.text
            },
            legend: {
                orientation: 'h',
                y: -0.2
            },
            margin: { t: 60, b: 80, l: 60, r: 60 },
            plot_bgcolor: COLORS.background,
            paper_bgcolor: 'white',
            hovermode: 'closest',
            height: 450
        };
        
        // Create the plot
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error('Error creating training progress chart:', error);
        handlePlotError(containerId, 'Failed to create training progress chart.');
    }
}

// Create live performance meter gauge chart
function createPerformanceMeter(containerId, value, title) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Convert value to percentage if it's a decimal
        const percentage = value <= 1 ? value * 100 : value;
        
        // Create gauge chart
        const data = [{
            type: 'indicator',
            mode: 'gauge+number',
            value: percentage,
            title: { text: title, font: { size: 18 } },
            gauge: {
                axis: { range: [0, 100], tickwidth: 1, tickcolor: COLORS.text },
                bar: { color: getColorForPercentage(percentage) },
                bgcolor: 'white',
                borderwidth: 2,
                bordercolor: '#ccc',
                steps: [
                    { range: [0, 50], color: '#ffcccc' },
                    { range: [50, 75], color: '#ffebcc' },
                    { range: [75, 90], color: '#e6ffcc' },
                    { range: [90, 100], color: '#ccffcc' }
                ],
                threshold: {
                    line: { color: 'red', width: 4 },
                    thickness: 0.75,
                    value: 90
                }
            }
        }];
        
        // Layout configuration
        const layout = {
            font: { family: 'Segoe UI, sans-serif' },
            margin: { t: 50, b: 20, l: 20, r: 30 },
            paper_bgcolor: 'white',
            height: 220
        };
        
        // Create the plot
        Plotly.newPlot(containerId, data, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error(`Error creating performance meter for ${title}:`, error);
        container.innerHTML = `<div class="alert alert-danger">Error loading performance meter</div>`;
    }
}

// Create model comparison radar chart
function createModelComparisonRadar(containerId, latestMetrics) {
    try {
        showPlotLoading(containerId);
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Prepare data
        const modelNames = [];
        const modelData = [];
        
        // Add global model if available
        if (latestMetrics.global) {
            modelNames.push('Global MLP');
            modelData.push({
                accuracy: latestMetrics.global.accuracy || 0,
                f1_score: latestMetrics.global.f1_score || 0,
                precision: latestMetrics.global.precision || 0,
                recall: latestMetrics.global.recall || 0
            });
        }
        
        // Add local models if available
        if (latestMetrics.local) {
            Object.entries(latestMetrics.local).forEach(([name, metrics]) => {
                modelNames.push(name.charAt(0).toUpperCase() + name.slice(1));
                modelData.push({
                    accuracy: metrics.accuracy || 0,
                    f1_score: metrics.f1_score || 0,
                    precision: metrics.precision || 0,
                    recall: metrics.recall || 0
                });
            });
        }
        
        // If no data, show message
        if (modelNames.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-chart-pie fa-3x mb-3"></i>
                    <h5>No model data available</h5>
                    <p>Load models to see radar comparison.</p>
                </div>
            `;
            return;
        }
        
        // Create radar chart traces
        const traces = modelNames.map((name, index) => {
            // Convert values to percentages
            const metrics = modelData[index];
            
            return {
                type: 'scatterpolar',
                name: name,
                r: [
                    metrics.accuracy * 100,
                    metrics.f1_score * 100,
                    metrics.precision * 100, 
                    metrics.recall * 100,
                    metrics.accuracy * 100 // Close the polygon
                ],
                theta: ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
                fill: 'toself',
                line: {
                    color: Object.values(COLORS)[index % Object.values(COLORS).length]
                },
                opacity: 0.7
            };
        });
        
        // Layout configuration
        const layout = {
            title: 'Model Metrics Comparison',
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100],
                    angle: 90,
                    ticksuffix: '%'
                }
            },
            font: {
                family: 'Segoe UI, sans-serif',
                color: COLORS.text
            },
            legend: {
                orientation: 'h',
                y: -0.2
            },
            margin: { t: 60, b: 80, l: 60, r: 60 },
            paper_bgcolor: 'white',
            height: 450
        };
        
        // Create the plot
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error('Error creating model comparison radar:', error);
        handlePlotError(containerId, 'Failed to create model comparison radar chart.');
    }
}

// Helper function to get color based on percentage value
function getColorForPercentage(percentage) {
    if (percentage < 50) return '#e74c3c'; // Red
    if (percentage < 75) return '#f39c12'; // Orange
    if (percentage < 90) return '#3498db'; // Blue
    return '#2ecc71'; // Green
}

// Function to regenerate a plot (call from UI)
function regeneratePlot(containerId) {
    // This will be implemented based on the specific plot type
    const plotType = containerId.split('-')[0]; // Assumes IDs like 'accuracy-chart', 'f1-chart', etc.
    
    switch(plotType) {
        case 'accuracy':
            // Send a request to the server to regenerate the accuracy plot
            fetch('/api/plots/regenerate_all')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Reload the page or just the specific plot
                        loadPlots();
                    } else {
                        alert('Error regenerating plots: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error regenerating plots:', error);
                    alert('Error regenerating plots. See console for details.');
                });
            break;
            
        default:
            // Default regeneration logic
            loadPlots();
            break;
    }
}

// Function to update XGBoost model
function rebuildXGBoostModel() {
    showPlotLoading('xgboost-status');
    document.getElementById('xgboost-status').innerHTML = `
        <div class="alert alert-info">
            <i class="fas fa-sync fa-spin me-2"></i>
            Rebuilding XGBoost model... This may take a few minutes.
        </div>
    `;
    
    fetch('/api/models/rebuild_xgboost', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('xgboost-status').innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    XGBoost model rebuilt successfully!
                </div>
            `;
            // Reload model info
            loadModelsData();
        } else {
            document.getElementById('xgboost-status').innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Error rebuilding XGBoost model: ${data.error}
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error rebuilding XGBoost model:', error);
        document.getElementById('xgboost-status').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                Error rebuilding XGBoost model. See console for details.
            </div>
        `;
    });
} 
