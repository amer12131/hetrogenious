// HETROFL System Advanced Plotting Functions
// Modern plotting capabilities with real-time updates and advanced visualizations

// Enhanced color palette with gradients and modern colors
const COLORS = {
    global: '#667eea',       // Modern purple for global model
    xgboost: '#06b6d4',      // Cyan for XGBoost
    catboost: '#10b981',     // Emerald for CatBoost
    random_forest: '#f59e0b', // Amber for Random Forest
    background: 'rgba(255, 255, 255, 0.1)',
    grid: 'rgba(255, 255, 255, 0.2)',
    text: '#1f2937',
    
    // Gradient colors
    gradients: {
        primary: ['#667eea', '#764ba2'],
        success: ['#10b981', '#34d399'],
        warning: ['#f59e0b', '#fbbf24'],
        danger: ['#ef4444', '#f87171'],
        info: ['#06b6d4', '#22d3ee']
    },
    
    // Chart specific colors
    chart: {
        accuracy: '#667eea',
        f1_score: '#10b981',
        precision: '#f59e0b',
        recall: '#ef4444',
        loss: '#8b5cf6'
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
        filename: 'hetrofl_plot',
        height: 800,
        width: 1200,
        scale: 2
    }
};

// Modern plot layout template
const MODERN_LAYOUT = {
    font: {
        family: 'Inter, system-ui, -apple-system, sans-serif',
        size: 12,
        color: COLORS.text
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 60, b: 60, l: 60, r: 60 },
    showlegend: true,
    legend: {
        orientation: 'h',
        y: -0.15,
        x: 0.5,
        xanchor: 'center',
        bgcolor: 'rgba(255,255,255,0.8)',
        bordercolor: 'rgba(0,0,0,0.1)',
        borderwidth: 1
    },
    xaxis: {
        gridcolor: COLORS.grid,
        gridwidth: 1,
        zeroline: false,
        showline: true,
        linecolor: COLORS.grid
    },
    yaxis: {
        gridcolor: COLORS.grid,
        gridwidth: 1,
        zeroline: false,
        showline: true,
        linecolor: COLORS.grid
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
        
        // Layout configuration
        const layout = {
            title: 'Model Accuracy Comparison',
            xaxis: {
                title: 'Training Round',
                gridcolor: COLORS.grid,
                gridwidth: 1,
                zeroline: false
            },
            yaxis: {
                title: 'Accuracy (%)',
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

// Advanced plot creation functions

// Create real-time confusion matrix
function createConfusionMatrix(containerId) {
    showPlotLoading(containerId);
    
    fetch('/api/plots/realtime/confusion_matrix')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                handlePlotError(containerId, data.error);
            } else {
                const plotData = JSON.parse(data.plot_json);
                Plotly.newPlot(containerId, plotData.data, plotData.layout, PLOT_CONFIG);
            }
        })
        .catch(error => {
            console.error('Error creating confusion matrix:', error);
            handlePlotError(containerId, 'Failed to load confusion matrix');
        });
}

// Create feature importance plot
function createFeatureImportance(containerId) {
    showPlotLoading(containerId);
    
    fetch('/api/plots/realtime/feature_importance')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                handlePlotError(containerId, data.error);
            } else {
                const plotData = JSON.parse(data.plot_json);
                Plotly.newPlot(containerId, plotData.data, plotData.layout, PLOT_CONFIG);
            }
        })
        .catch(error => {
            console.error('Error creating feature importance:', error);
            handlePlotError(containerId, 'Failed to load feature importance');
        });
}

// Create system resource usage plot
function createResourceUsage(containerId) {
    showPlotLoading(containerId);
    
    fetch('/api/plots/realtime/resource_usage')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                handlePlotError(containerId, data.error);
            } else {
                const plotData = JSON.parse(data.plot_json);
                Plotly.newPlot(containerId, plotData.data, plotData.layout, PLOT_CONFIG);
            }
        })
        .catch(error => {
            console.error('Error creating resource usage:', error);
            handlePlotError(containerId, 'Failed to load resource usage');
        });
}

// Create model architecture visualization
function createModelArchitecture(containerId) {
    showPlotLoading(containerId);
    
    fetch('/api/plots/realtime/model_architecture')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                handlePlotError(containerId, data.error);
            } else {
                const plotData = JSON.parse(data.plot_json);
                Plotly.newPlot(containerId, plotData.data, plotData.layout, PLOT_CONFIG);
            }
        })
        .catch(error => {
            console.error('Error creating model architecture:', error);
            handlePlotError(containerId, 'Failed to load model architecture');
        });
}

// Create animated gauge chart for performance metrics
function createPerformanceGauge(containerId, value, title, color = COLORS.chart.accuracy) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const percentage = value <= 1 ? value * 100 : value;
        
        const data = [{
            type: 'indicator',
            mode: 'gauge+number+delta',
            value: percentage,
            title: { 
                text: title, 
                font: { size: 16, color: COLORS.text }
            },
            delta: { reference: 80, increasing: { color: COLORS.chart.accuracy } },
            gauge: {
                axis: { 
                    range: [0, 100], 
                    tickwidth: 1, 
                    tickcolor: COLORS.text,
                    tickfont: { size: 10 }
                },
                bar: { color: color, thickness: 0.3 },
                bgcolor: 'rgba(255,255,255,0.1)',
                borderwidth: 2,
                bordercolor: 'rgba(255,255,255,0.2)',
                steps: [
                    { range: [0, 50], color: 'rgba(239, 68, 68, 0.2)' },
                    { range: [50, 75], color: 'rgba(245, 158, 11, 0.2)' },
                    { range: [75, 90], color: 'rgba(16, 185, 129, 0.2)' },
                    { range: [90, 100], color: 'rgba(16, 185, 129, 0.4)' }
                ],
                threshold: {
                    line: { color: color, width: 4 },
                    thickness: 0.75,
                    value: 90
                }
            }
        }];
        
        const layout = {
            ...MODERN_LAYOUT,
            height: 250,
            margin: { t: 40, b: 20, l: 20, r: 20 }
        };
        
        Plotly.newPlot(containerId, data, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error(`Error creating performance gauge for ${title}:`, error);
        handlePlotError(containerId, 'Error loading performance gauge');
    }
}

// Create real-time metrics dashboard
function createMetricsDashboard(containerId, metricsData) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Create subplots for different metrics
        const traces = [];
        const colors = [COLORS.chart.accuracy, COLORS.chart.f1_score, COLORS.chart.precision, COLORS.chart.recall];
        const metrics = ['accuracy', 'f1_score', 'precision', 'recall'];
        
        metrics.forEach((metric, index) => {
            if (metricsData.global && metricsData.global.length > 0) {
                const rounds = metricsData.global.map((_, i) => i + 1);
                const values = metricsData.global.map(m => (m[metric] || 0) * 100);
                
                traces.push({
                    x: rounds,
                    y: values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: metric.replace('_', ' ').toUpperCase(),
                    line: {
                        color: colors[index],
                        width: 3,
                        shape: 'spline'
                    },
                    marker: {
                        size: 8,
                        color: colors[index],
                        line: { color: 'white', width: 2 }
                    },
                    fill: 'tonexty',
                    fillcolor: colors[index] + '20'
                });
            }
        });
        
        const layout = {
            ...MODERN_LAYOUT,
            title: {
                text: 'Real-time Performance Metrics',
                font: { size: 18, color: COLORS.text }
            },
            xaxis: {
                ...MODERN_LAYOUT.xaxis,
                title: 'Training Round'
            },
            yaxis: {
                ...MODERN_LAYOUT.yaxis,
                title: 'Performance (%)',
                range: [0, 100]
            },
            height: 400,
            hovermode: 'x unified'
        };
        
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
    } catch (error) {
        console.error('Error creating metrics dashboard:', error);
        handlePlotError(containerId, 'Failed to create metrics dashboard');
    }
}

// Create animated training progress visualization
function createAnimatedTrainingProgress(containerId, progressData) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        if (!progressData || !progressData.global || progressData.global.length === 0) {
            container.innerHTML = `
                <div class="plot-warning">
                    <i class="fas fa-chart-line fa-3x mb-3"></i>
                    <h5>No Training Data</h5>
                    <p>Start training to see animated progress visualization.</p>
                </div>
            `;
            return;
        }
        
        const rounds = progressData.global.map((_, i) => i + 1);
        const accuracy = progressData.global.map(m => (m.accuracy || 0) * 100);
        const loss = progressData.global.map(m => m.loss || 0);
        
        const traces = [
            {
                x: rounds,
                y: accuracy,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Accuracy (%)',
                line: {
                    color: COLORS.chart.accuracy,
                    width: 4,
                    shape: 'spline'
                },
                marker: {
                    size: 10,
                    color: COLORS.chart.accuracy,
                    line: { color: 'white', width: 2 }
                },
                yaxis: 'y'
            },
            {
                x: rounds,
                y: loss,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Loss',
                line: {
                    color: COLORS.chart.loss,
                    width: 4,
                    shape: 'spline'
                },
                marker: {
                    size: 10,
                    color: COLORS.chart.loss,
                    line: { color: 'white', width: 2 }
                },
                yaxis: 'y2'
            }
        ];
        
        const layout = {
            ...MODERN_LAYOUT,
            title: {
                text: 'Training Progress Animation',
                font: { size: 18, color: COLORS.text }
            },
            xaxis: {
                ...MODERN_LAYOUT.xaxis,
                title: 'Training Round'
            },
            yaxis: {
                ...MODERN_LAYOUT.yaxis,
                title: 'Accuracy (%)',
                side: 'left',
                range: [0, 100]
            },
            yaxis2: {
                title: 'Loss',
                overlaying: 'y',
                side: 'right',
                gridcolor: 'rgba(0,0,0,0)'
            },
            height: 400,
            hovermode: 'x unified'
        };
        
        Plotly.newPlot(containerId, traces, layout, PLOT_CONFIG);
        
        // Add animation frames
        const frames = rounds.map((round, i) => ({
            name: round.toString(),
            data: [
                {
                    x: rounds.slice(0, i + 1),
                    y: accuracy.slice(0, i + 1)
                },
                {
                    x: rounds.slice(0, i + 1),
                    y: loss.slice(0, i + 1)
                }
            ]
        }));
        
        Plotly.addFrames(containerId, frames);
        
    } catch (error) {
        console.error('Error creating animated training progress:', error);
        handlePlotError(containerId, 'Failed to create training progress animation');
    }
}

// Function to regenerate plots with enhanced functionality
function regeneratePlot(containerId) {
    const plotType = containerId.split('-')[0];
    
    switch(plotType) {
        case 'accuracy':
            // Reload accuracy comparison with latest data
            fetch('/api/metrics/history')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        createAccuracyComparisonChart(containerId, data);
                    }
                })
                .catch(error => console.error('Error reloading accuracy chart:', error));
            break;
            
        case 'confusion':
            createConfusionMatrix(containerId);
            break;
            
        case 'feature':
            createFeatureImportance(containerId);
            break;
            
        case 'resource':
            createResourceUsage(containerId);
            break;
            
        case 'architecture':
            createModelArchitecture(containerId);
            break;
            
        default:
            // Default regeneration
            loadAllPlots();
            break;
    }
}

// Load all plots function
function loadAllPlots() {
    // Load metrics history and create charts
    fetch('/api/metrics/history')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                createAccuracyComparisonChart('accuracy-chart', data);
                createF1ScoreComparisonChart('f1-chart', data);
                createAnimatedTrainingProgress('training-chart', data);
                createMetricsDashboard('metrics-dashboard', data);
            }
        })
        .catch(error => console.error('Error loading plots:', error));
    
    // Load latest metrics for gauges and radar
    fetch('/api/metrics/latest')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                createModelComparisonRadar('radar-chart', data);
                
                // Create performance gauges
                if (data.global) {
                    createPerformanceGauge('accuracy-gauge', data.global.accuracy || 0, 'Global Accuracy');
                    createPerformanceGauge('f1-gauge', data.global.f1_score || 0, 'Global F1 Score');
                }
            }
        })
        .catch(error => console.error('Error loading latest metrics:', error));
    
    // Load advanced visualizations
    createConfusionMatrix('confusion-matrix');
    createFeatureImportance('feature-importance');
    createResourceUsage('resource-usage');
    createModelArchitecture('model-architecture');
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
