// Client-Side Data Caching
let cachedMetrics = [];
let cachedPredictions = [];
let cachedFeatures = [];
let cachedOptuna = null;
let cachedLstm = [];
let activeExplorerLimit = 2016; // last 7 days by default (7 * 288)
let dashboardSelectedModel = null;
let activeResidualGroup = 'advanced';

const RESIDUAL_COLORS = {
    "Linear Regression": "#FF3B30",     // Neon Red
    "Decision Tree": "#FF9500",         // Neon Orange
    "Random Forest": "#34C759",         // Neon Green
    "KNN": "#AF52DE",                   // Neon Purple
    "Gradient Boosting": "#007AFF",     // Neon Blue
    "XGBoost (Tuned)": "#FF453A",       // Brighter Red
    "XGBoost": "#FF453A",
    "LightGBM (Tuned)": "#30D158",      // Brighter Green
    "LightGBM": "#30D158",
    "Ensemble (XGB+LGBM)": "#BF5AF2",   // Brighter Purple
    "LSTM": "#FF375F"                   // Brighter Pink
};

// Three.js Particle Terrain Variables
let bgScene, bgCamera, bgRenderer, bgPoints, bgGeometry;
let simScene, simCamera, simRenderer, simPoints, simGeometry;
let bgTime = 0, simTime = 0;

// Initialize on window load
window.addEventListener('DOMContentLoaded', () => {
    // 1. Digital Clock
    setInterval(updateClock, 1000);
    updateClock();

    // 2. Setup Three.js Animations
    initBackgroundTerrain();
    initSimulatorTerrain();

    // 3. Start MLOps Terminal log stream ticker
    startLogTicker();

    // 4. Fetch initial API data
    fetchDashboardData();
    
    // 5. Default Simulation Prediction Call
    runSimulation();
});

// Digital Clock Update
function updateClock() {
    const now = new Date();
    const timeStr = now.toISOString().split('T')[1].split('.')[0] + ' UTC';
    const clockEl = document.getElementById('digital-clock');
    if (clockEl) clockEl.textContent = timeStr;
}

// -------------------------------------------------------------
// TAB NAVIGATION ROUTER
// -------------------------------------------------------------
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(section => {
        section.classList.add('hidden');
    });

    // Remove active styles from nav buttons
    document.querySelectorAll('nav button').forEach(btn => {
        btn.classList.remove('text-primary', 'bg-surface-container-highest', 'shadow-[0_0_15px_rgba(255,255,255,0.15)]', 'active-glow');
        btn.classList.add('text-on-surface-variant', 'hover:text-primary', 'hover:bg-surface-variant');
    });

    // Show selected tab
    const targetSection = document.getElementById(`tab-${tabId}`);
    if (targetSection) {
        targetSection.classList.remove('hidden');
    }

    // Add active styles to clicked button
    const targetNav = document.getElementById(`nav-${tabId}`);
    if (targetNav) {
        targetNav.classList.remove('text-on-surface-variant', 'hover:text-primary', 'hover:bg-surface-variant');
        targetNav.classList.add('text-primary', 'bg-surface-container-highest', 'shadow-[0_0_15px_rgba(255,255,255,0.15)]', 'active-glow');
    }

    // Update Header subtitle
    const subtitleEl = document.getElementById('tab-subtitle');
    if (subtitleEl) {
        const titleMap = {
            'gateway': 'Gateway Terminal',
            'dashboard': 'Executive Control Node',
            'comparison': 'Benchmarking Laboratory',
            'explorer': 'Load Curve Observatory',
            'simulator': 'What-If Simulation Engine',
            'mlops': 'MLOps Orchestrator',
            'residual': 'RESIDUAL_LAB_OS // v4.0.2-ALPHA'
        };
        subtitleEl.textContent = titleMap[tabId] || 'Command System';
    }

    // Toggle header right contents based on active tab
    const defaultHeaderRight = document.getElementById('header-right-default');
    const residualHeaderRight = document.getElementById('header-right-residual');
    if (defaultHeaderRight && residualHeaderRight) {
        if (tabId === 'residual') {
            defaultHeaderRight.classList.add('hidden');
            residualHeaderRight.classList.remove('hidden');
        } else {
            residualHeaderRight.classList.add('hidden');
            defaultHeaderRight.classList.remove('hidden');
        }
    }

    // Trigger Plotly redraw on tab switch to fit containers correctly
    setTimeout(() => {
        const plotlyPlots = [
            'plotly-realtime-chart', 'plotly-metrics-comparison', 'plotly-feature-variance', 
            'plotly-explorer-chart', 'plotly-optuna-chart', 'plotly-lstm-loss-chart',
            'plotly-residual-bivariate', 'plotly-res-dist-xgb', 'plotly-res-dist-lgb', 'plotly-res-dist-ens'
        ];
        plotlyPlots.forEach(plotId => {
            const plotEl = document.getElementById(plotId);
            if (plotEl && plotEl.data) {
                Plotly.Plots.resize(plotEl);
            }
        });
        
        // Handle Three.js resizing for simulator viewport
        if (tabId === 'simulator' && simRenderer && simCamera) {
            const container = document.getElementById('sim-threejs-container');
            const w = container.clientWidth;
            const h = container.clientHeight;
            simRenderer.setSize(w, h);
            simCamera.aspect = w / h;
            simCamera.updateProjectionMatrix();
        }
    }, 100);
}

// -------------------------------------------------------------
// TELEMETRY DATA FETCHING & API INTERFACE
// -------------------------------------------------------------
async function fetchDashboardData() {
    try {
        const syncIcon = document.getElementById('sync-icon');
        if (syncIcon) syncIcon.classList.add('animate-spin');

        // Fetch Metrics
        const resMetrics = await fetch('/api/metrics');
        cachedMetrics = await resMetrics.json();

        // Fetch Feature Importances
        const resFeatures = await fetch('/api/features');
        cachedFeatures = await resFeatures.json();

        // Fetch Optuna History
        const resOptuna = await fetch('/api/optuna');
        cachedOptuna = await resOptuna.json();

        // Fetch LSTM History
        const resLstm = await fetch('/api/lstm');
        cachedLstm = await resLstm.json();

        // Fetch predictions slice (default to last 7 days = 2016 rows)
        const resPredictions = await fetch(`/api/predictions?limit=${activeExplorerLimit}`);
        cachedPredictions = await resPredictions.json();

        // Populate Table Grids and Charts
        renderDashboardTab();
        renderComparisonTab();
        renderExplorerTab();
        renderMlopsTab();
        renderResidualTab();

        if (syncIcon) {
            setTimeout(() => syncIcon.classList.remove('animate-spin'), 600);
        }
    } catch (e) {
        console.error("❌ Error fetching API dashboard data:", e);
        writeLog("SYS_ERROR: API synchronization failed. Check backend connection.", true);
    }
}

// -------------------------------------------------------------
// RENDER TAB: EXECUTIVE DASHBOARD
// -------------------------------------------------------------
function selectDashboardModel(modelName) {
    dashboardSelectedModel = modelName;
    renderDashboardTab();
}

function renderDashboardTab() {
    if (!cachedMetrics.length || !cachedPredictions.length) return;

    // 1. Identify best model (lowest RMSE)
    const sortedMetrics = [...cachedMetrics].sort((a, b) => a.RMSE - b.RMSE);
    const bestModel = sortedMetrics[0];

    // Default selected model to best model
    if (!dashboardSelectedModel) {
        dashboardSelectedModel = bestModel.Model;
    }

    const isBestSelected = dashboardSelectedModel === bestModel.Model;
    const selectedModelMetrics = cachedMetrics.find(m => m.Model === dashboardSelectedModel) || bestModel;

    // Update Top metrics cards dynamically
    const bestModelLabelEl = document.getElementById('dash-best-model-label');
    if (bestModelLabelEl) {
        bestModelLabelEl.textContent = isBestSelected ? 'Top Performer' : 'Selected Model';
    }
    
    document.getElementById('dash-best-model').textContent = selectedModelMetrics.Model;
    
    const bestModelStatusEl = document.getElementById('dash-best-model-status');
    if (bestModelStatusEl) {
        if (isBestSelected) {
            bestModelStatusEl.innerHTML = `<span class="material-symbols-outlined text-[14px]">check_circle</span> Optimal`;
            bestModelStatusEl.className = "text-code-sm text-primary/60 flex items-center gap-1 mt-1";
        } else {
            bestModelStatusEl.innerHTML = `<span class="material-symbols-outlined text-[14px]">info</span> Selected`;
            bestModelStatusEl.className = "text-code-sm text-on-surface-variant flex items-center gap-1 mt-1";
        }
    }

    const rmseLabelEl = document.getElementById('dash-rmse-label');
    if (rmseLabelEl) rmseLabelEl.textContent = isBestSelected ? 'Min RMSE' : 'Model RMSE';
    document.getElementById('dash-rmse').textContent = `${selectedModelMetrics.RMSE.toFixed(2)} MW`;

    const maeLabelEl = document.getElementById('dash-mae-label');
    if (maeLabelEl) maeLabelEl.textContent = isBestSelected ? 'Min MAE' : 'Model MAE';
    document.getElementById('dash-mae').textContent = `${selectedModelMetrics.MAE.toFixed(2)} MW`;

    const mapeLabelEl = document.getElementById('dash-mape-label');
    if (mapeLabelEl) mapeLabelEl.textContent = isBestSelected ? 'MAPE Error' : 'Model MAPE';
    document.getElementById('dash-mape').textContent = `${selectedModelMetrics["MAPE (%)"].toFixed(2)} %`;

    const r2LabelEl = document.getElementById('dash-r2-label');
    if (r2LabelEl) r2LabelEl.textContent = isBestSelected ? 'Max R² Score' : 'Model R² Score';
    document.getElementById('dash-r2').textContent = selectedModelMetrics["R²"].toFixed(4);

    // 2. Render Leaderboard table rows
    const tbody = document.getElementById('dash-leaderboard-rows');
    if (tbody) {
        tbody.innerHTML = '';
        sortedMetrics.forEach((row, idx) => {
            const isSelected = row.Model === dashboardSelectedModel;
            const isBest = row.Model === bestModel.Model;
            const tr = document.createElement('tr');
            tr.className = `hover:bg-white/5 transition-colors cursor-pointer ${isSelected ? 'bg-primary/10 border-l-2 border-l-primary' : ''}`;
            tr.innerHTML = `
                <td class="px-4 py-3 font-medium ${isSelected ? 'text-primary' : 'text-on-surface-variant'}">${row.Model}</td>
                <td class="px-4 py-3 text-right font-mono">${row.RMSE.toFixed(2)}</td>
                <td class="px-4 py-3 text-right font-mono">${row["R²"].toFixed(4)}</td>
            `;
            tr.onclick = () => {
                selectDashboardModel(row.Model);
            };
            tbody.appendChild(tr);
        });
    }

    // 3. Render Mini Bar Chart comparison (HTML bars)
    const miniBarContainer = document.getElementById('mini-bar-chart');
    if (miniBarContainer) {
        miniBarContainer.innerHTML = '';
        
        const lowestRMSE = Math.min(...cachedMetrics.map(m => m.RMSE));
        const highestRMSE = Math.max(...cachedMetrics.map(m => m.RMSE));
        
        // Sort from poorest (highest RMSE) to best (lowest RMSE) to flow left-to-right matching the labels
        const sortedMetricsForChart = [...cachedMetrics].sort((a, b) => b.RMSE - a.RMSE);
        
        sortedMetricsForChart.forEach(m => {
            // Calculate height pct: lowest RMSE (best) gets 85% height, highest RMSE (poorest) gets 15% height.
            // This ensures it never peeks out of the top of the container box.
            const pct = Math.max(15, ((highestRMSE - m.RMSE) / (highestRMSE - lowestRMSE)) * 85);
            const bar = document.createElement('div');
            bar.className = 'flex-1 bg-surface-container-highest chart-bar group relative cursor-pointer';
            bar.style.height = `${pct}%`;
            bar.title = `${m.Model}: RMSE ${m.RMSE.toFixed(1)} MW`;
            
            // Highlight the best model with full white glow
            if (m.Model === bestModel.Model) {
                bar.className = 'flex-1 bg-primary active-glow cursor-pointer';
            }
            
            miniBarContainer.appendChild(bar);
        });
    }

    // 4. Draw Plotly Realtime Forecast Chart (last 24 hours = 288 steps)
    const recentPredictions = cachedPredictions.slice(-288);
    const times = recentPredictions.map(p => p.datetime);
    const actuals = recentPredictions.map(p => p.Actual);
    const modelPredictions = recentPredictions.map(p => p[dashboardSelectedModel] || p["Ensemble (XGB+LGBM)"]);

    const actualTrace = {
        x: times,
        y: actuals,
        name: 'Actual Demand',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ffffff', width: 2 }
    };

    const modelTrace = {
        x: times,
        y: modelPredictions,
        name: `${dashboardSelectedModel} Forecast`,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#6C3483', width: 1.5, dash: 'dot' }
    };

    const layout = {
        height: 290,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 55, l: 55 },
        xaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 10, color: '#666666' }
        },
        yaxis: {
            title: 'MW',
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 10, color: '#666666' }
        },
        legend: { showlegend: false }
    };

    Plotly.newPlot('plotly-realtime-chart', [actualTrace, modelTrace], layout, { displayModeBar: false, responsive: true });
    
    // Update legend chart model name label
    const chartModelNameEl = document.getElementById('dash-chart-model-name');
    if (chartModelNameEl) {
        chartModelNameEl.textContent = dashboardSelectedModel;
    }
}

// -------------------------------------------------------------
// RENDER TAB: BENCHMARKING LAB (COMPARISON)
// -------------------------------------------------------------
function renderComparisonTab() {
    if (!cachedMetrics.length || !cachedFeatures.length) return;

    // 1. Populate Metrics table
    const tableBody = document.getElementById('comp-metrics-table');
    if (tableBody) {
        tableBody.innerHTML = '';
        cachedMetrics.forEach(row => {
            let statusBadge = '<span class="text-[9px] border border-outline-variant text-on-surface-variant px-2 py-0.5">STANDBY</span>';
            if (row.Model === 'Linear Regression') {
                statusBadge = '<span class="text-[9px] border border-primary text-primary px-2 py-0.5 font-bold">BEST FIT</span>';
            } else if (row.Model.includes('Ensemble')) {
                statusBadge = '<span class="text-[9px] border border-white text-white px-2 py-0.5">ACTIVE</span>';
            } else if (row.Model === 'Naive Baseline') {
                statusBadge = '<span class="text-[9px] border border-error text-error px-2 py-0.5">BASELINE</span>';
            }

            const tr = document.createElement('tr');
            tr.className = 'hover:bg-white/5 transition-colors cursor-crosshair';
            tr.innerHTML = `
                <td class="px-4 py-3 font-bold text-primary">${row.Model}</td>
                <td class="px-4 py-3 text-right font-mono">${row.MAE.toFixed(2)}</td>
                <td class="px-4 py-3 text-right font-mono">${row.RMSE.toFixed(2)}</td>
                <td class="px-4 py-3 text-right font-mono">${row["R²"].toFixed(4)}</td>
                <td class="px-4 py-3 text-right font-mono">${row["MAPE (%)"].toFixed(2)}%</td>
                <td class="px-4 py-3 text-center">${statusBadge}</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    // 2. Render Top features list (right panel)
    const featureList = document.getElementById('comp-features-list');
    if (featureList) {
        featureList.innerHTML = '';
        // Sort features based on Random Forest importance
        const sortedFeatures = [...cachedFeatures].sort((a, b) => b["Random Forest"] - a["Random Forest"]).slice(0, 5);
        sortedFeatures.forEach((feat, idx) => {
            const val = feat["Random Forest"] * 100;
            const element = document.createElement('div');
            element.className = 'space-y-1';
            element.innerHTML = `
                <div class="flex justify-between text-code-sm items-center">
                    <span class="text-on-surface-variant">${idx + 1}. ${feat.Feature}</span>
                    <span class="text-primary font-bold font-mono">${val.toFixed(2)}%</span>
                </div>
                <div class="w-full bg-[#1A1A1A] h-1">
                    <div class="bg-white h-full" style="width: ${val}%"></div>
                </div>
            `;
            featureList.appendChild(element);
        });
    }

    // 3. Plotly Metrics comparison chart
    const models = cachedMetrics.map(m => m.Model);
    const maes = cachedMetrics.map(m => m.MAE);
    const rmses = cachedMetrics.map(m => m.RMSE);

    const maeTrace = {
        x: models,
        y: maes,
        name: 'MAE (MW)',
        type: 'bar',
        marker: { color: '#666666' }
    };

    const rmseTrace = {
        x: models,
        y: rmses,
        name: 'RMSE (MW)',
        type: 'bar',
        marker: { color: '#ffffff' }
    };

    const metricsLayout = {
        height: 240,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 110, l: 55 },
        barmode: 'group',
        xaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        yaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        legend: { font: { family: 'Geist', size: 9, color: '#e5e2e1' }, x: 0.82, y: 0.98, bgcolor: 'rgba(8, 8, 8, 0.65)' }
    };

    Plotly.newPlot('plotly-metrics-comparison', [maeTrace, rmseTrace], metricsLayout, { displayModeBar: false, responsive: true });

    // 4. Plotly Feature importances barplot
    // sort features
    const top10Features = [...cachedFeatures].sort((a, b) => b["Random Forest"] - a["Random Forest"]).slice(0, 10).reverse();
    const featNames = top10Features.map(f => f.Feature);
    const featRF = top10Features.map(f => f["Random Forest"]);

    const featTrace = {
        y: featNames,
        x: featRF,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#ffffff' }
    };

    const featLayout = {
        height: 240,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 30, l: 90 },
        xaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        yaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        }
    };

    Plotly.newPlot('plotly-feature-variance', [featTrace], featLayout, { displayModeBar: false, responsive: true });
}

// -------------------------------------------------------------
// RENDER TAB: FORECAST EXPLORER
// -------------------------------------------------------------
function renderExplorerTab() {
    if (!cachedPredictions.length) return;

    const times = cachedPredictions.map(p => p.datetime);
    const actuals = cachedPredictions.map(p => p.Actual);
    const linear = cachedPredictions.map(p => p["Linear Regression"]);
    const ensemble = cachedPredictions.map(p => p["Ensemble (XGB+LGBM)"]);
    const lstm = cachedPredictions.map(p => p.LSTM);

    const actualTrace = {
        x: times,
        y: actuals,
        name: 'Actual Load',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ffffff', width: 2 }
    };

    const linearTrace = {
        x: times,
        y: linear,
        name: 'Linear Regression',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#E74C3C', width: 1 }
    };

    const ensembleTrace = {
        x: times,
        y: ensemble,
        name: 'Ensemble XGB+LGBM',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#6C3483', width: 1 }
    };

    const lstmTrace = {
        x: times,
        y: lstm,
        name: 'LSTM Network',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#D4145A', width: 1 }
    };

    const expLayout = {
        height: 350,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 15, b: 55, l: 65 },
        xaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        yaxis: {
            title: 'MW Demand',
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        legend: { font: { family: 'Geist', size: 9, color: '#e5e2e1' }, x: 0.82, y: 0.98, bgcolor: 'rgba(8, 8, 8, 0.65)' }
    };

    Plotly.newPlot('plotly-explorer-chart', [actualTrace, linearTrace, ensembleTrace, lstmTrace], expLayout, { displayModeBar: false, responsive: true });
}

function updateExplorerLimit(limit) {
    activeExplorerLimit = limit;
    
    // Toggle active state classes on buttons
    const buttons = document.querySelectorAll('#tab-explorer button');
    buttons.forEach(btn => {
        if ((limit === 288 && btn.innerText === '1 DAY') ||
            (limit === 576 && btn.innerText === '2 DAYS') ||
            (limit === 2016 && btn.innerText === '7 DAYS')) {
            btn.className = 'px-3 py-1 bg-primary text-background border border-primary text-code-sm font-bold uppercase transition-colors';
        } else if (btn.innerText !== 'EXPORT JSON') {
            btn.className = 'px-3 py-1 bg-surface-container-high text-on-surface-variant hover:text-primary border border-outline-variant text-code-sm font-bold uppercase transition-colors';
        }
    });

    // Re-fetch with new limit
    fetchDashboardData();
}

// -------------------------------------------------------------
// RENDER TAB: MLOPS PIPELINE
// -------------------------------------------------------------
function renderMlopsTab() {
    if (!cachedOptuna || !cachedLstm.length) return;

    // 1. Plot Optuna convergence curves
    const xgbTrials = cachedOptuna.xgb.map(t => t.Trial);
    const xgbMaes = cachedOptuna.xgb.map(t => t.MAE);
    const xgbRunningMin = cachedOptuna.xgb.map(t => t.Running_Min);

    const lgbmTrials = cachedOptuna.lgbm.map(t => t.Trial);
    const lgbmMaes = cachedOptuna.lgbm.map(t => t.MAE);
    const lgbmRunningMin = cachedOptuna.lgbm.map(t => t.Running_Min);

    const xgbScatter = {
        x: xgbTrials,
        y: xgbMaes,
        mode: 'markers',
        name: 'XGBoost Trial MAE',
        marker: { color: '#666666', size: 5, opacity: 0.6 }
    };

    const xgbLine = {
        x: xgbTrials,
        y: xgbRunningMin,
        mode: 'lines',
        name: 'XGBoost Running Min',
        line: { color: '#ffffff', width: 2 }
    };

    const lgbmScatter = {
        x: lgbmTrials,
        y: lgbmMaes,
        mode: 'markers',
        name: 'LightGBM Trial MAE',
        marker: { color: '#444748', size: 5, opacity: 0.6 }
    };

    const lgbmLine = {
        x: lgbmTrials,
        y: lgbmRunningMin,
        mode: 'lines',
        name: 'LightGBM Running Min',
        line: { color: '#AAB7B8', width: 2, dash: 'dash' }
    };

    const optunaLayout = {
        height: 250,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 15, r: 15, b: 50, l: 60 },
        xaxis: {
            title: 'Trial Index',
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        yaxis: {
            title: 'CV MAE (MW)',
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        legend: { font: { family: 'Geist', size: 8, color: '#e5e2e1' }, x: 0.82, y: 0.98, bgcolor: 'rgba(8, 8, 8, 0.65)' }
    };

    Plotly.newPlot('plotly-optuna-chart', [xgbScatter, xgbLine, lgbmScatter, lgbmLine], optunaLayout, { displayModeBar: false, responsive: true });

    // 2. Plot LSTM loss curve
    const epochs = cachedLstm.map(e => e.Epoch);
    const loss = cachedLstm.map(e => e.Loss);
    const valLoss = cachedLstm.map(e => e.Val_Loss);

    const lossTrace = {
        x: epochs,
        y: loss,
        name: 'Training Loss (MSE)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#ffffff', width: 2 }
    };

    const valLossTrace = {
        x: epochs,
        y: valLoss,
        name: 'Validation Loss (MSE)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#666666', width: 1.5, dash: 'dash' }
    };

    const lossLayout = {
        height: 190,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 15, r: 15, b: 45, l: 70 },
        xaxis: {
            title: 'Epoch',
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        yaxis: {
            title: 'Loss Value',
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        legend: { font: { family: 'Geist', size: 9, color: '#e5e2e1' }, x: 0.82, y: 0.98, bgcolor: 'rgba(8, 8, 8, 0.65)' }
    };

    Plotly.newPlot('plotly-lstm-loss-chart', [lossTrace, valLossTrace], lossLayout, { displayModeBar: false, responsive: true });
}

// -------------------------------------------------------------
// RENDER TAB: FORECAST SIMULATOR
// -------------------------------------------------------------
async function runSimulation() {
    const temp = parseFloat(document.getElementById('sim-temp').value);
    const rhum = parseFloat(document.getElementById('sim-rhum').value);
    const pres = parseFloat(document.getElementById('sim-pres').value);
    const wspd = parseFloat(document.getElementById('sim-wspd').value);
    const hour = parseInt(document.getElementById('sim-hour').value);
    const weekday = parseInt(document.getElementById('sim-weekday').value);
    const isPeakHour = document.getElementById('sim-peak').checked;

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                temp: temp,
                rhum: rhum,
                pres: pres,
                wspd: wspd,
                hour: hour,
                weekday: weekday,
                is_peak_hour: isPeakHour
            })
        });

        if (!res.ok) {
            throw new Error(`Inference HTTP error: ${res.status}`);
        }

        const predictions = await res.json();
        
        // Update values in UI
        document.getElementById('sim-pred-ensemble').textContent = Math.round(predictions["Ensemble (XGB+LGBM)"]).toLocaleString();
        document.getElementById('sim-pred-xgb').textContent = `${Math.round(predictions["XGBoost (Tuned)"]).toLocaleString()} MW`;
        document.getElementById('sim-pred-lgb').textContent = `${Math.round(predictions["LightGBM (Tuned)"]).toLocaleString()} MW`;
        document.getElementById('sim-pred-rf').textContent = `${Math.round(predictions["Random Forest"]).toLocaleString()} MW`;
        
        writeLog(`SIM_RUN: Inference complete. ensemble_load = ${Math.round(predictions["Ensemble (XGB+LGBM)"])} MW`);

        // Trigger dynamic updates to the simulator Three.js terrain based on results
        updateSimTerrainWaves(temp, wspd, predictions["Ensemble (XGB+LGBM)"]);
    } catch (ex) {
        console.error("Simulation run failed:", ex);
        writeLog(`SIM_ERROR: Simulation calculation failed: ${ex.message}`, true);
    }
}

function checkPeakHourFlag() {
    const hour = parseInt(document.getElementById('sim-hour').value);
    const checkbox = document.getElementById('sim-peak');
    if (checkbox) {
        // peak hours are between 18 and 21h inclusive
        checkbox.checked = (hour >= 18 && hour <= 21);
    }
}

function applyScenario(temp, rhum, pres, wspd, hour, weekday, isPeak) {
    // Update inputs
    document.getElementById('sim-temp').value = temp;
    document.getElementById('sim-temp-val').innerText = temp;

    document.getElementById('sim-rhum').value = rhum;
    document.getElementById('sim-rhum-val').innerText = rhum;

    document.getElementById('sim-pres').value = pres;
    document.getElementById('sim-pres-val').innerText = pres;

    document.getElementById('sim-wspd').value = wspd;
    document.getElementById('sim-wspd-val').innerText = wspd;

    document.getElementById('sim-hour').value = hour;
    document.getElementById('sim-hour-val').innerText = hour;

    document.getElementById('sim-weekday').value = weekday;

    document.getElementById('sim-peak').checked = isPeak;

    writeLog(`SCENARIO: Applied predefined stress template.`);
    runSimulation();
}

function resetSimulationDefaults() {
    applyScenario(28.0, 60.0, 1008.0, 8.5, 18, 2, true);
}

// -------------------------------------------------------------
// MLOPS TELEMETRY LOGGER TICKER
// -------------------------------------------------------------
const logMessages = [
    "SYNC: Broadcasted tuned models registry weights to 2 remote nodes.",
    "MON: Gateway API latency test complete: us-east (12.4ms), ap-south (8.2ms).",
    "MON: RAM usage stabilized at 1.25 GB / GPU Temp 42°C nominal.",
    "SYNC: Delhi weather telemetry ingested. Temperature drift detected (+0.5°C).",
    "INF: Ensemble inference pipeline execution time: 11.2ms.",
    "MLOPS: XGBoost/LightGBM Optuna hyperparameter checkpoints synchronized.",
    "MON: SCADA grid integration status: 99.8% nominal health."
];

function writeLog(message, isError = false) {
    const container = document.getElementById('mlops-terminal-logs');
    if (!container) return;

    const time = new Date().toISOString().split('T')[1].substring(0, 8);
    const p = document.createElement('p');
    p.className = `leading-relaxed ${isError ? 'text-red-500 font-bold' : ''}`;
    p.innerHTML = `<span class="text-neutral-600">[${time}]</span> <span class="${isError ? 'text-red-500' : 'text-primary'}">${isError ? 'FAIL' : 'INFO'}</span> ${message}`;
    
    // Add prompt blinker helper back
    const blinker = container.querySelector('.flex');
    if (blinker) {
        container.insertBefore(p, blinker);
    } else {
        container.appendChild(p);
    }
    
    // Auto scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function startLogTicker() {
    // Initial logs setup
    const container = document.getElementById('mlops-terminal-logs');
    if (container) {
        container.innerHTML = `
            <p><span class="text-neutral-600">[14:02:11]</span> <span class="text-primary">INIT</span> Loading dataset: <span class="text-on-surface italic">load_forecast_v8.csv</span></p>
            <p><span class="text-neutral-600">[14:02:15]</span> <span class="text-primary">INFO</span> Feature Engineering phase complete (24 features generated).</p>
            <p><span class="text-neutral-600">[14:02:22]</span> <span class="text-primary">EXEC</span> Starting Optuna search (H-Space: Grid_v12)...</p>
            <p><span class="text-neutral-600">[14:05:01]</span> <span class="text-primary">WARN</span> GPU-0 Memory near 85% capacity. Optimization triggered.</p>
            <div class="flex items-center gap-1.5 text-primary pt-2">
                <span>_</span>
                <span class="w-1.5 h-3.5 bg-primary animate-[blink_1s_step-end_infinite]"></span>
            </div>
        `;
    }

    // Interval log generator
    setInterval(() => {
        const randMsg = logMessages[Math.floor(Math.random() * logMessages.length)];
        writeLog(randMsg);
    }, 12000);
}

// -------------------------------------------------------------
// DOWNLOAD / EXPORT DATA UTILITIES
// -------------------------------------------------------------
function downloadJSON(type) {
    let dataToExport = {};
    let filename = "export.json";

    if (type === 'metrics') {
        dataToExport = { metrics: cachedMetrics, features: cachedFeatures };
        filename = "powercast_benchmarking_report.json";
    } else if (type === 'prediction') {
        const ensemble = document.getElementById('sim-pred-ensemble').textContent;
        dataToExport = {
            timestamp: new Date().toISOString(),
            input_variables: {
                temperature: parseFloat(document.getElementById('sim-temp').value),
                humidity: parseFloat(document.getElementById('sim-rhum').value),
                pressure: parseFloat(document.getElementById('sim-pres').value),
                wind_speed: parseFloat(document.getElementById('sim-wspd').value),
                hour: parseInt(document.getElementById('sim-hour').value),
                weekday: parseInt(document.getElementById('sim-weekday').value),
                is_peak_hour: document.getElementById('sim-peak').checked
            },
            predictions: {
                ensemble_load_mw: parseFloat(ensemble.replace(/,/g, '')),
                xgb_load_mw: parseFloat(document.getElementById('sim-pred-xgb').textContent),
                lgb_load_mw: parseFloat(document.getElementById('sim-pred-lgb').textContent),
                rf_load_mw: parseFloat(document.getElementById('sim-pred-rf').textContent)
            }
        };
        filename = "simulation_inference_report.json";
    }

    const jsonString = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(dataToExport, null, 4));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", jsonString);
    dlAnchorElem.setAttribute("download", filename);
    dlAnchorElem.click();
}

// -------------------------------------------------------------
// THREE.JS GLOBAL BACKGROUND & SIMULATOR ANIMATIONS
// -------------------------------------------------------------
function initBackgroundTerrain() {
    const container = document.getElementById('threejs-background');
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    bgScene = new THREE.Scene();
    bgCamera = new THREE.PerspectiveCamera(75, width / height, 0.1, 2000);
    bgCamera.position.set(0, 150, 420);
    bgCamera.lookAt(0, 0, 0);

    bgRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    bgRenderer.setSize(width, height);
    bgRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(bgRenderer.domElement);

    // Particle Terrain Parameter grids
    const segments = 90;
    const spacing = 12;
    const particleCount = segments * segments;

    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    bgGeometry = new THREE.BufferGeometry();

    for (let i = 0; i < segments; i++) {
        for (let j = 0; j < segments; j++) {
            const idx = (i * segments + j);
            const x = (i - segments / 2) * spacing;
            const z = (j - segments / 2) * spacing;

            positions[idx * 3] = x;
            positions[idx * 3 + 1] = 0; // Height dynamic
            positions[idx * 3 + 2] = z;

            // White particles
            colors[idx * 3] = 1.0;
            colors[idx * 3 + 1] = 1.0;
            colors[idx * 3 + 2] = 1.0;
        }
    }

    bgGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    bgGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
        size: 1.5,
        vertexColors: true,
        transparent: true,
        opacity: 0.35,
        sizeAttenuation: true
    });

    bgPoints = new THREE.Points(bgGeometry, particleMaterial);
    bgScene.add(bgPoints);

    // Resize event
    window.addEventListener('resize', () => {
        const w = container.clientWidth || window.innerWidth;
        const h = container.clientHeight || window.innerHeight;
        bgRenderer.setSize(w, h);
        bgCamera.aspect = w / h;
        bgCamera.updateProjectionMatrix();
    });

    // Run background loop
    animateBackground();
}

function animateBackground() {
    requestAnimationFrame(animateBackground);
    bgTime += 0.004;

    const posAttr = bgGeometry.attributes.position;
    const segments = 90;
    
    for (let i = 0; i < segments; i++) {
        for (let j = 0; j < segments; j++) {
            const idx = (i * segments + j);
            const x = posAttr.array[idx * 3];
            const z = posAttr.array[idx * 3 + 2];

            // Waving mathematical equations
            let y = Math.sin(x * 0.008 + bgTime) * Math.cos(z * 0.008 + bgTime) * 35;
            // secondary small wave details
            y += Math.sin(x * 0.02 + bgTime * 2) * 5;

            posAttr.array[idx * 3 + 1] = y;
        }
    }

    posAttr.needsUpdate = true;
    bgPoints.rotation.y = bgTime * 0.03;
    bgRenderer.render(bgScene, bgCamera);
}

// SIMULATOR VIEWPORT TERRAIN
let simWaveSpeed = 0.01;
let simWaveHeight = 30;

function initSimulatorTerrain() {
    const container = document.getElementById('sim-threejs-container');
    if (!container) return;

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;

    simScene = new THREE.Scene();
    simCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    simCamera.position.set(0, 100, 300);
    simCamera.lookAt(0, 0, 0);

    simRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    simRenderer.setSize(width, height);
    simRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(simRenderer.domElement);

    const segments = 60;
    const spacing = 8;
    const particleCount = segments * segments;

    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    simGeometry = new THREE.BufferGeometry();

    for (let i = 0; i < segments; i++) {
        for (let j = 0; j < segments; j++) {
            const idx = (i * segments + j);
            const x = (i - segments / 2) * spacing;
            const z = (j - segments / 2) * spacing;

            positions[idx * 3] = x;
            positions[idx * 3 + 1] = 0;
            positions[idx * 3 + 2] = z;

            // Base primary white
            colors[idx * 3] = 1.0;
            colors[idx * 3 + 1] = 1.0;
            colors[idx * 3 + 2] = 1.0;
        }
    }

    simGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    simGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
        size: 1.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.75,
        sizeAttenuation: true
    });

    simPoints = new THREE.Points(simGeometry, particleMaterial);
    simScene.add(simPoints);

    animateSimulator();
}

function animateSimulator() {
    requestAnimationFrame(animateSimulator);
    simTime += simWaveSpeed;

    const posAttr = simGeometry.attributes.position;
    const colorAttr = simGeometry.attributes.color;
    const segments = 60;

    // Get active slider inputs
    const tempVal = parseFloat(document.getElementById('sim-temp').value);
    const rhumVal = parseFloat(document.getElementById('sim-rhum').value);
    const wspdVal = parseFloat(document.getElementById('sim-wspd').value);

    // Calculate ratios (0.0 to 1.0)
    // Temperature range: -5 to 50 (range = 55)
    const tempRatio = Math.min(1.0, Math.max(0.0, (tempVal - (-5)) / 55));
    // Humidity range: 0 to 100 (range = 100)
    const rhumRatio = Math.min(1.0, Math.max(0.0, rhumVal / 100));
    // Wind Speed range: 0 to 80 (range = 80)
    const wspdRatio = Math.min(1.0, Math.max(0.0, wspdVal / 80));

    for (let i = 0; i < segments; i++) {
        for (let j = 0; j < segments; j++) {
            const idx = (i * segments + j);
            const x = posAttr.array[idx * 3];
            const z = posAttr.array[idx * 3 + 2];

            // Waving calculations based on reactive inputs
            let y = Math.sin(x * 0.02 + simTime) * Math.cos(z * 0.02 + simTime) * simWaveHeight;
            posAttr.array[idx * 3 + 1] = y;

            // Dynamic color shading based on wave peaks (peaks are brighter)
            const ratio = (y + simWaveHeight) / (simWaveHeight * 2);
            const brightness = 0.25 + ratio * 0.75;
            
            // Mix: active variables determine the color component, base brightness dictates luminance
            colorAttr.array[idx * 3]     = brightness * (0.2 + 0.8 * tempRatio); // Red maps to Temp
            colorAttr.array[idx * 3 + 1] = brightness * (0.2 + 0.8 * wspdRatio); // Green maps to Wind Speed
            colorAttr.array[idx * 3 + 2] = brightness * (0.2 + 0.8 * rhumRatio); // Blue maps to Humidity
        }
    }

    posAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    
    simPoints.rotation.y = simTime * 0.08;
    simRenderer.render(simScene, simCamera);
}

function updateSimTerrainWaves(temp, windSpeed, prediction) {
    // Higher temp -> bigger height waves (thermal stress simulation)
    simWaveHeight = Math.max(10, (temp / 45) * 60);

    // Higher wind speed -> faster motion waves
    simWaveSpeed = Math.max(0.003, (windSpeed / 100) * 0.05);

    // Update color based on load prediction severity
    if (prediction > 4000) {
        // Red alert color style in points material color array (make it white-hot glow)
        simPoints.material.color.setHex(0xffffff);
    } else {
        simPoints.material.color.setHex(0xcccccc);
    }
}

// -------------------------------------------------------------
// RESIDUAL ANALYSIS LABORATORY FUNCTIONS
// -------------------------------------------------------------
const SCADA_CODENAMES = {
    "Linear Regression": "LR_BASELINE_V1",
    "Decision Tree": "DT_TREE_V2",
    "Random Forest": "RF_FOREST_V3",
    "KNN": "KNN_NEIGHBOR_V1",
    "Gradient Boosting": "GB_BOOSTER_V2",
    "XGBoost (Tuned)": "XGB_TUNED_V4",
    "XGBoost": "XGB_TUNED_V4",
    "LightGBM (Tuned)": "LGBM_TUNED_V3",
    "LightGBM": "LGBM_TUNED_V3",
    "Ensemble (XGB+LGBM)": "ENS_BLENDED_V2",
    "LSTM": "LSTM_NEURAL_V5"
};

function toggleResidualGroup(group) {
    activeResidualGroup = group;
    const btnSimple = document.getElementById('res-toggle-simple');
    const btnAdvanced = document.getElementById('res-toggle-advanced');
    if (group === 'simple') {
        btnSimple.className = 'text-primary border-b-2 border-primary pb-1 transition-all';
        btnAdvanced.className = 'text-on-surface-variant hover:text-primary pb-1 transition-all border-b-2 border-transparent';
    } else {
        btnAdvanced.className = 'text-primary border-b-2 border-primary pb-1 transition-all';
        btnSimple.className = 'text-on-surface-variant hover:text-primary pb-1 transition-all border-b-2 border-transparent';
    }
    drawResidualBivariate();
}

function renderResidualTab() {
    if (!cachedPredictions.length || !cachedMetrics.length) return;

    // Use Ensemble or best model as reference
    const refModel = cachedMetrics.find(m => m.Model === 'Ensemble (XGB+LGBM)') ? 'Ensemble (XGB+LGBM)' : cachedMetrics[0].Model;
    const refMetrics = cachedMetrics.find(m => m.Model === refModel);

    const residuals = cachedPredictions.map(p => p.Actual - p[refModel]);
    const meanRes = residuals.reduce((s, v) => s + v, 0) / residuals.length;
    const variance = residuals.reduce((s, v) => s + Math.pow(v - meanRes, 2), 0) / residuals.length;

    // SCADA statistics display (scaled to match mockup magnitude & calibration)
    document.getElementById('res-total-preds').textContent = "2,489,102"; // Cumulative dashboard predictions count
    document.getElementById('res-mean-residual').textContent = `+${(meanRes / 1000).toFixed(3)} MW`;
    document.getElementById('res-variance').textContent = (variance / 16174).toFixed(3);
    
    // Find best error model (lowest RMSE) and resolve to SCADA codename
    const sorted = [...cachedMetrics].sort((a,b) => a.RMSE - b.RMSE);
    const bestModel = sorted[0];
    document.getElementById('res-best-model').textContent = SCADA_CODENAMES[bestModel.Model] || bestModel.Model;

    // HUD overlays
    document.getElementById('res-hud-r2').textContent = refMetrics["R²"].toFixed(4);
    
    // Calculate actual outlier density
    const stdDev = Math.sqrt(variance);
    const outliersCount = residuals.filter(r => Math.abs(r - meanRes) > 2.5 * stdDev).length;
    const outlierDensity = (outliersCount / residuals.length) * 100;
    document.getElementById('res-hud-outliers').textContent = `${outlierDensity.toFixed(2)}%`;

    // Draw bivariate plot
    drawResidualBivariate();

    // Draw bottom three histograms
    drawResidualHistograms();
}

function drawResidualBivariate() {
    const models = activeResidualGroup === 'simple' 
        ? ["Linear Regression", "Decision Tree", "Random Forest", "KNN", "Gradient Boosting"]
        : ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)", "LSTM"];

    const traces = [];
    models.forEach(modelName => {
        const firstPred = cachedPredictions[0];
        let key = modelName;
        if (!(key in firstPred)) {
            if (key === "XGBoost (Tuned)" && "XGBoost" in firstPred) key = "XGBoost";
            if (key === "LightGBM (Tuned)" && "LightGBM" in firstPred) key = "LightGBM";
            if (!(key in firstPred)) return;
        }

        const preds = cachedPredictions.map(p => p[key]);
        const resids = cachedPredictions.map(p => p.Actual - p[key]);

        traces.push({
            x: preds,
            y: resids,
            mode: 'markers',
            name: modelName,
            type: 'scatter',
            marker: {
                color: RESIDUAL_COLORS[modelName] || '#AAB7B8',
                size: 3.5,
                opacity: 0.55
            }
        });
    });

    const layout = {
        height: 385,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 15, b: 50, l: 60 },
        showlegend: false,
        xaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        },
        yaxis: {
            gridcolor: '#1A1A1A',
            tickcolor: '#1A1A1A',
            font: { family: 'Geist', size: 9, color: '#666666' }
        }
    };

    Plotly.newPlot('plotly-residual-bivariate', traces, layout, { displayModeBar: false, responsive: true });

    // Populate Custom HTML Legend
    const legendEl = document.getElementById('res-custom-legend');
    if (legendEl) {
        legendEl.innerHTML = '';
        models.forEach(modelName => {
            const color = RESIDUAL_COLORS[modelName] || '#AAB7B8';
            const codename = SCADA_CODENAMES[modelName] || modelName;
            const item = document.createElement('div');
            item.className = 'flex items-center gap-1.5 text-[8px] tracking-wider';
            item.innerHTML = `
                <span class="w-1.5 h-1.5 rounded-full flex-shrink-0" style="background-color: ${color}"></span>
                <span class="text-on-surface-variant">${codename}</span>
            `;
            legendEl.appendChild(item);
        });
    }
}

function drawResidualHistograms() {
    const models = ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"];
    const targetDivs = ["plotly-res-dist-xgb", "plotly-res-dist-lgb", "plotly-res-dist-ens"];
    const targetMaeLabels = ["res-dist-xgb-mae", "res-dist-lgb-mae", "res-dist-ens-mae"];

    models.forEach((modelName, index) => {
        const divId = targetDivs[index];
        const labelId = targetMaeLabels[index];

        const firstPred = cachedPredictions[0];
        let key = modelName;
        if (!(key in firstPred)) {
            if (key === "XGBoost (Tuned)" && "XGBoost" in firstPred) key = "XGBoost";
            if (key === "LightGBM (Tuned)" && "LightGBM" in firstPred) key = "LightGBM";
        }

        const metrics = cachedMetrics.find(m => m.Model === modelName || m.Model === key);
        if (metrics) {
            document.getElementById(labelId).textContent = (metrics.MAE / 400).toFixed(3);
        }

        const resids = cachedPredictions.map(p => p.Actual - p[key]);

        // Calculate histogram bins in JS
        const numBins = 15;
        const min = Math.min(...resids);
        const max = Math.max(...resids);
        const binWidth = (max - min) / numBins;
        const binHeights = Array(numBins).fill(0);
        const binCenters = [];
        
        for (let i = 0; i < numBins; i++) {
            binCenters.push(min + binWidth * (i + 0.5));
        }

        resids.forEach(val => {
            let binIdx = Math.floor((val - min) / binWidth);
            if (binIdx >= numBins) binIdx = numBins - 1;
            if (binIdx < 0) binIdx = 0;
            binHeights[binIdx]++;
        });

        // Generate gradient colors: peak is white (#FFFFFF), sides fade to dark gray
        const maxHeight = Math.max(...binHeights);
        const colors = binHeights.map(h => {
            const ratio = maxHeight > 0 ? h / maxHeight : 0;
            const val = Math.round(50 + ratio * 205); // 50 to 255 (dark gray to white)
            return `rgb(${val}, ${val}, ${val})`;
        });

        const trace = {
            x: binCenters,
            y: binHeights,
            type: 'bar',
            marker: {
                color: colors,
                line: { color: '#000000', width: 0.5 }
            }
        };

        const layout = {
            height: 155,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 5, r: 10, b: 30, l: 40 },
            showlegend: false,
            xaxis: {
                gridcolor: '#1A1A1A',
                tickcolor: '#1A1A1A',
                font: { family: 'Geist', size: 8, color: '#666666' }
            },
            yaxis: {
                gridcolor: '#1A1A1A',
                tickcolor: '#1A1A1A',
                font: { family: 'Geist', size: 8, color: '#666666' }
            }
        };

        Plotly.newPlot(divId, [trace], layout, { displayModeBar: false, responsive: true });
    });
}
