// JS for benchmark.html

let currentScenario = null;
let scenarios = [];

// Initializing on page load
document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();
});

function switchTab(tabName) {
    // Hiding all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Showing selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
    
    // Refreshing results
    if (tabName === 'results') {
        loadLatestResults();
    } else if (tabName === 'reports') {
        loadReports();
    }
}

async function loadScenarios() {
    try {
        const response = await fetch('/api/benchmark/scenarios');
        const data = await response.json();
        
        if (data.status === 'success') {
            scenarios = data.scenarios;
            renderScenarios();
        } else {
            showError('❌ Failed to load scenarios');
        }
    } catch (error) {
        showError('❌ Error loading scenarios: ' + error.message);
    }
}

function renderScenarios() {
    const container = document.getElementById('scenarios-list');
    
    if (scenarios.length === 0) {
        container.innerHTML = `
            <div class="status error">
                No scenarios found. Create a scenario in data/scenarios/
            </div>
            <div style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h3>How to create a scenario:</h3>
                <ol style="margin-top: 10px; line-height: 1.8;">
                    <li>Create JSON file in <code>data/scenarios/</code></li>
                    <li>Add turns with queries and ground truth</li>
                    <li>Place document in <code>data/documents/</code></li>
                    <li>Reload this page</li>
                </ol>
            </div>
        `;
        return;
    }
    
    container.innerHTML = scenarios.map(scenario => `
        <div class="scenario-card" onclick="selectScenario('${scenario}')">
            <h3>${scenario}</h3>
            <p>Click to run this benchmark scenario</p>
            <div class="button-group">
                <button onclick="event.stopPropagation(); runBenchmark('${scenario}')">
                    ▶ Run Benchmark
                </button>
                <button class="secondary" onclick="event.stopPropagation(); viewScenario('${scenario}')">
                    Details
                </button>
            </div>
        </div>
    `).join('');
}

function selectScenario(scenarioName) {
    currentScenario = scenarioName;
    alert(`Selected scenario: ${scenarioName}`);
}

async function runBenchmark(scenarioName) {
    currentScenario = scenarioName;
    
    // Showing progress
    document.getElementById('progress-section').style.display = 'block';
    document.getElementById('progress-info').textContent = `Running ${scenarioName}...`;
    document.getElementById('progress-bar').style.width = '10%';
    
    try {
        const response = await fetch(`/api/benchmark/run/${scenarioName}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            document.getElementById('progress-bar').style.width = '100%';
            document.getElementById('progress-info').className = 'status success';
            document.getElementById('progress-info').textContent = '✅ Benchmark completed successfully!';
            
            // Showing results
            showResults(result);
            
            // Switching to results tab
            switchTab('results');
            
            // Auto-generating the report after 2 seconds
            setTimeout(() => {
                generateReport(scenarioName);
            }, 2000);
        } else {
            showError(result.message || '❌ Benchmark failed');
        }
    } catch (error) {
        showError('❌ Benchmark error: ' + error.message);
        console.error(error);
    }
}

function showResults(result) {
    const container = document.getElementById('results-content');
    
    const summary = result.summary;
    const metrics = summary.performance_metrics || {};
    const retrieval = summary.retrieval_metrics || {};
    const generation = summary.generation_metrics || {};
    
    container.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Turns</div>
                <div class="metric-value">${summary.total_turns || 0}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value">${metrics.latency?.avg_ms || 0} ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak VRAM</div>
                <div class="metric-value">${metrics.vram?.peak_allocated_gb || 0} GB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Recall</div>
                <div class="metric-value">${((retrieval.avg_recall || 0) * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Coverage</div>
                <div class="metric-value">${((summary.final_coverage || 0) * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cache Hit</div>
                <div class="metric-value">${((summary.final_cache_hit_rate || 0) * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        <div class="result-section">
            <h3>Detailed Results</h3>
            <pre>${JSON.stringify(result.summary, null, 2)}</pre>
        </div>
        
        <div class="button-group" style="margin-top: 20px;">
            <button onclick="generateReport('${result.scenario}')">
                Generate Visual Report
            </button>
            <button class="secondary" onclick="downloadResults('${result.json_file}')">
                Download JSON
            </button>
            <button class="secondary" onclick="location.reload()">
                New Benchmark
            </button>
        </div>
    `;
}

async function generateReport(scenarioName) {
    document.getElementById('progress-info').className = 'status loading';
    document.getElementById('progress-info').textContent = 'Generating report...';
    
    try {
        const response = await fetch(`/api/benchmark/generate-report/${scenarioName}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            document.getElementById('progress-info').className = 'status success';
            document.getElementById('progress-info').textContent = '✅ Report generated!';
            
            // Loading reports tab
            switchTab('reports');
            loadReports();
        } else {
            showError(result.message || '❌ Report generation failed');
        }
    } catch (error) {
        showError('❌ Report error: ' + error.message);
    }
}

async function loadReports() {
    const container = document.getElementById('reports-content');
    
    if (!currentScenario) {
        container.innerHTML = '<div class="status info">Select a scenario first</div>';
        return;
    }
    
    try {
        const response = await fetch(`/api/benchmark/results/${currentScenario}`);
        const result = await response.json();
        
        if (result.status === 'success') {
            const summary = result.results.summary;
            
            container.innerHTML = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Scenario</div>
                        <div class="metric-value">${currentScenario}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Final Coverage</div>
                        <div class="metric-value">${(summary.final_coverage * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Recall</div>
                        <div class="metric-value">${((summary.retrieval_metrics?.avg_recall || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Peak VRAM</div>
                        <div class="metric-value">${summary.performance_metrics?.vram?.peak_allocated_gb || 0} GB</div>
                    </div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h3>Graphs</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h4>Coverage Progression</h4>
                            <img src="/static/plots/${currentScenario}_coverage.png" 
                                 alt="Coverage" 
                                 style="max-width: 100%; border-radius: 5px; margin-top: 10px;"
                                 onerror="this.style.display='none'; this.parentElement.innerHTML='<p>No coverage graph available</p>'">
                        </div>
                        <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h4>VRAM Usage</h4>
                            <img src="/static/plots/${currentScenario}_vram.png" 
                                 alt="VRAM" 
                                 style="max-width: 100%; border-radius: 5px; margin-top: 10px;"
                                 onerror="this.style.display='none'; this.parentElement.innerHTML='<p>No VRAM graph available</p>'">
                        </div>
                        <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h4>Latency</h4>
                            <img src="/static/plots/${currentScenario}_latency.png" 
                                 alt="Latency" 
                                 style="max-width: 100%; border-radius: 5px; margin-top: 10px;"
                                 onerror="this.style.display='none'; this.parentElement.innerHTML='<p>No latency graph available</p>'">
                        </div>
                        <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h4>Active Chunks</h4>
                            <img src="/static/plots/${currentScenario}_chunks.png" 
                                 alt="Chunks" 
                                 style="max-width: 100%; border-radius: 5px; margin-top: 10px;"
                                 onerror="this.style.display='none'; this.parentElement.innerHTML='<p>No chunks graph available</p>'">
                        </div>
                    </div>
                </div>
                
                <div class="button-group" style="margin-top: 20px;">
                    <button onclick="window.open('/static/plots/${currentScenario}_coverage.png', '_blank')">
                        View Coverage Graph
                    </button>
                    <button onclick="window.open('/static/plots/${currentScenario}_vram.png', '_blank')">
                        View VRAM Graph
                    </button>
                    <button class="secondary" onclick="location.reload()">
                        Refresh
                    </button>
                </div>
            `;
        } else {
            container.innerHTML = '<div class="status error">❌ No reports found. Run a benchmark first.</div>';
        }
    } catch (error) {
        container.innerHTML = `<div class="status error">❌ Error loading reports: ${error.message}</div>`;
    }
}

async function loadLatestResults() {
    if (!currentScenario) {
        document.getElementById('results-content').innerHTML = 
            '<div class="status info">Run a benchmark first</div>';
        return;
    }
    
    try {
        const response = await fetch(`/api/benchmark/results/${currentScenario}`);
        const result = await response.json();
        
        if (result.status === 'success') {
            showResults({
                scenario: currentScenario,
                summary: result.results.summary
            });
        }
    } catch (error) {
        console.error('❌ Error loading results:', error);
    }
}

function viewScenario(scenarioName) {
    alert(`Scenario: ${scenarioName}\n\nDetails will be shown in a modal in the future.`);
}

function downloadResults(filePath) {
    window.open(filePath, '_blank');
}

function showError(message) {
    document.getElementById('progress-info').className = 'status error';
    document.getElementById('progress-info').textContent = `❌ ${message}`;
}