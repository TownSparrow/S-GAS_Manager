// S-GAS Benchmark UI — supports S-GAS vs Baseline comparison

let currentScenario = null;
let lastResult = null;

document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();
});

function switchTab(tabName, clickedEl) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');
    if (clickedEl) clickedEl.classList.add('active');
    if (tabName === 'results' && lastResult) showResults(lastResult);
    if (tabName === 'reports') loadReports();
}

// ── Scenarios ──────────────────────────────────────────────────────

async function loadScenarios() {
    try {
        const resp = await fetch('/api/benchmark/scenarios');
        const data = await resp.json();
        if (data.status === 'success') renderScenarios(data.scenarios);
        else showStatus('scenarios-list', 'error', 'Failed to load scenarios');
    } catch (e) {
        showStatus('scenarios-list', 'error', 'Error: ' + e.message);
    }
}

function renderScenarios(scenarios) {
    const el = document.getElementById('scenarios-list');
    if (!scenarios.length) {
        el.innerHTML = '<div class="status info">No scenarios found. Place JSON files in tests/scenarios/</div>';
        return;
    }
    el.innerHTML = scenarios.map(s => `
        <div class="scenario-card">
            <h3>${s}</h3>
            <p>Runs S-GAS and Baseline modes, then compares results</p>
            <div class="button-group">
                <button onclick="runBenchmark('${s}')">Run Full Benchmark</button>
                <button class="secondary" onclick="runSingleMode('${s}', 'sgas')">S-GAS Only</button>
                <button class="secondary" onclick="runSingleMode('${s}', 'baseline')">Baseline Only</button>
            </div>
        </div>
    `).join('');
}

// ── Run Benchmark ──────────────────────────────────────────────────

async function runBenchmark(scenarioName) {
    currentScenario = scenarioName;
    const progress = document.getElementById('progress-section');
    const bar = document.getElementById('progress-bar');
    const info = document.getElementById('progress-info');
    progress.style.display = 'block';
    bar.style.width = '15%'; bar.textContent = '15%';
    info.className = 'status info';
    info.textContent = `Running ${scenarioName} (S-GAS + Baseline)...`;

    try {
        const resp = await fetch(`/api/benchmark/run/${scenarioName}`, { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        const result = await resp.json();
        if (result.status !== 'success') throw new Error(result.message || 'Benchmark failed');

        bar.style.width = '100%'; bar.textContent = '100%';
        info.className = 'status success';
        info.textContent = 'Benchmark completed!';
        lastResult = result;
        showResults(result);
        switchTab('results');
    } catch (e) {
        bar.style.width = '0%';
        info.className = 'status error';
        info.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

async function runSingleMode(scenarioName, mode) {
    currentScenario = scenarioName;
    const progress = document.getElementById('progress-section');
    const bar = document.getElementById('progress-bar');
    const info = document.getElementById('progress-info');
    progress.style.display = 'block';
    bar.style.width = '15%'; bar.textContent = '15%';
    info.className = 'status info';
    info.textContent = `Running ${scenarioName} (${mode.toUpperCase()} only)...`;

    try {
        const resp = await fetch(`/api/benchmark/run/${scenarioName}/${mode}`, { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        const result = await resp.json();
        if (result.status !== 'success') throw new Error(result.message || 'Benchmark failed');

        bar.style.width = '100%'; bar.textContent = '100%';
        info.className = 'status success';
        info.textContent = `${mode.toUpperCase()} benchmark completed!`;
        showSingleResult(result);
        switchTab('results');
    } catch (e) {
        bar.style.width = '0%';
        info.className = 'status error';
        info.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

function showSingleResult(result) {
    const el = document.getElementById('results-content');
    const s = result.summary || {};
    const mode = result.mode || 'unknown';
    const gen = s.generation_metrics || {};
    const ret = s.retrieval_metrics || {};

    el.innerHTML = `
        <h2>${mode.toUpperCase()} Only — ${result.scenario}</h2>

        <div class="verdict-box ${mode === 'sgas' ? 'verdict-sgas' : 'verdict-baseline'}">
            <h3>${mode.toUpperCase()} Single-Mode Run</h3>
        </div>

        <h3>Quality Metrics</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Text Recall</td><td>${((ret.avg_text_recall || 0) * 100).toFixed(1)}%</td></tr>
                <tr><td>Semantic Similarity</td><td>${((ret.avg_semantic_similarity || 0) * 100).toFixed(1)}%</td></tr>
                <tr><td>Multi-turn Accuracy</td><td>${((gen.multi_turn_accuracy || 0) * 100).toFixed(1)}% (${gen.correct_turns || 0}/${gen.total_turns || 0})</td></tr>
                <tr><td>Avg BERTScore</td><td>${(gen.avg_bertscore || 0).toFixed(4)}</td></tr>
                <tr><td>Coverage</td><td>${((s.final_coverage || 0) * 100).toFixed(1)}%</td></tr>
            </tbody>
        </table>

        <h3>Performance</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Avg Latency</td><td>${s.avg_latency_ms?.toFixed(0) || 0} ms</td></tr>
                <tr><td>Avg Latency (excl. 1st)</td><td>${s.avg_latency_excl_first_ms?.toFixed(0) || 0} ms</td></tr>
                <tr><td>Avg VRAM</td><td>${s.avg_vram_gb || 0} GB</td></tr>
                <tr><td>Cache Hit Rate</td><td>${((s.avg_cache_hit_rate || 0) * 100).toFixed(1)}%</td></tr>
                <tr><td>Total Swap Ops</td><td>${s.total_swap_operations || 0}</td></tr>
            </tbody>
        </table>

        <h3>Session Info</h3>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">Turns</div><div class="metric-value">${s.total_turns || 0}</div></div>
            <div class="metric-card"><div class="metric-label">Duration</div><div class="metric-value">${s.session_duration_s || 0}s</div></div>
            <div class="metric-card"><div class="metric-label">Avg Chunks</div><div class="metric-value">${s.avg_active_chunks || 0}</div></div>
            <div class="metric-card"><div class="metric-label">Peak RAM</div><div class="metric-value">${s.peak_ram_used_gb || 0} GB</div></div>
        </div>

        <h3>Download</h3>
        <div class="button-group">
            <button onclick="downloadBlob(new Blob([JSON.stringify(${JSON.stringify(s)}, null, 2)], {type:'application/json'}), '${result.scenario}_${mode}.json')">Download JSON</button>
        </div>
    `;
}

// ── Results Display ────────────────────────────────────────────────

function showResults(result) {
    const el = document.getElementById('results-content');
    const comp = result.comparison || {};
    const quality = comp.quality_metrics || {};
    const perf = comp.performance_metrics || {};
    const verdict = comp.verdict || {};
    const graph = comp.graph_usage || {};
    const sgasSummary = result.sgas?.summary || {};
    const baseSummary = result.baseline?.summary || {};

    el.innerHTML = `
        <h2>S-GAS vs Baseline — ${result.scenario}</h2>

        <!-- Verdict -->
        <div class="verdict-box ${verdict.overall?.includes('S-GAS') ? 'verdict-sgas' : verdict.overall?.includes('Baseline') ? 'verdict-baseline' : 'verdict-tie'}">
            <h3>${verdict.overall || 'No verdict'}</h3>
        </div>

        <!-- Per-metric verdicts -->
        <div class="verdicts-grid">
            ${Object.entries(verdict).filter(([k]) => k !== 'overall').map(([k, v]) => `
                <div class="verdict-item ${v.includes('S-GAS') ? 'win-sgas' : v.includes('Baseline') ? 'win-baseline' : 'win-tie'}">
                    <span class="verdict-metric">${k}</span>
                    <span class="verdict-result">${v}</span>
                </div>
            `).join('')}
        </div>

        <!-- Quality Metrics Comparison Table -->
        <h3>Quality Metrics</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>S-GAS</th><th>Baseline</th><th>Delta</th><th>Change</th></tr></thead>
            <tbody>
                ${renderRow('Text Recall (key-phrase)', quality.text_recall, true)}
                ${renderRow('Semantic Similarity', quality.semantic_similarity, true)}
                ${renderRow('Coverage', quality.coverage, true)}
                ${renderRow('Multi-turn Accuracy', quality.multi_turn_accuracy, true)}
            </tbody>
        </table>

        <!-- Performance Metrics -->
        <h3>Performance Metrics</h3>
        <table class="comparison-table">
            <thead><tr><th>Metric</th><th>S-GAS</th><th>Baseline</th><th>Delta</th><th>Change</th></tr></thead>
            <tbody>
                ${renderRow('Avg Latency (ms)', perf.avg_latency_ms)}
                ${renderRow('Avg Latency excl. 1st turn (ms)', perf.avg_latency_excl_first_ms)}
                ${renderRow('Avg VRAM Overhead (GB)', perf.avg_vram_gb)}
                ${renderRow('Cache Hit Rate', perf.avg_cache_hit_rate)}
                <tr>
                    <td>Swap Operations</td>
                    <td>${perf.total_swap_operations?.sgas ?? 0}</td>
                    <td>${perf.total_swap_operations?.baseline ?? 0}</td>
                    <td>-</td><td>-</td>
                </tr>
            </tbody>
        </table>

        <!-- System Resources -->
        <h3>System Resources</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">S-GAS Peak RAM</div>
                <div class="metric-value">${sgasSummary.peak_ram_used_gb || 0} GB</div>
                <div class="metric-label">${sgasSummary.avg_ram_percent || 0}% avg</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Baseline Peak RAM</div>
                <div class="metric-value">${baseSummary.peak_ram_used_gb || 0} GB</div>
                <div class="metric-label">${baseSummary.avg_ram_percent || 0}% avg</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">S-GAS Peak Process RSS</div>
                <div class="metric-value">${sgasSummary.peak_process_rss_mb || 0} MB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Baseline Peak Process RSS</div>
                <div class="metric-value">${baseSummary.peak_process_rss_mb || 0} MB</div>
            </div>
        </div>

        <!-- Graph Usage -->
        <h3>Graph Statistics (S-GAS only)</h3>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">Avg Nodes</div><div class="metric-value">${graph.avg_graph_nodes_sgas || 0}</div></div>
            <div class="metric-card"><div class="metric-label">Avg Edges</div><div class="metric-value">${graph.avg_graph_edges_sgas || 0}</div></div>
        </div>

        <!-- Summary cards -->
        <h3>Session Summaries</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div class="summary-card">
                <h4>S-GAS</h4>
                <p>Turns: ${sgasSummary.total_turns || 0} | Duration: ${sgasSummary.session_duration_s || 0}s</p>
                <p>Avg Latency: ${sgasSummary.avg_latency_ms?.toFixed(0) || 0}ms | Avg VRAM: ${sgasSummary.avg_vram_gb || 0} GB</p>
                <p>Coverage: ${((sgasSummary.final_coverage || 0) * 100).toFixed(1)}% | Swaps: ${sgasSummary.total_swap_operations || 0}</p>
            </div>
            <div class="summary-card">
                <h4>Baseline</h4>
                <p>Turns: ${baseSummary.total_turns || 0} | Duration: ${baseSummary.session_duration_s || 0}s</p>
                <p>Avg Latency: ${baseSummary.avg_latency_ms?.toFixed(0) || 0}ms | Avg VRAM: ${baseSummary.avg_vram_gb || 0} GB</p>
                <p>Coverage: ${((baseSummary.final_coverage || 0) * 100).toFixed(1)}%</p>
            </div>
        </div>

        <!-- Download buttons -->
        <h3>Download Results</h3>
        <div class="button-group">
            <button onclick="downloadJSON('comparison')">Download Comparison JSON</button>
            <button class="secondary" onclick="downloadJSON('sgas')">Download S-GAS JSON</button>
            <button class="secondary" onclick="downloadJSON('baseline')">Download Baseline JSON</button>
            <button class="secondary" onclick="downloadCSV('sgas')">Download S-GAS CSV</button>
            <button class="secondary" onclick="downloadCSV('baseline')">Download Baseline CSV</button>
            <button onclick="generateReport('${result.scenario}')">Generate Visual Report</button>
        </div>
    `;
}

function renderRow(label, data, isPercent = false) {
    if (!data || typeof data !== 'object') return `<tr><td>${label}</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>`;
    // isPercent: values are 0–1 fractions that should display as X.X%
    const fmt = v => isPercent ? (v * 100).toFixed(1) + '%' : typeof v === 'number' ? v.toFixed(4) : v;
    const pct = data.improvement_pct || 0;
    // For lower-is-better metrics a positive improvement_pct means S-GAS is better
    const cls = pct > 0 ? 'positive' : pct < 0 ? 'negative' : '';
    return `<tr>
        <td>${label}</td>
        <td>${fmt(data.sgas)}</td>
        <td>${fmt(data.baseline)}</td>
        <td>${data.delta !== undefined ? (data.delta >= 0 ? '+' : '') + data.delta.toFixed(4) : '-'}</td>
        <td class="${cls}">${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</td>
    </tr>`;
}

// ── Downloads ──────────────────────────────────────────────────────

function downloadJSON(mode) {
    if (!lastResult) return alert('No results available');
    let data, filename;
    if (mode === 'comparison') {
        data = lastResult.comparison;
        filename = `${currentScenario}_comparison.json`;
    } else if (mode === 'sgas') {
        data = lastResult.sgas?.summary;
        filename = `${currentScenario}_sgas.json`;
    } else {
        data = lastResult.baseline?.summary;
        filename = `${currentScenario}_baseline.json`;
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    downloadBlob(blob, filename);
}

function downloadCSV(mode) {
    if (!lastResult) return alert('No results available');
    const file = mode === 'sgas' ? lastResult.sgas?.csv_file : lastResult.baseline?.csv_file;
    if (file) {
        alert(`CSV saved at: ${file}\nOpen the file directly on the server.`);
    } else {
        alert('CSV file path not available');
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ── Reports ────────────────────────────────────────────────────────

async function generateReport(scenarioName) {
    const info = document.getElementById('progress-info');
    info.className = 'status info';
    info.textContent = 'Generating visual report...';
    try {
        const resp = await fetch(`/api/benchmark/generate-report/${scenarioName}`, { method: 'POST' });
        const result = await resp.json();
        if (result.status === 'success') {
            info.className = 'status success';
            info.textContent = 'Report generated!';
            loadReports();
            switchTab('reports');
        } else {
            info.className = 'status error';
            info.textContent = 'Report generation failed: ' + (result.detail || result.message);
        }
    } catch (e) {
        info.className = 'status error';
        info.textContent = 'Error: ' + e.message;
    }
}

async function loadReports() {
    const el = document.getElementById('reports-content');
    if (!currentScenario) {
        el.innerHTML = '<div class="status info">Run a benchmark first to see reports</div>';
        return;
    }
    try {
        const resp = await fetch(`/api/benchmark/results/${currentScenario}`);
        const result = await resp.json();
        if (result.status === 'success') {
            el.innerHTML = `
                <h3>Report for ${currentScenario}</h3>
                <div class="plots-grid">
                    ${['coverage', 'vram', 'latency', 'chunks'].map(type => `
                        <div class="plot-card">
                            <h4>${type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                            <img src="/static/plots/${currentScenario}_${type}.png"
                                 alt="${type}" style="max-width:100%"
                                 onerror="this.parentElement.innerHTML='<p>No ${type} plot available</p>'">
                        </div>
                    `).join('')}
                </div>
                <div class="button-group" style="margin-top:20px;">
                    <button onclick="downloadFullReport()">Download Full Report (JSON)</button>
                </div>
            `;
        } else {
            el.innerHTML = '<div class="status info">No reports found. Run a benchmark first.</div>';
        }
    } catch (e) {
        el.innerHTML = `<div class="status error">Error: ${e.message}</div>`;
    }
}

function downloadFullReport() {
    if (!lastResult) return alert('No results');
    const report = {
        scenario: currentScenario,
        timestamp: new Date().toISOString(),
        comparison: lastResult.comparison,
        sgas_summary: lastResult.sgas?.summary,
        baseline_summary: lastResult.baseline?.summary,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    downloadBlob(blob, `${currentScenario}_full_report.json`);
}

async function loadLatestResults() {
    if (!currentScenario) return;
    try {
        const resp = await fetch(`/api/benchmark/results/${currentScenario}`);
        const result = await resp.json();
        if (result.status === 'success' && result.type === 'comparison') {
            lastResult = { scenario: currentScenario, comparison: result.results };
            showResults(lastResult);
        }
    } catch (e) { console.error(e); }
}

function showStatus(elementId, type, message) {
    document.getElementById(elementId).innerHTML = `<div class="status ${type}">${message}</div>`;
}
