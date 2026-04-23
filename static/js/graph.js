// S-GAS Knowledge Graph — visualization client
// Endpoint: GET /api/graph/data
// Response shape: { status, node_count, edge_count, statistics, nodes, edges }
//   nodes: [{ id, type, label, text }]
//   edges: [{ source, target, weight, relation }]

const GRAPH_API = '/api/graph/data';

const NODE_RADIUS = { chunk: 8, concept: 5 };

// D3 selections that need to be accessible across functions
let svgEl, rootG, simulation;

const zoomBehavior = d3.zoom()
    .scaleExtent([0.04, 5])
    .on('zoom', e => rootG.attr('transform', e.transform));

// ── Initialise SVG and zoom once the DOM is ready ─────────────────

document.addEventListener('DOMContentLoaded', () => {
    svgEl = d3.select('#graph-svg');
    rootG = d3.select('#root');
    svgEl.call(zoomBehavior);
    loadGraph();
});

// ── Public helpers (called from HTML onclick) ─────────────────────

function resetZoom() {
    const w = svgEl.node().clientWidth;
    const h = svgEl.node().clientHeight;
    svgEl.transition().duration(400).call(
        zoomBehavior.transform,
        d3.zoomIdentity.translate(w / 2, h / 2).scale(0.55)
    );
}

// ── Data loading ──────────────────────────────────────────────────

async function loadGraph() {
    setLoading(true);
    hideError();
    rootG.selectAll('*').remove();
    if (simulation) simulation.stop();

    try {
        const resp = await fetch(GRAPH_API);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const data = await resp.json();
        if (data.status !== 'success') throw new Error(data.message || 'API returned non-success status');

        updateStats(data);

        if (!data.nodes || data.nodes.length === 0) {
            setLoading(true, 'Graph is empty — run a benchmark first.');
            return;
        }

        renderGraph(data.nodes, data.edges || []);
    } catch (err) {
        showError('Failed to load graph: ' + err.message);
    } finally {
        setLoading(false);
    }
}

// ── Stats bar ─────────────────────────────────────────────────────

function updateStats(data) {
    setText('stat-nodes', data.node_count ?? '—');
    setText('stat-edges', data.edge_count ?? '—');
    const s = data.statistics || {};
    setText('stat-chunks',   s.chunk_nodes   ?? '—');
    setText('stat-concepts', s.concept_nodes ?? '—');
}

// ── Graph rendering ───────────────────────────────────────────────

function renderGraph(nodes, edges) {
    // D3 force layout uses string ids via forceLink.id(); no index mapping needed.
    const nodeById = new Map(nodes.map(n => [n.id, n]));

    // Keep only edges whose both endpoints exist in the node list.
    const validEdges = edges.filter(
        e => nodeById.has(e.source) && nodeById.has(e.target)
    );

    // ── Links ─────────────────────────────────────────────────────
    const linkSel = rootG.append('g').attr('class', 'links')
        .selectAll('line')
        .data(validEdges)
        .join('line')
        .attr('class', d => 'link ' + edgeClass(d.relation))
        .attr('stroke-width', d => strokeWidth(d.weight));

    // ── Nodes ─────────────────────────────────────────────────────
    const nodeSel = rootG.append('g').attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', d => 'node ' + (d.type || 'unknown'))
        .call(dragBehavior())
        .on('click', (_event, d) => showNodeDetail(d));

    nodeSel.append('circle').attr('r', d => NODE_RADIUS[d.type] ?? 5);

    // Show short label only for chunk nodes to avoid text clutter.
    nodeSel.filter(d => d.type === 'chunk')
        .append('text')
        .attr('dy', d => (NODE_RADIUS[d.type] ?? 5) + 10)
        .text(d => truncate(d.text, 22));

    // ── Force simulation ──────────────────────────────────────────
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(validEdges)
            .id(d => d.id)
            .distance(d => d.relation && d.relation.includes('semantic') ? 90 : 45)
            .strength(0.35))
        .force('charge', d3.forceManyBody().strength(-70))
        .force('center', d3.forceCenter(0, 0))
        .force('collision', d3.forceCollide().radius(d => (NODE_RADIUS[d.type] ?? 5) + 4))
        .on('tick', () => {
            linkSel
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
        });

    // Zoom to fit after the layout has settled a bit.
    setTimeout(resetZoom, 700);
}

// ── Node detail panel ─────────────────────────────────────────────

function showNodeDetail(d) {
    d3.selectAll('.node').classed('selected', false);
    d3.selectAll('.node').filter(n => n.id === d.id).classed('selected', true);

    const fields = [
        ['ID',    d.id],
        ['Type',  d.type],
        ['Label', d.label],
        ['Text',  d.text ? truncate(d.text, 160) : '—'],
    ];
    document.getElementById('node-detail').innerHTML = fields.map(([k, v]) =>
        `<div class="field">
            <div class="label">${k}</div>
            <div class="value">${v ?? '—'}</div>
        </div>`
    ).join('');
}

// ── Drag behaviour ────────────────────────────────────────────────

function dragBehavior() {
    return d3.drag()
        .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        })
        .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
        })
        .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        });
}

// ── Helpers ───────────────────────────────────────────────────────

function edgeClass(relation) {
    if (!relation) return '';
    if (relation.includes('semantic'))    return 'semantic';
    if (relation.includes('common'))      return 'common';
    if (relation.includes('contains') || relation.includes('found_in')) return 'concept';
    return '';
}

function strokeWidth(weight) {
    if (weight == null) return 0.8;
    return Math.max(0.5, (1 - Number(weight)) * 2.5);
}

function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.slice(0, maxLen) + '…' : str;
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setLoading(visible, message) {
    const el = document.getElementById('loading');
    if (!el) return;
    el.style.display = visible ? 'flex' : 'none';
    if (message) el.textContent = message;
}

function showError(msg) {
    const el = document.getElementById('error-msg');
    if (!el) return;
    el.textContent = msg;
    el.style.display = 'block';
}

function hideError() {
    const el = document.getElementById('error-msg');
    if (el) el.style.display = 'none';
}
