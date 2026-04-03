import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================
// State
// ============================================================
const state = {
    // Mesh data
    numVerts: 0,
    numFaces: 0,
    numExpr: 0,
    templateVerts: null,   // Float32Array [numVerts * 3]
    faceIndices: null,     // Uint32Array  [numFaces * 3]
    exprBasis: null,       // Float32Array [numExpr * numVerts * 3]
    meshLoaded: false,

    // Sequence data
    sequences: [],         // [{name, data, frames, dims, has_gt, gt_filename}]
    activeSequence: null,
    activeGT: null,        // Ground truth sequence data {data, frames, dims} or null
    viewMode: 'prediction', // 'prediction' | 'ground_truth' | 'overlay'
    errorPerFrame: null,   // Float32Array [T] of per-frame L1 errors

    // Playback
    playing: false,
    currentFrame: 0,
    speed: 1.0,
    fps: 30,
    lastFrameTime: 0,

    // Audio
    audio: null,
    audioLoaded: false,
    audioAvailable: false,

    // WebSocket
    ws: null,
    wsConnected: false,

    // Three.js
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    mesh: null,
    geometry: null,
    gtMesh: null,          // Overlay wireframe mesh for ground truth
    gtGeometry: null,
};

// ============================================================
// Three.js Setup
// ============================================================
function initThreeJS() {
    const viewport = document.getElementById('viewport');
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;

    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x1a1a2e);

    state.camera = new THREE.PerspectiveCamera(30, width / height, 0.001, 10);
    state.camera.position.set(0, 0, 0.5);

    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(width, height);
    state.renderer.setPixelRatio(window.devicePixelRatio);
    viewport.appendChild(state.renderer.domElement);

    state.controls = new OrbitControls(state.camera, state.renderer.domElement);
    state.controls.target.set(0, 0, 0);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.1;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    state.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(1, 1, 2);
    state.scene.add(dirLight);

    const backLight = new THREE.DirectionalLight(0x6688cc, 0.4);
    backLight.position.set(-1, 0.5, -1);
    state.scene.add(backLight);

    // Grid helper for reference
    const grid = new THREE.GridHelper(0.5, 10, 0x333355, 0x222244);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.15;
    state.scene.add(grid);

    window.addEventListener('resize', () => {
        const w = viewport.clientWidth;
        const h = viewport.clientHeight;
        state.camera.aspect = w / h;
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(w, h);
    });
}

// ============================================================
// FLAME Mesh Loading
// ============================================================
function loadFlameBinary(buffer) {
    const headerView = new DataView(buffer, 0, 12);
    state.numVerts = headerView.getUint32(0, true);
    state.numFaces = headerView.getUint32(4, true);
    state.numExpr = headerView.getUint32(8, true);

    let offset = 12;

    // Template vertices
    const vertBytes = state.numVerts * 3 * 4;
    state.templateVerts = new Float32Array(buffer, offset, state.numVerts * 3);
    offset += vertBytes;

    // Face indices
    const faceBytes = state.numFaces * 3 * 4;
    state.faceIndices = new Uint32Array(buffer, offset, state.numFaces * 3);
    offset += faceBytes;

    // Expression basis
    const basisBytes = state.numExpr * state.numVerts * 3 * 4;
    state.exprBasis = new Float32Array(buffer, offset, state.numExpr * state.numVerts * 3);

    state.meshLoaded = true;
    createMesh();
    setStatus(`Mesh loaded: ${state.numVerts} verts, ${state.numFaces} faces, ${state.numExpr} expressions`);
    document.getElementById('mesh-status').textContent =
        `${state.numVerts} vertices, ${state.numFaces} faces, ${state.numExpr} expr dims`;
    document.getElementById('no-data-msg').style.display = 'none';

    // Build weight sliders
    buildWeightSliders();
}

function createMesh() {
    // Remove old mesh
    if (state.mesh) {
        state.scene.remove(state.mesh);
        state.geometry.dispose();
    }

    state.geometry = new THREE.BufferGeometry();

    // Clone template for working positions
    const positions = new Float32Array(state.templateVerts.length);
    positions.set(state.templateVerts);

    state.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    state.geometry.setIndex(new THREE.BufferAttribute(state.faceIndices, 1));
    state.geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
        color: 0xc8a88c,
        specular: 0x222222,
        shininess: 20,
        side: THREE.DoubleSide,
        flatShading: false,
    });

    state.mesh = new THREE.Mesh(state.geometry, material);
    state.scene.add(state.mesh);

    // Center camera on mesh
    state.geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    state.geometry.boundingBox.getCenter(center);
    state.controls.target.copy(center);
    state.camera.position.set(center.x, center.y, center.z + 0.5);
    state.controls.update();
}

function createGTOverlayMesh() {
    if (state.gtMesh) {
        state.scene.remove(state.gtMesh);
        state.gtGeometry.dispose();
    }

    state.gtGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(state.templateVerts.length);
    positions.set(state.templateVerts);

    state.gtGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    state.gtGeometry.setIndex(new THREE.BufferAttribute(state.faceIndices, 1));
    state.gtGeometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
        color: 0x4a90d9,
        specular: 0x222222,
        shininess: 20,
        side: THREE.DoubleSide,
        wireframe: true,
        opacity: 0.5,
        transparent: true,
    });

    state.gtMesh = new THREE.Mesh(state.gtGeometry, material);
    state.gtMesh.visible = false;
    state.scene.add(state.gtMesh);
}

// ============================================================
// Expression Weight Application
// ============================================================
function applyWeightsToGeometry(weights, geometry) {
    if (!state.meshLoaded || !geometry) return;

    const positions = geometry.attributes.position.array;
    const numVerts3 = state.numVerts * 3;

    positions.set(state.templateVerts);

    for (let e = 0; e < state.numExpr && e < weights.length; e++) {
        const w = weights[e];
        if (Math.abs(w) < 0.0001) continue;
        const basisOffset = e * numVerts3;
        for (let i = 0; i < numVerts3; i++) {
            positions[i] += w * state.exprBasis[basisOffset + i];
        }
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
}

function applyWeights(weights) {
    if (!state.meshLoaded || !state.geometry) return;

    // Apply to main mesh
    applyWeightsToGeometry(weights, state.geometry);

    // In overlay mode, also apply GT weights to the overlay mesh
    if (state.viewMode === 'overlay' && state.gtMesh && state.activeGT) {
        const gtWeights = getGTFrameWeights(state.currentFrame);
        if (gtWeights) {
            applyWeightsToGeometry(gtWeights, state.gtGeometry);
        }
    }

    // Update weight display
    updateWeightDisplay(weights);

    // Update error display
    updateErrorDisplay();
}

// ============================================================
// Sequence Management
// ============================================================
function loadSequenceFromBuffer(name, buffer) {
    // .npy format: parse header to get shape, then read float32 data
    const data = parseNpy(buffer);
    if (!data) {
        setStatus(`Failed to parse ${name}`);
        return;
    }

    const seq = {
        name: name,
        data: data.data,
        frames: data.shape[0],
        dims: data.shape.length > 1 ? data.shape[1] : 1,
    };

    // Replace if name exists, otherwise add
    const idx = state.sequences.findIndex(s => s.name === name);
    if (idx >= 0) {
        state.sequences[idx] = seq;
    } else {
        state.sequences.push(seq);
    }

    renderSequenceList();
    selectSequence(name);
    setStatus(`Loaded sequence: ${name} (${seq.frames} frames, ${seq.dims} dims)`);
}

function selectSequence(name) {
    state.activeSequence = state.sequences.find(s => s.name === name) || null;
    state.activeGT = null;
    state.errorPerFrame = null;
    state.currentFrame = 0;
    state.playing = false;
    state.viewMode = 'prediction';

    // Unload previous audio
    unloadAudio();

    if (state.activeSequence) {
        document.getElementById('timeline').max = state.activeSequence.frames - 1;
        document.getElementById('timeline').value = 0;

        // Load audio if available
        if (state.activeSequence.has_audio && state.activeSequence.audio_filename) {
            state.audioAvailable = true;
            loadAudio(state.activeSequence.audio_filename);
        } else {
            state.audioAvailable = false;
            updateAudioUI();
        }

        // Load ground truth if available
        if (state.activeSequence.has_gt && state.activeSequence.gt_filename) {
            loadGroundTruth(state.activeSequence.gt_filename);
        }

        updateComparisonUI();
        updateFrameInfo();

        const weights = getFrameWeights(0);
        if (weights) applyWeights(weights);
    }

    renderSequenceList();
}

async function loadGroundTruth(gt_filename) {
    try {
        const resp = await fetch(`/api/sequences/${gt_filename}`);
        if (!resp.ok) return;
        const buf = await resp.arrayBuffer();
        const header = new DataView(buf, 0, 8);
        const rows = header.getUint32(0, true);
        const cols = header.getUint32(4, true);
        const floatData = new Float32Array(buf, 8);

        state.activeGT = { data: floatData, frames: rows, dims: cols };

        // Compute per-frame L1 error
        computeErrorPerFrame();
        drawErrorGraph();
        updateComparisonUI();
        setStatus(`Ground truth loaded (${rows} frames)`);
    } catch (e) {
        console.error('Failed to load ground truth:', e);
    }
}

function computeErrorPerFrame() {
    if (!state.activeSequence || !state.activeGT) return;
    const pred = state.activeSequence;
    const gt = state.activeGT;
    const T = Math.min(pred.frames, gt.frames);
    const D = Math.min(pred.dims, gt.dims);

    state.errorPerFrame = new Float32Array(T);
    for (let t = 0; t < T; t++) {
        let sum = 0;
        for (let d = 0; d < D; d++) {
            sum += Math.abs(pred.data[t * pred.dims + d] - gt.data[t * gt.dims + d]);
        }
        state.errorPerFrame[t] = sum / D;
    }
}

function drawErrorGraph() {
    const canvas = document.getElementById('error-graph');
    if (!canvas || !state.errorPerFrame) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.clientWidth * window.devicePixelRatio;
    const H = canvas.height = 40 * window.devicePixelRatio;
    ctx.clearRect(0, 0, W, H);

    const errs = state.errorPerFrame;
    const maxErr = Math.max(...errs) || 1;

    ctx.strokeStyle = '#d94a4a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < errs.length; i++) {
        const x = (i / errs.length) * W;
        const y = H - (errs[i] / maxErr) * H;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

function updateComparisonUI() {
    const section = document.getElementById('comparison-section');
    if (state.activeGT) {
        section.style.display = '';
        if (state.gtMesh) state.gtMesh.visible = false;
        // Create overlay mesh if needed
        if (!state.gtMesh && state.meshLoaded) createGTOverlayMesh();
    } else {
        section.style.display = 'none';
    }
    setViewMode(state.viewMode);
}

function setViewMode(mode) {
    state.viewMode = mode;

    // Update button styles
    ['btn-pred', 'btn-gt', 'btn-overlay'].forEach(id => {
        document.getElementById(id)?.classList.remove('mode-active');
    });
    const activeBtn = { prediction: 'btn-pred', ground_truth: 'btn-gt', overlay: 'btn-overlay' }[mode];
    document.getElementById(activeBtn)?.classList.add('mode-active');

    // Update mesh visibility and color
    if (state.mesh) {
        state.mesh.visible = true;
        state.mesh.material.color.setHex(mode === 'ground_truth' ? 0x7a9fc8 : 0xc8a88c);
        state.mesh.material.wireframe = false;
    }
    if (state.gtMesh) {
        state.gtMesh.visible = (mode === 'overlay' && state.activeGT !== null);
    }

    // Re-apply current frame
    const weights = getFrameWeights(state.currentFrame);
    if (weights) applyWeights(weights);
}

function getFrameWeights(frame) {
    // In ground_truth mode, return GT weights; otherwise return prediction
    const useGT = state.viewMode === 'ground_truth' && state.activeGT;
    const seq = useGT ? state.activeGT : state.activeSequence;
    if (!seq) return null;
    if (frame >= seq.frames) return null;
    const offset = frame * seq.dims;
    return Array.from(seq.data.slice(offset, offset + seq.dims));
}

function getGTFrameWeights(frame) {
    if (!state.activeGT || frame >= state.activeGT.frames) return null;
    const offset = frame * state.activeGT.dims;
    return Array.from(state.activeGT.data.slice(offset, offset + state.activeGT.dims));
}

// ============================================================
// NPY Parser
// ============================================================
function parseNpy(buffer) {
    const view = new DataView(buffer);
    // Check magic number: \x93NUMPY
    const magic = new Uint8Array(buffer, 0, 6);
    if (magic[0] !== 0x93 || String.fromCharCode(magic[1], magic[2], magic[3], magic[4], magic[5]) !== 'NUMPY') {
        console.error('Not a valid .npy file');
        return null;
    }

    const major = view.getUint8(6);
    const minor = view.getUint8(7);

    let headerLen;
    let headerOffset;
    if (major === 1) {
        headerLen = view.getUint16(8, true);
        headerOffset = 10;
    } else {
        headerLen = view.getUint32(8, true);
        headerOffset = 12;
    }

    const headerStr = new TextDecoder().decode(new Uint8Array(buffer, headerOffset, headerLen));
    const dataOffset = headerOffset + headerLen;

    // Parse header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (150, 100), }
    const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
    if (!shapeMatch) return null;

    const shape = shapeMatch[1].split(',').map(s => s.trim()).filter(s => s).map(Number);
    const totalElements = shape.reduce((a, b) => a * b, 1);

    // Determine dtype
    const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']*)'/);
    const descr = descrMatch ? descrMatch[1] : '<f4';

    let data;
    if (descr.includes('f4') || descr.includes('float32')) {
        data = new Float32Array(buffer, dataOffset, totalElements);
    } else if (descr.includes('f8') || descr.includes('float64')) {
        const f64 = new Float64Array(buffer, dataOffset, totalElements);
        data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) data[i] = f64[i];
    } else {
        console.error('Unsupported dtype:', descr);
        return null;
    }

    return { shape, data };
}

// ============================================================
// WebSocket
// ============================================================
function wsConnect() {
    const seqName = document.getElementById('ws-sequence-name').value || 'demo';
    const host = window.location.hostname || 'localhost';
    const port = window.location.port || '8765';

    state.ws = new WebSocket(`ws://${host}:${port}/ws/stream`);

    state.ws.onopen = () => {
        state.wsConnected = true;
        document.getElementById('ws-indicator').className = 'ws-status connected';
        state.ws.send(JSON.stringify({ sequence: seqName, fps: 30 }));
        setStatus('WebSocket connected');
    };

    state.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'frame' && msg.weights && !state.activeSequence) {
            // Only apply WS frames when no local sequence is loaded
            applyWeights(msg.weights);
            state.currentFrame = msg.frame;
            updateFrameInfo();
        } else if (msg.type === 'init') {
            document.getElementById('timeline').max = msg.frames - 1;
            setStatus(`Streaming: ${msg.frames} frames at ${msg.fps} fps`);
        } else if (msg.error) {
            setStatus(`WS Error: ${msg.error}`);
        }
    };

    state.ws.onclose = () => {
        state.wsConnected = false;
        document.getElementById('ws-indicator').className = 'ws-status disconnected';
        setStatus('WebSocket disconnected');
    };
}

function wsDisconnect() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
    state.wsConnected = false;
    document.getElementById('ws-indicator').className = 'ws-status disconnected';
}

// ============================================================
// UI
// ============================================================
function renderSequenceList() {
    const container = document.getElementById('sequence-list');
    container.innerHTML = '';

    for (const seq of state.sequences) {
        const div = document.createElement('div');
        div.className = 'sequence-item' + (state.activeSequence === seq ? ' active' : '');
        const gtBadge = seq.has_gt
            ? '<span class="badge badge-gt">Pred + GT</span>'
            : '<span class="badge badge-pred">Pred only</span>';
        div.innerHTML = `
            <div>${seq.name}${gtBadge}</div>
            <div class="sequence-meta">${seq.frames} frames (${(seq.frames / 30).toFixed(1)}s) | ${seq.dims} dims</div>
        `;
        div.addEventListener('click', () => selectSequence(seq.name));
        container.appendChild(div);
    }
}

function buildWeightSliders() {
    const container = document.getElementById('weight-sliders');
    container.innerHTML = '';

    const numToShow = Math.min(state.numExpr, 50); // Show first 50 for performance
    for (let i = 0; i < numToShow; i++) {
        const row = document.createElement('div');
        row.className = 'weight-row';
        row.innerHTML = `
            <span class="weight-label">${i}</span>
            <div class="weight-bar"><div class="weight-bar-fill" id="bar-${i}"></div></div>
            <span class="weight-value" id="wval-${i}">0.00</span>
        `;
        container.appendChild(row);
    }
}

function updateWeightDisplay(weights) {
    const numToShow = Math.min(weights.length, 50);
    for (let i = 0; i < numToShow; i++) {
        const val = weights[i] || 0;
        const valEl = document.getElementById(`wval-${i}`);
        const barEl = document.getElementById(`bar-${i}`);
        if (valEl) valEl.textContent = val.toFixed(2);
        if (barEl) {
            // Map value to bar width (handle negative values)
            const pct = Math.min(Math.abs(val) * 100, 100);
            barEl.style.width = pct + '%';
            barEl.style.background = val >= 0 ? '#4a90d9' : '#d94a4a';
        }
    }
}

function updateFrameInfo() {
    const info = document.getElementById('frame-info');
    if (state.activeSequence) {
        const t = (state.currentFrame / state.fps).toFixed(2);
        const modeLabel = { prediction: 'Pred', ground_truth: 'GT', overlay: 'Overlay' }[state.viewMode];
        info.textContent = `[${modeLabel}] Frame ${state.currentFrame} / ${state.activeSequence.frames - 1} | ${t}s`;
    } else {
        info.textContent = '';
    }
    document.getElementById('timeline').value = state.currentFrame;
}

function updateErrorDisplay() {
    const el = document.getElementById('error-display');
    if (!el) return;
    if (!state.errorPerFrame || state.currentFrame >= state.errorPerFrame.length) {
        el.textContent = 'Frame L1: --';
        return;
    }
    const frameErr = state.errorPerFrame[state.currentFrame];
    const avgErr = state.errorPerFrame.reduce((a, b) => a + b, 0) / state.errorPerFrame.length;
    el.textContent = `Frame L1: ${frameErr.toFixed(4)} | Avg: ${avgErr.toFixed(4)}`;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

// ============================================================
// Audio Sync
// ============================================================
function loadAudio(audioFilename) {
    unloadAudio();
    state.audio = new Audio(`/static/sequences/${audioFilename}`);
    state.audio.loop = false;
    state.audio.volume = parseFloat(document.getElementById('volume-slider').value);

    const statusEl = document.getElementById('audio-status');
    statusEl.textContent = 'Loading audio...';
    statusEl.className = 'audio-indicator loading';

    state.audio.addEventListener('canplaythrough', () => {
        state.audioLoaded = true;
        statusEl.textContent = 'Audio ready';
        statusEl.className = 'audio-indicator ready';
    }, { once: true });

    state.audio.addEventListener('error', () => {
        state.audioLoaded = false;
        statusEl.textContent = 'Audio error';
        statusEl.className = 'audio-indicator no-audio';
    });

    state.audio.load();
}

function unloadAudio() {
    if (state.audio) {
        state.audio.pause();
        state.audio.src = '';
        state.audio = null;
    }
    state.audioLoaded = false;
    state.audioAvailable = false;
    updateAudioUI();
}

function startAudio() {
    if (!state.audio || !state.audioLoaded) return;
    state.audio.currentTime = state.currentFrame / state.fps;
    state.audio.playbackRate = state.speed;
    state.audio.play().catch(() => {});
}

function pauseAudio() {
    if (state.audio && !state.audio.paused) {
        state.audio.pause();
    }
}

function resetAudio() {
    if (state.audio) {
        state.audio.pause();
        state.audio.currentTime = 0;
    }
}

function syncAudioToFrame() {
    if (state.audio && state.audioLoaded) {
        state.audio.currentTime = state.currentFrame / state.fps;
    }
}

function updateAudioUI() {
    const statusEl = document.getElementById('audio-status');
    if (!statusEl) return;
    if (!state.audioAvailable) {
        statusEl.textContent = 'No audio';
        statusEl.className = 'audio-indicator no-audio';
    }
}

// ============================================================
// Animation Loop
// ============================================================
let fpsFrames = 0;
let fpsTime = performance.now();

function animate() {
    requestAnimationFrame(animate);

    // FPS counter
    fpsFrames++;
    const now = performance.now();
    if (now - fpsTime >= 1000) {
        document.getElementById('fps-counter').textContent = fpsFrames + ' fps';
        fpsFrames = 0;
        fpsTime = now;
    }

    // Playback (local file sequences)
    if (state.playing && state.activeSequence) {
        if (state.audioLoaded && state.audio && !state.audio.paused) {
            // Audio-driven: derive frame from audio's currentTime
            const audioFrame = Math.floor(state.audio.currentTime * state.fps);
            if (audioFrame >= state.activeSequence.frames) {
                state.audio.currentTime = 0;
                state.currentFrame = 0;
            } else {
                state.currentFrame = audioFrame;
            }
            const weights = getFrameWeights(state.currentFrame);
            if (weights) applyWeights(weights);
            updateFrameInfo();
        } else {
            // Fallback: time-based frame counting (no audio)
            const frameInterval = 1000 / (state.fps * state.speed);
            if (now - state.lastFrameTime >= frameInterval) {
                state.currentFrame++;
                if (state.currentFrame >= state.activeSequence.frames) {
                    state.currentFrame = 0;
                }
                const weights = getFrameWeights(state.currentFrame);
                if (weights) applyWeights(weights);
                updateFrameInfo();
                state.lastFrameTime = now;
            }
        }
    }

    state.controls.update();
    state.renderer.render(state.scene, state.camera);
}

// ============================================================
// Event Bindings
// ============================================================
function setupEvents() {
    // Mesh file loading
    const dropZone = document.getElementById('drop-zone');
    const meshInput = document.getElementById('mesh-file-input');

    dropZone.addEventListener('click', () => meshInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) file.arrayBuffer().then(buf => loadFlameBinary(buf));
    });
    meshInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (file) file.arrayBuffer().then(buf => loadFlameBinary(buf));
    });

    // Sequence file loading
    const seqDrop = document.getElementById('drop-zone-seq');
    const seqInput = document.getElementById('seq-file-input');

    seqDrop.addEventListener('click', () => seqInput.click());
    seqDrop.addEventListener('dragover', e => { e.preventDefault(); seqDrop.style.borderColor = '#4a90d9'; });
    seqDrop.addEventListener('dragleave', () => seqDrop.style.borderColor = '#3a3a5a');
    seqDrop.addEventListener('drop', e => {
        e.preventDefault();
        seqDrop.style.borderColor = '#3a3a5a';
        const file = e.dataTransfer.files[0];
        if (file) file.arrayBuffer().then(buf => loadSequenceFromBuffer(file.name.replace('.npy', ''), buf));
    });
    seqInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (file) file.arrayBuffer().then(buf => loadSequenceFromBuffer(file.name.replace('.npy', ''), buf));
    });

    // Playback controls
    document.getElementById('btn-play').addEventListener('click', () => {
        state.playing = true;
        state.lastFrameTime = performance.now();
        startAudio();
    });
    document.getElementById('btn-pause').addEventListener('click', () => {
        state.playing = false;
        pauseAudio();
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
        state.currentFrame = 0;
        state.playing = false;
        const weights = getFrameWeights(0);
        if (weights) applyWeights(weights);
        updateFrameInfo();
        resetAudio();
    });

    document.getElementById('timeline').addEventListener('input', e => {
        state.currentFrame = parseInt(e.target.value);
        const weights = getFrameWeights(state.currentFrame);
        if (weights) applyWeights(weights);
        updateFrameInfo();
        syncAudioToFrame();
    });

    document.getElementById('speed-slider').addEventListener('input', e => {
        state.speed = parseFloat(e.target.value);
        document.getElementById('speed-value').textContent = state.speed.toFixed(1) + 'x';
        if (state.audio) state.audio.playbackRate = state.speed;
    });

    document.getElementById('volume-slider').addEventListener('input', e => {
        const vol = parseFloat(e.target.value);
        document.getElementById('volume-value').textContent = Math.round(vol * 100) + '%';
        if (state.audio) state.audio.volume = vol;
    });

    // View mode buttons
    document.getElementById('btn-pred').addEventListener('click', () => setViewMode('prediction'));
    document.getElementById('btn-gt').addEventListener('click', () => setViewMode('ground_truth'));
    document.getElementById('btn-overlay').addEventListener('click', () => setViewMode('overlay'));

    // WebSocket
    document.getElementById('btn-ws-connect').addEventListener('click', wsConnect);
    document.getElementById('btn-ws-disconnect').addEventListener('click', wsDisconnect);

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT') return;
        if (e.code === 'Space') {
            e.preventDefault();
            state.playing = !state.playing;
            state.lastFrameTime = performance.now();
            if (state.playing) startAudio(); else pauseAudio();
        } else if (e.code === 'ArrowRight') {
            state.currentFrame = Math.min(state.currentFrame + 1, (state.activeSequence?.frames || 1) - 1);
            const weights = getFrameWeights(state.currentFrame);
            if (weights) applyWeights(weights);
            updateFrameInfo();
            syncAudioToFrame();
        } else if (e.code === 'ArrowLeft') {
            state.currentFrame = Math.max(state.currentFrame - 1, 0);
            const weights = getFrameWeights(state.currentFrame);
            if (weights) applyWeights(weights);
            updateFrameInfo();
            syncAudioToFrame();
        } else if (e.code === 'KeyR') {
            state.currentFrame = 0;
            state.playing = false;
            const weights = getFrameWeights(0);
            if (weights) applyWeights(weights);
            updateFrameInfo();
            resetAudio();
        } else if (e.code === 'Digit1') {
            setViewMode('prediction');
        } else if (e.code === 'Digit2') {
            if (state.activeGT) setViewMode('ground_truth');
        } else if (e.code === 'Digit3') {
            if (state.activeGT) setViewMode('overlay');
        }
    });
}

// ============================================================
// Try auto-loading from server
// ============================================================
async function tryAutoLoad() {
    // Try loading flame_data.bin from static
    try {
        const resp = await fetch('/static/flame_data.bin');
        if (resp.ok) {
            const buf = await resp.arrayBuffer();
            loadFlameBinary(buf);
        }
    } catch (e) { /* Not available, user must load manually */ }

    // Try listing sequences from server API
    try {
        const resp = await fetch('/api/sequences');
        if (resp.ok) {
            const data = await resp.json();
            for (const seq of data.sequences) {
                const seqResp = await fetch(`/api/sequences/${seq.filename}`);
                if (seqResp.ok) {
                    const buf = await seqResp.arrayBuffer();
                    const header = new DataView(buf, 0, 8);
                    const rows = header.getUint32(0, true);
                    const cols = header.getUint32(4, true);
                    const floatData = new Float32Array(buf, 8);
                    state.sequences.push({
                        name: seq.name,
                        data: floatData,
                        frames: rows,
                        dims: cols,
                        has_gt: seq.has_gt || false,
                        gt_filename: seq.gt_filename || null,
                        has_audio: seq.has_audio || false,
                        audio_filename: seq.audio_filename || null,
                    });
                }
            }
            if (state.sequences.length > 0) {
                renderSequenceList();
                selectSequence(state.sequences[0].name);
            }
        }
    } catch (e) { /* Server API not available */ }
}

// ============================================================
// Init
// ============================================================
initThreeJS();
setupEvents();
animate();
tryAutoLoad();
