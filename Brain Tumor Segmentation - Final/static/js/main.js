// NeuroSeg AI Frontend Logic
document.addEventListener('DOMContentLoaded', () => {
    // 1. Application State
    const state = {
        volumeId: null,
        sliceIdx: 0,
        modality: 'flair',
        isPlaying: false,
        playInterval: null,
        fps: 20,
        stats: null,
        modelResults: null,
        opacities: {
            edema: 0.7,
            core: 0.7,
            enhancing: 0.8
        },
        metadata: {
            slicesCount: 100,
            startSlice: 22
        }
    };

    // 2. DOM Elements
    const elements = {
        sampleTab: document.getElementById('tab-sample'),
        uploadTab: document.getElementById('tab-upload'),
        sampleContent: document.getElementById('content-sample'),
        uploadContent: document.getElementById('content-upload'),
        sampleSelector: document.getElementById('sample-selector'),
        loadSampleBtn: document.getElementById('load-sample-btn'),
        uploadForm: document.getElementById('upload-form'),
        fileFlair: document.getElementById('file-flair'),
        fileT1: document.getElementById('file-t1'),
        fileT1ce: document.getElementById('file-t1ce'),
        fileT2: document.getElementById('file-t2'),
        labelFlair: document.getElementById('label-flair'),
        labelT1: document.getElementById('label-t1'),
        labelT1ce: document.getElementById('label-t1ce'),
        labelT2: document.getElementById('label-t2'),
        uploadBtn: document.getElementById('upload-btn'),
        
        sliceSlider: document.getElementById('slice-slider'),
        currentSliceNum: document.getElementById('current-slice-num'),
        playBtn: document.getElementById('play-btn'),
        stopBtn: document.getElementById('stop-btn'),
        playbackFps: document.getElementById('playback-fps'),
        
        modalityButtons: document.querySelectorAll('[data-modality]'),
        
        opacityEdema: document.getElementById('opacity-edema'),
        opacityCore: document.getElementById('opacity-core'),
        opacityEnhancing: document.getElementById('opacity-enhancing'),
        
        voxelCoords: document.getElementById('voxel-coords'),
        viewportLoader: document.getElementById('viewport-loader'),
        
        sliceImagePred: document.getElementById('slice-image-pred'),
        canvasMaskPred: document.getElementById('canvas-mask-pred'),
        
        sliceImageGt: document.getElementById('slice-image-gt'),
        canvasMaskGt: document.getElementById('canvas-mask-gt'),
        gtViewportContainer: document.getElementById('gt-viewport-container'),
        noGtWatermark: document.getElementById('no-gt-watermark'),
        
        volWt: document.getElementById('vol-wt'),
        volTc: document.getElementById('vol-tc'),
        volEt: document.getElementById('vol-et'),
        gtRefWt: document.getElementById('gt-ref-wt'),
        gtRefTc: document.getElementById('gt-ref-tc'),
        gtRefEt: document.getElementById('gt-ref-et'),
        diagnosticText: document.getElementById('diagnostic-text')
    };

    let statsChart = null;

    // 3. Setup UI Interaction
    
    // Tab switching
    elements.sampleTab.addEventListener('click', () => {
        elements.sampleTab.classList.add('active');
        elements.uploadTab.classList.remove('active');
        elements.sampleContent.classList.remove('hidden');
        elements.uploadContent.classList.add('hidden');
    });

    elements.uploadTab.addEventListener('click', () => {
        elements.uploadTab.classList.add('active');
        elements.sampleTab.classList.remove('active');
        elements.uploadContent.classList.remove('hidden');
        elements.sampleContent.classList.add('hidden');
    });

    // File selection label updates
    elements.fileFlair.addEventListener('change', (e) => {
        const file = e.target.files[0];
        elements.labelFlair.textContent = file ? file.name : 'No file selected';
    });

    elements.fileT1.addEventListener('change', (e) => {
        const file = e.target.files[0];
        elements.labelT1.textContent = file ? file.name : 'No file selected';
    });

    elements.fileT1ce.addEventListener('change', (e) => {
        const file = e.target.files[0];
        elements.labelT1ce.textContent = file ? file.name : 'No file selected';
    });

    elements.fileT2.addEventListener('change', (e) => {
        const file = e.target.files[0];
        elements.labelT2.textContent = file ? file.name : 'No file selected';
    });


    // Modality channel buttons
    elements.modalityButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            elements.modalityButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.modality = btn.dataset.modality;
            if (state.volumeId) {
                renderCurrentSlice();
            }
        });
    });

    // Opacity Sliders
    elements.opacityEdema.addEventListener('input', (e) => {
        state.opacities.edema = parseFloat(e.target.value);
        redrawOverlays();
    });
    elements.opacityCore.addEventListener('input', (e) => {
        state.opacities.core = parseFloat(e.target.value);
        redrawOverlays();
    });
    elements.opacityEnhancing.addEventListener('input', (e) => {
        state.opacities.enhancing = parseFloat(e.target.value);
        redrawOverlays();
    });

    // Slice navigation slider
    elements.sliceSlider.addEventListener('input', (e) => {
        state.sliceIdx = parseInt(e.target.value) - state.metadata.startSlice;
        elements.currentSliceNum.textContent = e.target.value;
        renderCurrentSlice();
    });

    // Playback Controls
    elements.playbackFps.addEventListener('change', (e) => {
        state.fps = parseInt(e.target.value);
        if (state.isPlaying) {
            pauseSlices();
            playSlices();
        }
    });

    elements.playBtn.addEventListener('click', () => {
        if (state.isPlaying) {
            pauseSlices();
        } else {
            playSlices();
        }
    });

    elements.stopBtn.addEventListener('click', () => {
        pauseSlices();
        state.sliceIdx = 0;
        elements.sliceSlider.value = state.metadata.startSlice;
        elements.currentSliceNum.textContent = state.metadata.startSlice;
        renderCurrentSlice();
    });

    function playSlices() {
        state.isPlaying = true;
        elements.playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
        state.playInterval = setInterval(() => {
            state.sliceIdx = (state.sliceIdx + 1) % state.metadata.slicesCount;
            const currentSlice = state.sliceIdx + state.metadata.startSlice;
            elements.sliceSlider.value = currentSlice;
            elements.currentSliceNum.textContent = currentSlice;
            renderCurrentSlice();
        }, 1000 / state.fps);
    }

    function pauseSlices() {
        state.isPlaying = false;
        elements.playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        clearInterval(state.playInterval);
    }

    // Load preloaded sample button
    elements.loadSampleBtn.addEventListener('click', () => {
        const selectedVolume = elements.sampleSelector.value;
        if (!selectedVolume) {
            alert('Please select a sample patient volume first.');
            return;
        }
        loadVolume(`/api/load_sample/${selectedVolume}`);
    });

    // Upload custom scans
    elements.uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const flairFile = elements.fileFlair.files[0];
        const t1File = elements.fileT1.files[0];
        const t1ceFile = elements.fileT1ce.files[0];
        const t2File = elements.fileT2.files[0];
        
        if (!flairFile || !t1File || !t1ceFile || !t2File) {
            alert('Attention U-Net and HRNet-OCR require all 4 modalities: FLAIR, T1, T1ce, and T2 files (.nii or .nii.gz).');
            return;
        }

        const formData = new FormData();
        formData.append('flair', flairFile);
        formData.append('t1', t1File);
        formData.append('t1ce', t1ceFile);
        formData.append('t2', t2File);

        loadVolume('/api/upload', {
            method: 'POST',
            body: formData
        });
    });

    // 4. Volume Loader Logic
    async function loadVolume(url, options = {}) {
        elements.viewportLoader.classList.remove('hidden');
        pauseSlices();
        disableControls();

        try {
            const response = await fetch(url, options);
            const result = await response.json();
            
            if (result.success) {
                state.volumeId = result.volume_id;
                state.metadata.slicesCount = result.slices_count;
                state.metadata.startSlice = result.start_slice;
                state.stats = result.stats;
                // This is deliberately model-specific. The UI never reuses one
                // prediction as the result of the other model.
                state.modelResults = result.model_results || null;
                // Start on an actual tumor-containing slice when one exists,
                // instead of showing an empty first slice after inference.
                state.sliceIdx = Number.isInteger(result.recommended_slice) ? result.recommended_slice : 0;
                
                // Determine Ground Truth availability
                state.hasGt = (result.has_gt !== undefined) ? result.has_gt : (result.stats && result.stats.gt_totals !== null);

                // Update slider limits
                elements.sliceSlider.min = state.metadata.startSlice;
                elements.sliceSlider.max = state.metadata.startSlice + state.metadata.slicesCount - 1;
                const initialSlice = state.metadata.startSlice + state.sliceIdx;
                elements.sliceSlider.value = initialSlice;
                elements.currentSliceNum.textContent = initialSlice;

                // Toggle expert label watermarks & visibility
                if (state.hasGt) {
                    elements.noGtWatermark.classList.add('hidden');
                    elements.sliceImageGt.style.opacity = '1';
                    elements.canvasMaskGt.classList.remove('hidden');
                } else {
                    elements.noGtWatermark.classList.remove('hidden');
                    elements.sliceImageGt.style.opacity = '0.5';
                    elements.canvasMaskGt.classList.add('hidden');
                }

                // Update text stats
                updateStatsUI();
                
                // Initialize analytics chart
                updateChart();
                renderModelResults();

                enableControls();
                renderCurrentSlice();
            } else {
                alert(`Error loading volume: ${result.error}`);
            }
        } catch (err) {
            console.error(err);
            alert(`Network / server error occurred: ${err.message}`);
        } finally {
            elements.viewportLoader.classList.add('hidden');
        }
    }

    function disableControls() {
        elements.sliceSlider.disabled = true;
        elements.playBtn.disabled = true;
        elements.stopBtn.disabled = true;
        elements.loadSampleBtn.disabled = true;
        elements.uploadBtn.disabled = true;
    }

    function enableControls() {
        elements.sliceSlider.disabled = false;
        elements.playBtn.disabled = false;
        elements.stopBtn.disabled = false;
        elements.loadSampleBtn.disabled = false;
        elements.uploadBtn.disabled = false;
    }

    // 5. Slice Rendering Logic
    // Store mask image objects in memory to prevent double fetches on draw
    let activePredMaskImg = new Image();
    let activeGtMaskImg = new Image();
    
    function renderCurrentSlice() {
        const sliceUrl = `/api/slice?volume_id=${state.volumeId}&slice_idx=${state.sliceIdx}&type=${state.modality}`;
        
        // Update raw MRI slice elements for both viewports
        elements.sliceImagePred.src = sliceUrl;
        elements.sliceImageGt.src = sliceUrl;

        // Fetch and load transparent overlays
        activePredMaskImg = new Image();
        activePredMaskImg.onload = () => {
            drawMaskOverlay(elements.canvasMaskPred, activePredMaskImg);
        };
        activePredMaskImg.src = `/api/slice?volume_id=${state.volumeId}&slice_idx=${state.sliceIdx}&type=pred`;

        if (state.hasGt) {
            activeGtMaskImg = new Image();
            activeGtMaskImg.onload = () => {
                drawMaskOverlay(elements.canvasMaskGt, activeGtMaskImg);
            };
            activeGtMaskImg.src = `/api/slice?volume_id=${state.volumeId}&slice_idx=${state.sliceIdx}&type=gt`;
        }

        // Highlight current slice on chart index line
        if (statsChart) {
            statsChart.update('none'); // Update without animation
        }
        // Keep each independently generated mask and overlay synchronized with
        // the MRI slice navigator.
        renderModelResults();
    }

    const METRICS = [
        ['pixel_accuracy', 'Pixel Accuracy', true],
        ['mean_dice', 'Mean Dice', true],
        ['tumor_dice', 'Tumor-only Dice', true],
        ['iou', 'IoU', true]
    ];

    const MODELS = [
        { key: 'attention_unet', name: 'Attention U-Net', accent: 'cyan' },
        { key: 'hrnet_ocr', name: 'HRNet-OCR', accent: 'purple' }
    ];

    function metricValue(value) {
        if (!isMetricNumber(value)) return 'Not available';
        return Number(value).toFixed(4);
    }

    function isMetricNumber(value) {
        return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
    }

    function modelImageUrl(modelKey, type, sliceIndex = state.sliceIdx) {
        return `/api/slice?volume_id=${encodeURIComponent(state.volumeId)}&slice_idx=${sliceIndex}&type=${type}&model=${modelKey}`;
    }

    function renderModelResults() {
        const panels = document.getElementById('model-result-panels');
        const comparison = document.getElementById('comparison-table-body');
        if (!panels || !comparison) return;

        const models = state.modelResults || {};
        panels.innerHTML = MODELS.map(model => {
            const result = models[model.key];
            const metrics = result && result.metrics ? result.metrics : {};
            const hasInference = result && result.available !== false;
            
            // Disable the demo visual fallback so the actual masks are always shown
            const showHrnetDemo = false;
            
            const modelSlice = Number.isInteger(result && result.recommended_slice) ? result.recommended_slice : state.sliceIdx;
            const source = state.volumeId ? modelImageUrl(model.key, state.modality, modelSlice) : '';
            const mask = state.volumeId && hasInference ? modelImageUrl(model.key, 'pred', modelSlice) : '';
            const demoVisual = ``;
            return `
                <article class="model-result-card ${model.accent}">
                    <header class="model-result-header">
                        <div><span class="model-tag">Independent inference</span><h2>${model.name}</h2></div>
                        <span class="detection-status ${result && result.tumor_detected ? 'detected' : 'pending'}">${hasInference ? (result.tumor_detected ? 'Tumor detected' : 'No tumor detected') : 'Model unavailable'}</span>
                    </header>
                    <p class="model-message">${hasInference ? `Original MRI → predicted tumor mask → segmentation overlay (strongest predicted slice: ${modelSlice + state.metadata.startSlice})` : (result && result.message ? result.message : 'Separate model output has not been returned by the inference service.')}</p>
                    <div class="model-image-grid">
                        <figure><figcaption>Original MRI / FLAIR</figcaption>${source ? `<img src="${source}" alt="Original MRI used by ${model.name}">` : '<div class="image-placeholder">Upload an MRI to begin</div>'}</figure>
                        <figure><figcaption>${model.name} Predicted Tumor Mask</figcaption>${mask ? `<img src="${mask}" alt="${model.name} predicted tumor mask">` : '<div class="image-placeholder">Awaiting model output</div>'}</figure>
                        <figure><figcaption>${model.name} Segmentation Overlay</figcaption>${source && mask ? `<div class="overlay-preview"><img src="${source}" alt="MRI"><img class="mask-layer" src="${mask}" alt="${model.name} overlay"></div>` : '<div class="image-placeholder">Awaiting model output</div>'}</figure>
                    </div>
                    <div class="metric-grid">${METRICS.map(([key, label]) => `<div class="metric-item"><span>${label}</span><strong>${metricValue(metrics[key])}</strong></div>`).join('')}</div>
                </article>`;
        }).join('');

        comparison.innerHTML = METRICS.map(([key, label, higherIsBetter]) => {
            const first = models.attention_unet && models.attention_unet.metrics ? models.attention_unet.metrics[key] : null;
            const second = models.hrnet_ocr && models.hrnet_ocr.metrics ? models.hrnet_ocr.metrics[key] : null;
            const valid = isMetricNumber(first) && isMetricNumber(second);
            let better = 'Awaiting evaluation';
            let attentionClass = '';
            let hrnetClass = '';
            if (valid) {
                if (Number(first) === Number(second)) better = 'Tie';
                else {
                    const attentionWins = higherIsBetter ? Number(first) > Number(second) : Number(first) < Number(second);
                    better = attentionWins ? 'Attention U-Net' : 'HRNet-OCR';
                    attentionClass = attentionWins ? 'best-value' : '';
                    hrnetClass = attentionWins ? '' : 'best-value';
                }
            }
            return `<tr><th>${label}</th><td class="${attentionClass}">${metricValue(first)}</td><td class="${hrnetClass}">${metricValue(second)}</td><td>${better}</td></tr>`;
        }).join('');
    }

    renderModelResults();

    function drawMaskOverlay(canvas, maskImg) {
        const ctx = canvas.getContext('2d');
        canvas.width = 240;
        canvas.height = 240;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        
        // Apply client-side custom layer opacities
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i+1];
            const b = data[i+2];
            const a = data[i+3];
            
            if (a > 0) {
                if (r > 200 && g < 100 && b < 100) {
                    data[i+3] = Math.round(state.opacities.core * 255);
                }
                else if (r > 200 && g > 150 && b < 50) {
                    data[i+3] = Math.round(state.opacities.edema * 255);
                }
                else if (r < 100 && g > 100 && b > 200) {
                    data[i+3] = Math.round(state.opacities.enhancing * 255);
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    function redrawOverlays() {
        if (state.volumeId) {
            if (activePredMaskImg.complete) drawMaskOverlay(elements.canvasMaskPred, activePredMaskImg);
            if (state.hasGt && activeGtMaskImg.complete) drawMaskOverlay(elements.canvasMaskGt, activeGtMaskImg);
        }
    }

    // Coordinate inspection tracker
    elements.sliceImagePred.parentNode.addEventListener('mousemove', (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        // Scale pointer offsets to 240x240 voxel dimensions
        const x = Math.floor((e.clientX - rect.left) / rect.width * 240);
        const y = Math.floor((e.clientY - rect.top) / rect.height * 240);
        
        if (x >= 0 && x < 240 && y >= 0 && y < 240) {
            elements.voxelCoords.textContent = `X: ${x}, Y: ${y}`;
        }
    });

    elements.sliceImagePred.parentNode.addEventListener('mouseleave', () => {
        elements.voxelCoords.textContent = 'X: -, Y: -';
    });

    // 6. Statistics and Analytics Updates
    function updateStatsUI() {
        const totals = state.stats.totals;
        elements.volWt.textContent = totals.whole_tumor_volume;
        elements.volTc.textContent = totals.tumor_core_volume;
        elements.volEt.textContent = totals.enhancing_volume;

        const gt = state.stats.gt_totals;
        if (gt) {
            elements.gtRefWt.textContent = `GT: ${gt.whole_tumor_volume} cm³`;
            elements.gtRefTc.textContent = `GT: ${gt.tumor_core_volume} cm³`;
            elements.gtRefEt.textContent = `GT: ${gt.enhancing_volume} cm³`;
            
            elements.diagnosticText.innerHTML = `
                Expert reference volume is loaded. Estimated Whole Tumor: ${totals.whole_tumor_volume} cm³ 
                vs Expert label: ${gt.whole_tumor_volume} cm³.
            `;
        } else {
            elements.gtRefWt.textContent = 'GT: N/A';
            elements.gtRefTc.textContent = 'GT: N/A';
            elements.gtRefEt.textContent = 'GT: N/A';
            elements.diagnosticText.innerHTML = `
                Custom user volume is loaded. Pred WT: ${totals.whole_tumor_volume} cm³, TC: ${totals.tumor_core_volume} cm³.
            `;
        }
    }

    function updateChart() {
        const ctx = document.getElementById('tumor-distribution-chart').getContext('2d');
        const slices = state.stats.slices;
        
        const labels = slices.map(s => `Sl ${s.slice_idx}`);
        const wholeData = slices.map(s => s.whole_tumor_area);
        const coreData = slices.map(s => s.tumor_core_area);
        const enhancingData = slices.map(s => s.enhancing_area);

        if (statsChart) {
            statsChart.destroy();
        }

        // Draw vertical index line at current slice
        const currentSlicePlugin = {
            id: 'currentSliceLine',
            afterDraw: (chart) => {
                if (state.sliceIdx !== null) {
                    const ctx = chart.ctx;
                    const xAxis = chart.scales.x;
                    const yAxis = chart.scales.y;
                    const xLoc = xAxis.getPixelForValue(state.sliceIdx);
                    
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(xLoc, yAxis.top);
                    ctx.lineTo(xLoc, yAxis.bottom);
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#3b82f6';
                    ctx.setLineDash([4, 4]);
                    ctx.stroke();
                    ctx.restore();
                }
            }
        };

        statsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Whole Tumor Area',
                        data: wholeData,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.05)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Tumor Core Area',
                        data: coreData,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.05)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Enhancing Area',
                        data: enhancingData,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.05)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            boxWidth: 10,
                            font: { size: 9, family: 'Inter' }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: '#64748b',
                            maxTicksLimit: 10,
                            font: { size: 9 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Area (mm²)',
                            color: '#64748b',
                            font: { size: 9 }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#64748b', font: { size: 9 } }
                    }
                }
            },
            plugins: [currentSlicePlugin]
        });

        // Click chart to navigate to slice
        document.getElementById('tumor-distribution-chart').onclick = (evt) => {
            const points = statsChart.getElementsAtEventForMode(evt, 'nearest', { intersect: false }, true);
            if (points.length) {
                const clickIdx = points[0].index;
                state.sliceIdx = clickIdx;
                const currentSlice = state.sliceIdx + state.metadata.startSlice;
                elements.sliceSlider.value = currentSlice;
                elements.currentSliceNum.textContent = currentSlice;
                renderCurrentSlice();
            }
        };
    }
});
