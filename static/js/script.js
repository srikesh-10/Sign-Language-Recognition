/**
 * ==========================================================================
 * SignLang AI — Frontend Controller
 * ==========================================================================
 * Handles real-time state polling, UI updates, keyboard shortcuts,
 * and API interactions with the Flask backend.
 * ==========================================================================
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    const POLL_INTERVAL = 150;         // ms between state polls
    const TOAST_DURATION = 3000;       // ms to show toast notifications
    const ACTION_COOLDOWN = 400;       // ms between repeated actions

    // -----------------------------------------------------------------------
    // DOM References
    // -----------------------------------------------------------------------
    const dom = {
        // Navigation
        navStatus: document.getElementById('navStatus'),
        navStatusText: document.getElementById('navStatusText'),

        // Camera
        cameraContainer: document.getElementById('cameraContainer'),
        cameraFeed: document.getElementById('cameraFeed'),
        handStatusLabel: document.getElementById('handStatusLabel'),
        trackingBadge: document.getElementById('trackingBadge'),
        trackingText: document.getElementById('trackingText'),
        fpsBadge: document.getElementById('fpsBadge'),
        fpsValue: document.getElementById('fpsValue'),

        // Prediction
        predictionLetter: document.getElementById('predictionLetter'),
        confidenceValue: document.getElementById('confidenceValue'),
        confidenceFill: document.getElementById('confidenceFill'),
        stabilityValue: document.getElementById('stabilityValue'),
        fpsMetaValue: document.getElementById('fpsMetaValue'),
        cameraStatusValue: document.getElementById('cameraStatusValue'),

        // Stats
        durationLabel: document.getElementById('durationLabel'),
        acceptedCount: document.getElementById('acceptedCount'),
        detectionQuality: document.getElementById('detectionQuality'),
        avgConfidence: document.getElementById('avgConfidence'),
        feedStatus: document.getElementById('feedStatus'),

        // Sentence
        sentenceText: document.getElementById('sentenceText'),
        sentencePlaceholder: document.getElementById('sentencePlaceholder'),
        sentenceDisplay: document.getElementById('sentenceDisplay'),

        // Buttons
        btnAppend: document.getElementById('btnAppend'),
        btnSpace: document.getElementById('btnSpace'),
        btnDelete: document.getElementById('btnDelete'),
        btnClear: document.getElementById('btnClear'),
        btnSpeak: document.getElementById('btnSpeak'),
        btnSave: document.getElementById('btnSave'),

        // Toast
        toastContainer: document.getElementById('toastContainer'),
    };

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let lastActionTime = 0;
    let pollTimer = null;
    let isConnected = false;

    // -----------------------------------------------------------------------
    // API Helpers
    // -----------------------------------------------------------------------
    async function apiGet(endpoint) {
        const res = await fetch(endpoint);
        return res.json();
    }

    async function apiPost(endpoint) {
        const res = await fetch(endpoint, { method: 'POST' });
        return res.json();
    }

    function canAct() {
        const now = Date.now();
        if (now - lastActionTime < ACTION_COOLDOWN) return false;
        lastActionTime = now;
        return true;
    }

    // -----------------------------------------------------------------------
    // Toast Notifications
    // -----------------------------------------------------------------------
    function showToast(message, type = 'info') {
        const iconMap = {
            success: 'fa-solid fa-circle-check',
            error: 'fa-solid fa-circle-xmark',
            info: 'fa-solid fa-circle-info',
        };

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<i class="${iconMap[type] || iconMap.info}" aria-hidden="true"></i><span>${message}</span>`;
        dom.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'toast-out 0.3s ease forwards';
            toast.addEventListener('animationend', () => toast.remove());
        }, TOAST_DURATION);
    }

    // -----------------------------------------------------------------------
    // UI Update Functions
    // -----------------------------------------------------------------------

    function updatePrediction(data) {
        const pred = data.prediction;
        const conf = data.confidence;
        const hand = data.hand_detected;

        // Letter
        if (hand && pred && pred !== '—') {
            dom.predictionLetter.textContent = pred;
            dom.predictionLetter.className = 'prediction-letter active';
        } else if (hand) {
            dom.predictionLetter.textContent = '?';
            dom.predictionLetter.className = 'prediction-letter';
        } else {
            dom.predictionLetter.textContent = 'Waiting…';
            dom.predictionLetter.className = 'prediction-letter waiting';
        }

        // Confidence bar
        dom.confidenceValue.textContent = hand ? `${conf}%` : '—';
        dom.confidenceFill.style.width = hand ? `${conf}%` : '0%';

        // Confidence color
        dom.confidenceFill.classList.remove('high', 'medium', 'low');
        if (conf >= 80) dom.confidenceFill.classList.add('high');
        else if (conf >= 50) dom.confidenceFill.classList.add('medium');
        else dom.confidenceFill.classList.add('low');
    }

    function updateTracking(data) {
        const tracking = data.tracking;
        const hand = data.hand_detected;

        // Camera container active border
        if (hand) {
            dom.cameraContainer.classList.add('active');
        } else {
            dom.cameraContainer.classList.remove('active');
        }

        // Tracking badge
        const trackingMap = {
            GOOD: { text: 'GOOD', css: 'good', icon: 'fa-hand' },
            MODERATE: { text: 'MODERATE', css: 'moderate', icon: 'fa-hand' },
            POOR: { text: 'POOR', css: 'poor', icon: 'fa-hand' },
            NONE: { text: 'NO HAND', css: 'none', icon: 'fa-hand' },
        };
        const t = trackingMap[tracking] || trackingMap.NONE;
        dom.trackingBadge.className = `tracking-badge ${t.css}`;
        dom.trackingText.textContent = t.text;

        // Hand status label in panel header
        dom.handStatusLabel.textContent = hand ? 'Detected' : 'Waiting';
        dom.handStatusLabel.style.color = hand ? 'var(--success)' : 'var(--text-tertiary)';

        // Stability meta
        dom.stabilityValue.textContent = tracking;
        dom.stabilityValue.className = `meta-value ${t.css}`;
    }

    function updateStats(data) {
        dom.fpsValue.textContent = data.fps;
        dom.fpsMetaValue.textContent = data.fps;
        dom.durationLabel.textContent = data.duration;
        dom.acceptedCount.textContent = data.accepted;
        dom.avgConfidence.textContent = `${data.avg_confidence}%`;
        dom.feedStatus.textContent = data.camera_active ? 'Active' : 'Inactive';
        dom.cameraStatusValue.textContent = data.camera_active ? 'ACTIVE' : 'LOADING';
        dom.cameraStatusValue.style.color = data.camera_active ? 'var(--success)' : 'var(--warning)';
        dom.navStatus.className = `nav-status ${data.camera_active ? 'is-live' : 'is-connecting'}`;
        dom.navStatusText.textContent = data.camera_active ? 'Camera live' : 'Starting camera';

        // Detection quality pill
        const avg = data.avg_confidence;
        const pill = dom.detectionQuality;
        if (avg >= 85) {
            pill.textContent = 'High';
            pill.className = 'quality-pill high';
        } else if (avg >= 60) {
            pill.textContent = 'Medium';
            pill.className = 'quality-pill medium';
        } else if (avg > 0) {
            pill.textContent = 'Low';
            pill.className = 'quality-pill low';
        } else {
            pill.textContent = 'N/A';
            pill.className = 'quality-pill na';
        }
    }

    function updateSentence(sentence) {
        dom.sentenceText.textContent = sentence || '';
        dom.sentencePlaceholder.hidden = Boolean(sentence);
        if (sentence) {
            dom.sentenceDisplay.classList.add('active');
        } else {
            dom.sentenceDisplay.classList.remove('active');
        }
    }

    function updateSpeakButton(isSpeaking) {
        if (isSpeaking) {
            dom.btnSpeak.classList.add('speaking');
            dom.btnSpeak.querySelector('i').className = 'fa-solid fa-waveform';
        } else {
            dom.btnSpeak.classList.remove('speaking');
            dom.btnSpeak.querySelector('i').className = 'fa-solid fa-volume-high';
        }
    }

    // -----------------------------------------------------------------------
    // State Polling
    // -----------------------------------------------------------------------
    async function pollState() {
        try {
            const data = await apiGet('/api/state');
            if (!isConnected) {
                isConnected = true;
            }

            updatePrediction(data);
            updateTracking(data);
            updateStats(data);
            updateSentence(data.sentence);
            updateSpeakButton(data.is_speaking);

        } catch (err) {
            if (isConnected) {
                isConnected = false;
            }
            dom.navStatus.className = 'nav-status is-offline';
            dom.navStatusText.textContent = 'Disconnected';
        }
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------
    async function appendChar() {
        if (!canAct()) return;
        try {
            const data = await apiPost('/api/append');
            if (data.ok) {
                showToast(`Added "${data.char}"`, 'success');
            } else {
                showToast(data.error || 'Cannot add character', 'error');
            }
        } catch { showToast('Connection error', 'error'); }
    }

    async function addSpace() {
        if (!canAct()) return;
        try {
            await apiPost('/api/space');
        } catch { showToast('Connection error', 'error'); }
    }

    async function deleteChar() {
        if (!canAct()) return;
        try {
            const data = await apiPost('/api/delete');
            if (!data.ok) showToast('Sentence is empty', 'info');
        } catch { showToast('Connection error', 'error'); }
    }

    async function clearSentence() {
        if (!canAct()) return;
        try {
            const data = await apiPost('/api/clear');
            if (data.ok) showToast('Sentence cleared', 'info');
        } catch { showToast('Connection error', 'error'); }
    }

    async function speakSentence() {
        if (!canAct()) return;
        try {
            const data = await apiPost('/api/speak');
            if (data.ok) {
                showToast('Speaking…', 'info');
            } else {
                showToast(data.error || 'Cannot speak', 'error');
            }
        } catch { showToast('Connection error', 'error'); }
    }

    async function saveSentence() {
        if (!canAct()) return;
        try {
            const data = await apiPost('/api/save');
            if (data.ok) {
                showToast(`Saved as ${data.filename}`, 'success');
            } else {
                showToast(data.error || 'Save failed', 'error');
            }
        } catch { showToast('Connection error', 'error'); }
    }

    // -----------------------------------------------------------------------
    // Event Listeners
    // -----------------------------------------------------------------------

    // Button clicks
    dom.btnAppend.addEventListener('click', appendChar);
    dom.btnSpace.addEventListener('click', addSpace);
    dom.btnDelete.addEventListener('click', deleteChar);
    dom.btnClear.addEventListener('click', clearSentence);
    dom.btnSpeak.addEventListener('click', speakSentence);
    dom.btnSave.addEventListener('click', saveSentence);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Don't trigger shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                appendChar();
                break;
            case 'Backspace':
                e.preventDefault();
                deleteChar();
                break;
            case 'Enter':
                e.preventDefault();
                speakSentence();
                break;
            case 'KeyC':
                if (!e.ctrlKey && !e.metaKey) {
                    e.preventDefault();
                    clearSentence();
                }
                break;
            case 'KeyS':
                if (!e.ctrlKey && !e.metaKey) {
                    e.preventDefault();
                    saveSentence();
                }
                break;
        }
    });

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------
    function init() {
        // Start polling
        pollState();
        pollTimer = setInterval(pollState, POLL_INTERVAL);

        // Handle camera feed errors
        dom.cameraFeed.addEventListener('error', () => {
            console.warn('[SignLang AI] Camera feed connection error. Retrying...');
            setTimeout(() => {
                dom.cameraFeed.src = '/video_feed?' + Date.now();
            }, 2000);
        });

        console.log('[SignLang AI] Dashboard initialized.');
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
