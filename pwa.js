/**
 * Akademie KI Suite - PWA Logic
 * Handles Service Worker registration, Installation Prompts, and Offline states.
 */

document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Service Worker Registration ---
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/service-worker.js')
            .then(reg => console.log('✅ Service Worker registered with scope:', reg.scope))
            .catch(err => console.error('❌ Service Worker registration failed:', err));
    }

    // --- 2. Install Prompt Logic ---
    let deferredPrompt;
    const COOLDOWN_HOURS = 24; 

    window.addEventListener('beforeinstallprompt', (e) => {
        // Prevent Chrome 67 and earlier from automatically showing the prompt
        e.preventDefault();
        deferredPrompt = e;

        // Check if user dismissed it recently
        const lastDismissed = localStorage.getItem('pwa_dismissed_ts');
        if (lastDismissed) {
            const hoursSince = (Date.now() - parseInt(lastDismissed)) / (1000 * 60 * 60);
            if (hoursSince < COOLDOWN_HOURS) {
                console.log(`⏳ Install prompt suppressed (Cooldown: ${Math.round(hoursSince)}h / ${COOLDOWN_HOURS}h)`);
                return;
            }
        }

        showInstallPromotion();
    });

    function showInstallPromotion() {
        // Check if element already exists
        if (document.getElementById('pwa-install-prompt')) return;

        const promptDiv = document.createElement('div');
        promptDiv.id = 'pwa-install-prompt';
        promptDiv.className = 'install-prompt show'; // Uses your custom.css classes
        
        // Inline styles for the inner layout to ensure it looks good without editing CSS
        promptDiv.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 10px; align-items: center;">
                <span style="font-weight: bold; font-size: 1.1em;">📱 Akademie KI App</span>
                <span style="font-size: 0.9em; opacity: 0.9;">Für bessere Performance installieren</span>
                <div style="display: flex; gap: 10px; margin-top: 5px; width: 100%;">
                    <button id="btn-install" style="flex: 1; padding: 10px; background: white; color: var(--primary-color, #1976d2); border: none; border-radius: 4px; font-weight: bold; cursor: pointer;">
                        Installieren
                    </button>
                    <button id="btn-later" style="flex: 1; padding: 10px; background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.5); border-radius: 4px; cursor: pointer;">
                        Später
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(promptDiv);

        // Attach listeners safely
        document.getElementById('btn-install').addEventListener('click', async () => {
            promptDiv.remove();
            if (deferredPrompt) {
                deferredPrompt.prompt();
                const { outcome } = await deferredPrompt.userChoice;
                console.log(`User response to install prompt: ${outcome}`);
                deferredPrompt = null;
            }
        });

        document.getElementById('btn-later').addEventListener('click', () => {
            promptDiv.remove();
            // Set cooldown timestamp
            localStorage.setItem('pwa_dismissed_ts', Date.now().toString());
        });
    }

    // --- 3. Offline/Online Detection ---
    function updateOnlineStatus() {
        const isOnline = navigator.onLine;
        let indicator = document.querySelector('.offline-indicator');

        if (!isOnline) {
            if (!indicator) {
                indicator = document.createElement('div');
                indicator.className = 'offline-indicator';
                indicator.textContent = '📵 Offline - Verbindung unterbrochen';
                document.body.prepend(indicator);
            }
            indicator.classList.add('show');
        } else {
            if (indicator) indicator.classList.remove('show');
        }
    }

    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);
    // Check immediately on load
    updateOnlineStatus();

    // --- 4. Mobile Viewport Fixes ---
    function setVH() {
        let vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    setVH();
    window.addEventListener('resize', setVH);
    window.addEventListener('orientationchange', () => {
        setTimeout(setVH, 200); // Small delay for rotation to finish
    });
});
