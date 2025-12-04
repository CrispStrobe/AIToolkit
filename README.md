# Akademie KI Suite - Complete Deployment Guide

This documentation covers the deployment of a **GDPR-compliant Multi-Provider AI Suite** with:
- 💬 **Chat** (Multiple LLM providers)
- 🎙️ **Audio Transcription** (Gladia, Scaleway, Groq with Whisper)
- 👁️ **Vision Analysis** (Image understanding)
- 🎨 **Image Generation** (FLUX models via Nebius)
- 💾 **Database Integration** (SQLite with user authentication)
- 📱 **PWA Support** (Progressive Web App for mobile installation)
- 🛡️ **Security** (Fail2Ban protection)

The application runs as a Python/Gradio service on `localhost:7860`, managed by **systemd**, and exposed via **Apache reverse proxy** with SSL.

## 📋 Prerequisites

- **OS:** Ubuntu 20.04 LTS or newer
- **RAM:** Minimum 8GB (for handling audio files)
- **Root/Sudo Access**
- **Domain:** DNS A-Record pointing to server IP (e.g., `ki.akademie-rs.de`)
- **API Keys:** From providers (Scaleway, Nebius, etc.)

---

## 🛠️ Step 1: System Dependencies & Security

**CRITICAL:** FFmpeg must be in PATH for audio. Fail2Ban is installed for intrusion prevention.

```bash
sudo apt update
sudo apt upgrade -y
# Install Python, FFmpeg, Apache, Certbot, SQLite, and Fail2Ban
sudo apt install -y python3-venv python3-pip ffmpeg apache2 certbot python3-certbot-apache sqlite3 fail2ban
````

**Verify Installations:**

```bash
which ffmpeg
# Output: /usr/bin/ffmpeg

sudo systemctl status fail2ban
# Should be active (running)
```

-----

## 🐍 Step 2: Application Setup

### 2.1 Create Directory Structure

We use a `static` folder for all PWA assets to keep the root clean and permissions manageable.

```bash
sudo mkdir -p /var/www/transkript_app
sudo mkdir -p /var/www/transkript_app/static
sudo mkdir -p /var/www/transkript_app/generated_images
cd /var/www/transkript_app
```

### 2.2 Create Virtual Environment

```bash
sudo python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install Python Dependencies

```bash
pip install --upgrade pip
pip install \
    "openai>=1.0" \
    gradio==4.44.0 \
    requests \
    pillow \
    pypdf2 \
    sqlalchemy \
    bcrypt
```

### 2.4 Application Code (`app.py`)

When creating `app.py`, you **must** inject the PWA HTML head tags.

**PWA Injection (Insert before `with gr.Blocks...`):**

```python
# ==========================================
# 📱 PWA CONFIGURATION
# ==========================================
PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#1976d2">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Akademie KI">

<link rel="manifest" href="/manifest.json" crossorigin="use-credentials">
<link rel="icon" type="image/png" sizes="192x192" href="/static/icon-192.png">
<link rel="icon" type="image/png" sizes="512x512" href="/static/icon-512.png">
<link rel="apple-touch-icon" href="/static/icon-192.png">

<link rel="stylesheet" href="/static/custom.css">
<script src="/static/pwa.js" defer></script>
"""

# Pass it to Gradio:
with gr.Blocks(title="Akademie KI Suite", theme=gr.themes.Soft(), head=PWA_HEAD) as demo:
    # ... rest of app ...
```

### 2.5 Initialize Logs & DB Permissions

```bash
sudo touch /var/www/transkript_app/app.log
sudo chmod 666 /var/www/transkript_app/app.log
# Database created on first run
sudo chmod -R 755 /var/www/transkript_app
```

-----

## 🌐 Step 3: Apache Configuration (The Robust Method)

This configuration serves PWA files directly via Apache and proxies the rest to Python. This prevents 404 errors.

### 3.1 Enable Modules

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite headers ssl
```

### 3.2 SSL Setup

First, generate the certificate:

```bash
sudo certbot --apache -d ki.akademie-rs.de
```

### 3.3 Final Apache Config

Edit the SSL config created by Certbot:

```bash
sudo nano /etc/apache2/sites-available/transkript-le-ssl.conf
```

**Replace content with:**

```apache
<IfModule mod_ssl.c>
<VirtualHost *:443>
    ServerName ki.akademie-rs.de

    # =================================================
    # 1. STATIC FILES (Served by Apache)
    # =================================================

    # Serve the static folder
    Alias /static /var/www/transkript_app/static
    <Directory /var/www/transkript_app/static>
        Require all granted
        Options -Indexes
        AddType text/css .css
        AddType application/javascript .js
        AddType image/png .png
        Header set Cache-Control "public, max-age=31536000, immutable"
    </Directory>

    # Manifest (Map root URL -> static file)
    Alias /manifest.json /var/www/transkript_app/static/manifest.json
    <Files "manifest.json">
        Require all granted
        Header set Content-Type "application/manifest+json"
        Header set Cache-Control "no-cache"
    </Files>

    # Service Worker (Map root URL -> static file)
    Alias /service-worker.js /var/www/transkript_app/static/service-worker.js
    <Files "service-worker.js">
        Require all granted
        Header set Content-Type "application/javascript"
        Header set Cache-Control "no-cache"
        Header set Service-Worker-Allowed "/"
    </Files>

    # =================================================
    # 2. PROXY SETTINGS (Gradio)
    # =================================================
    
    ProxyPreserveHost On
    RewriteEngine On

    # Websockets (Required for Gradio queue)
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://127.0.0.1:7860/$1 [P,L]

    # EXCLUDE PWA files from Proxy (Apache handles them)
    ProxyPass /static !
    ProxyPass /manifest.json !
    ProxyPass /service-worker.js !

    # Proxy everything else to Python
    ProxyPass / [http://127.0.0.1:7860/](http://127.0.0.1:7860/)
    ProxyPassReverse / [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

    # =================================================
    # 3. SSL & LIMITS
    # =================================================
    SSLCertificateFile /etc/letsencrypt/live/ki.akademie-rs.de/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/ki.akademie-rs.de/privkey.pem
    Include /etc/letsencrypt/options-ssl-apache.conf

    # Increase limits for large audio uploads
    LimitRequestBody 1048576000
    ProxyTimeout 600
    TimeOut 600
</VirtualHost>
</IfModule>
```

-----

## ⚙️ Step 4: Systemd Service

```bash
sudo nano /etc/systemd/system/transkript.service
```

```ini
[Unit]
Description=Gradio Akademie KI Suite
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/transkript_app
Environment="PATH=/var/www/transkript_app/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/var/www/transkript_app/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable transkript
sudo systemctl restart transkript
```

-----

## 📱 Step 5: PWA Files Setup

All PWA files must reside in `/var/www/transkript_app/static/`.

### 5.1 Directory Permissions

**Crucial:** Apache (`www-data`) needs ownership of static files.

```bash
sudo chown -R www-data:www-data /var/www/transkript_app/static
sudo chmod -R 755 /var/www/transkript_app/static
```

### 5.2 `manifest.json` (in /static/)

Note: Icons point to `/static/`.

```json
{
  "name": "Akademie KI Suite Ultimate",
  "short_name": "Akademie KI",
  "start_url": "/?source=pwa",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#1976d2",
  "icons": [
    {
      "src": "/static/icon-192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/static/icon-512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ]
}
```

### 5.3 `service-worker.js` (in /static/)

```javascript
const CACHE_NAME = 'akademie-ki-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/static/custom.css',
  '/static/pwa.js',
  '/manifest.json',
  '/static/icon-192.png'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS_TO_CACHE))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
```

-----

## 🛡️ Step 6: Fail2Ban Configuration

Fail2Ban is installed, but we can add a custom jail for Apache to block repeated 404/403 errors or bot scanners.

1.  Create a local jail config:

<!-- end list -->

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

2.  Ensure `[sshd]` is enabled.
3.  Optionally enable `[apache-auth]` or `[apache-badbots]`.

Restart to apply:

```bash
sudo systemctl restart fail2ban
```

-----

## 📁 Final File Structure

```
/var/www/transkript_app/
├── app.py                      # Main App (with PWA_HEAD)
├── akademie_suite.db           # Database
├── venv/                       # Virtual Environment
└── static/                     # Owned by www-data
    ├── custom.css
    ├── pwa.js
    ├── manifest.json
    ├── service-worker.js
    ├── icon-192.png
    └── icon-512.png
```

-----

## 🔍 Troubleshooting PWA

If "Install App" does not appear:

1.  **Clear Mobile Cache:** Browsers cache 404 errors aggressively.
2.  **Check Console:** Look for Red 404s.
3.  **Verify HTTPS:** PWA requires valid SSL.
4.  **Test URLs:**
      * `https://ki.akademie-rs.de/manifest.json` (Should load JSON)
      * `https://ki.akademie-rs.de/static/pwa.js` (Should load JS)

