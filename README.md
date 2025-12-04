# Akademie KI Suite - Complete Deployment Guide

This documentation covers the deployment of a **GDPR-compliant Multi-Provider AI Suite** with:
- 💬 **Chat** (Multiple LLM providers)
- 🎙️ **Audio Transcription** (Gladia, Scaleway, Groq with Whisper)
- 👁️ **Vision Analysis** (Image understanding)
- 🎨 **Image Generation** (FLUX models via Nebius)
- 💾 **Database Integration** (SQLite with user authentication)
- 📱 **PWA Support** (Progressive Web App for mobile installation)

The application runs as a Python/Gradio service on `localhost:7860`, managed by **systemd**, and exposed via **Apache reverse proxy** with SSL. It coexists with other services (e.g., LimeSurvey) on the same server using different subdomains.

## 📋 Prerequisites

- **OS:** Ubuntu 20.04 LTS (tested) or newer
- **RAM:** Minimum 8GB (for handling audio files)
- **Root/Sudo Access**
- **Domain/Subdomain:** DNS A-Record pointing to server IP (e.g., `ai.your-domain.com`)
- **API Keys:** From providers you wish to use (see Configuration section)

---

## 🛠️ Step 1: System Dependencies

**CRITICAL:** FFmpeg must be in PATH for audio transcription to work.
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip ffmpeg apache2 certbot python3-certbot-apache sqlite3
```

**Verify FFmpeg installation:**
```bash
which ffmpeg
# Should output: /usr/bin/ffmpeg
```

---

## 🐍 Step 2: Application Setup

### 2.1 Create Directory Structure
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

**For Ubuntu 20.04 / Python 3.8:**
```bash
pip install --upgrade pip
pip install --break-system-packages \
    "openai<2.0" \
    gradio==4.44.0 \
    requests \
    pillow \
    pypdf2 \
    sqlalchemy \
    bcrypt
```

**For Ubuntu 22.04+ / Python 3.10+:**
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

### 2.4 Create Application File
```bash
sudo nano /var/www/transkript_app/app.py
```

Paste your complete application code. **Important sections to configure:**

1. **API Keys Section** (top of file):
```python
API_KEYS = {
    "SCALEWAY": os.getenv("SCALEWAY_API_KEY", ""),
    "MISTRAL": os.getenv("MISTRAL_API_KEY", ""),
    "NEBIUS": os.getenv("NEBIUS_API_KEY", ""),
    "OPENROUTER": os.getenv("OPENROUTER_API_KEY", ""),
    "GROQ": os.getenv("GROQ_API_KEY", ""),
    "GLADIA": os.getenv("GLADIA_API_KEY", "")
}
```

2. **Default Users** (in `create_default_users()` function):
```python
admin = User(
    username="admin",
    password_hash=hash_password("YOUR_ADMIN_PASSWORD"),
    email="your-email@domain.com",
    is_admin=True
)
```

3. **Gladia Custom Vocabulary** (if using):
```python
GLADIA_CONFIG = {
    "url": "https://api.gladia.io/v2",
    "vocab": [
        "Christian Ströbele",  # Replace with your specific terms
        "Jesus Christus",
        # Add more domain-specific terms...
    ]
}
```

### 2.5 Create Log File
```bash
sudo touch /var/www/transkript_app/app.log
sudo chmod 666 /var/www/transkript_app/app.log
```

### 2.6 Initialize Database

The database will auto-create on first run, but verify permissions:
```bash
sudo chmod 755 /var/www/transkript_app
sudo chmod 666 /var/www/transkript_app/akademie_suite.db  # After first run
```

---

## 🌐 Step 3: Apache Reverse Proxy Configuration

### 3.1 Enable Required Modules
```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod proxy_wstunnel
sudo a2enmod rewrite
sudo a2enmod headers
sudo a2enmod ssl
sudo systemctl restart apache2
```

### 3.2 Create HTTP VirtualHost (Initial)
```bash
sudo nano /etc/apache2/sites-available/ai-suite.conf
```

**Content:**
```apache
<VirtualHost *:80>
    ServerName ai.your-domain.com
    
    ErrorLog ${APACHE_LOG_DIR}/ai-error.log
    CustomLog ${APACHE_LOG_DIR}/ai-access.log combined

    # Static file serving for PWA
    Alias /static /var/www/transkript_app/static
    Alias /manifest.json /var/www/transkript_app/manifest.json
    Alias /service-worker.js /var/www/transkript_app/service-worker.js
    
    <Directory /var/www/transkript_app/static>
        Require all granted
    </Directory>
    
    <Files "manifest.json">
        Require all granted
    </Files>
    
    <Files "service-worker.js">
        Require all granted
    </Files>

    # Prevent proxying static files
    ProxyPass /static !
    ProxyPass /manifest.json !
    ProxyPass /service-worker.js !

    # WebSocket support (critical for Gradio)
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://127.0.0.1:7860/$1 [P,L]
    RewriteCond %{HTTP:Upgrade} !=websocket [NC]
    RewriteRule /(.*)           http://127.0.0.1:7860/$1 [P,L]

    # Standard proxy
    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:7860/
    ProxyPassReverse / http://127.0.0.1:7860/
</VirtualHost>
```

### 3.3 Enable Site
```bash
sudo a2ensite ai-suite.conf
sudo apache2ctl configtest
sudo systemctl reload apache2
```

---

## 🔒 Step 4: SSL/HTTPS with Let's Encrypt
```bash
sudo certbot --apache -d ai.your-domain.com
```

**Select option 2 (Redirect HTTP to HTTPS)**

Certbot will create `/etc/apache2/sites-available/ai-suite-le-ssl.conf`

### 4.1 Update SSL Config for Large Uploads
```bash
sudo nano /etc/apache2/sites-available/ai-suite-le-ssl.conf
```

**Add inside `<VirtualHost *:443>`:**
```apache
    # Increase limits for audio file uploads (100MB)
    LimitRequestBody 104857600
    ProxyTimeout 600
    TimeOut 600
```

### 4.2 Also Update Main Apache Config
```bash
sudo nano /etc/apache2/apache2.conf
```

**Add at the end:**
```apache
# File upload limits
LimitRequestBody 104857600
TimeOut 600
```

**Reload Apache:**
```bash
sudo apache2ctl configtest
sudo systemctl reload apache2
```

---

## ⚙️ Step 5: Systemd Service Configuration

**CRITICAL:** The PATH environment variable must include `/usr/bin` for FFmpeg access.

### 5.1 Create Service File
```bash
sudo nano /etc/systemd/system/transkript.service
```

**Content:**
```ini
[Unit]
Description=Gradio Akademie KI Suite
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/transkript_app

# CRITICAL: Full PATH including /usr/bin for ffmpeg
Environment="PATH=/var/www/transkript_app/venv/bin:/usr/local/bin:/usr/bin:/bin"

# Force immediate logging to journalctl
Environment="PYTHONUNBUFFERED=1"

ExecStart=/var/www/transkript_app/venv/bin/python app.py

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 5.2 Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable transkript
sudo systemctl start transkript
```

### 5.3 Verify Service Status
```bash
sudo systemctl status transkript
```

Should show **"active (running)"**.

**Monitor logs in real-time:**
```bash
sudo journalctl -u transkript -f
```

**Also check app log:**
```bash
tail -f /var/www/transkript_app/app.log
```

---

## 📱 Step 6: PWA Setup (Progressive Web App)

### 6.1 Create Manifest
```bash
sudo nano /var/www/transkript_app/manifest.json
```

**Content:**
```json
{
  "name": "Akademie KI Suite",
  "short_name": "KI Suite",
  "description": "AI Tools für Transkription, Chat und Bildgenerierung",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1a1a2e",
  "theme_color": "#6366f1",
  "orientation": "portrait-primary",
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

### 6.2 Create Service Worker
```bash
sudo nano /var/www/transkript_app/service-worker.js
```

**Content:**
```javascript
const CACHE_NAME = 'akademie-ki-v1';
const urlsToCache = [
  '/',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
```

### 6.3 Generate App Icons
```bash
cd /var/www/transkript_app
source venv/bin/activate
python3 << 'EOF'
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    img = Image.new('RGB', (size, size), color='#6366f1')
    draw = ImageDraw.Draw(img)
    
    # Draw simple "KI" text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size//3)
    except:
        font = ImageFont.load_default()
    
    text = "KI"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((size - text_width) / 2, (size - text_height) / 2)
    draw.text(position, text, fill='white', font=font)
    
    os.makedirs('static', exist_ok=True)
    img.save(f'static/{filename}')
    print(f"Created {filename}")

create_icon(192, 'icon-192.png')
create_icon(512, 'icon-512.png')
EOF
```

---

## 🔧 Step 7: Configuration & API Keys

### 7.1 Set Environment Variables (Recommended)
```bash
sudo nano /etc/systemd/system/transkript.service
```

**Add Environment lines:**
```ini
Environment="SCALEWAY_API_KEY=your_key_here"
Environment="MISTRAL_API_KEY=your_key_here"
Environment="NEBIUS_API_KEY=your_key_here"
Environment="GROQ_API_KEY=your_key_here"
Environment="GLADIA_API_KEY=your_key_here"
```

**Then reload:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart transkript
```

### 7.2 Alternative: Direct in app.py

Edit `/var/www/transkript_app/app.py` and fill in the `API_KEYS` dictionary directly.

---

## 📊 Database Information

### Location
```
/var/www/transkript_app/akademie_suite.db
```

### Default Users

Created on first run:

| Username | Password | Admin | Email |
|----------|----------|-------|-------|
| `admin` | `akademie2025` | Yes | `stroebele@akademie-rs.de` |
| `user` | `dialog2025` | No | `dialog@akademie-rs.de` |

**IMPORTANT:** Change these passwords in `app.py` before deployment!

### Database Schema

- **User**: Authentication and authorization
- **ChatHistory**: Saved conversations
- **Transcription**: Saved audio transcriptions
- **VisionResult**: Vision analysis results
- **GeneratedImage**: Generated images metadata
- **CustomPrompt**: User-defined prompt templates

### Backup Database
```bash
sudo cp /var/www/transkript_app/akademie_suite.db \
       /var/www/transkript_app/akademie_suite.db.backup
```

---

## 🔍 Troubleshooting

### Service Won't Start
```bash
# Check logs
sudo journalctl -u transkript -f

# Common issues:
# 1. FFmpeg not found: Check PATH in service file
# 2. Port 7860 already in use: sudo ss -tuln | grep 7860
# 3. Import errors: Check venv installation
```

### Transcription Fails with "Connection errored out"

**Causes:**
1. Audio file too large (>100MB)
2. Apache timeout too short
3. FFmpeg not accessible

**Fix:**
```bash
# Verify FFmpeg
/usr/bin/ffmpeg -version

# Check Apache limits
grep -r "LimitRequestBody" /etc/apache2/

# Test locally first
cd /var/www/transkript_app
source venv/bin/activate
python3 -c "import subprocess; print(subprocess.run(['ffmpeg', '-version'], capture_output=True))"
```

### Database Errors
```bash
# Check permissions
ls -la /var/www/transkript_app/akademie_suite.db

# Should be readable/writable by root
sudo chmod 666 /var/www/transkript_app/akademie_suite.db
```

### PWA Not Installing

1. **Check HTTPS:** PWAs require HTTPS (except localhost)
2. **Check manifest:** `curl https://ai.your-domain.com/manifest.json`
3. **Check Apache static file serving:**
```bash
   curl -I https://ai.your-domain.com/static/icon-192.png
```

---

## 🔄 Maintenance

### Update Application Code
```bash
sudo nano /var/www/transkript_app/app.py
sudo systemctl restart transkript
```

### Update Python Dependencies
```bash
cd /var/www/transkript_app
source venv/bin/activate
pip install --upgrade gradio openai
sudo systemctl restart transkript
```

### View Logs
```bash
# Service logs
sudo journalctl -u transkript -f

# Application logs
tail -f /var/www/transkript_app/app.log

# Apache logs
sudo tail -f /var/log/apache2/ai-error.log
```

### SSL Certificate Renewal

Certbot auto-renews. To test:
```bash
sudo certbot renew --dry-run
```

---

## 📁 Complete File Structure
```
/var/www/transkript_app/
├── app.py                          # Main application
├── akademie_suite.db               # SQLite database
├── app.log                         # Application logs
├── venv/                           # Python virtual environment
├── static/                         # Static files
│   ├── icon-192.png
│   ├── icon-512.png
│   ├── custom.css (if used)
│   └── pwa.js (if used)
├── generated_images/               # Saved generated images
├── manifest.json                   # PWA manifest
└── service-worker.js               # PWA service worker

/etc/systemd/system/
└── transkript.service              # Systemd service config

/etc/apache2/sites-available/
├── ai-suite.conf                   # HTTP config
└── ai-suite-le-ssl.conf            # HTTPS config (created by certbot)
```

---

## 🎯 Provider-Specific Notes

### Gladia (GDPR-Compliant, France)
- **Best for:** German transcriptions with diarization
- **Features:** Speaker identification, translation, custom vocabulary
- **Pricing:** Pay per minute
- **API:** https://api.gladia.io/v2

### Scaleway (GDPR-Compliant, France)
- **Best for:** Fast Whisper transcriptions in EU
- **Models:** whisper-large-v3
- **Critical:** Use `response_format="json"` (NOT "verbose_json")
- **Language:** Always set explicitly for non-English (e.g., `language="de"`)

### Groq (US-Based)
- **Best for:** Ultra-fast inference, free tier
- **Models:** whisper-large-v3-turbo, whisper-large-v3
- **Supports:** `response_format="verbose_json"` with segments

### Nebius (GDPR-Compliant, EU)
- **Best for:** Image generation (FLUX models)
- **Max inference steps:** 16 (critical for Gradio slider)

---

## 🛡️ Security Recommendations

1. **Change default passwords** in `app.py`
2. **Use environment variables** for API keys
3. **Restrict database permissions:** `chmod 600 akademie_suite.db`
4. **Enable Apache security headers:**
```apache
   Header always set X-Frame-Options "SAMEORIGIN"
   Header always set X-Content-Type-Options "nosniff"
```
5. **Regular backups** of database
6. **Monitor logs** for suspicious activity
7. **Keep system updated:** `sudo apt update && sudo apt upgrade`

---

## 📞 Support Resources

- **Gradio Docs:** https://gradio.app/docs
- **Apache Proxy:** https://httpd.apache.org/docs/2.4/mod/mod_proxy.html
- **Let's Encrypt:** https://letsencrypt.org/docs/
- **SQLAlchemy:** https://docs.sqlalchemy.org/

---

**Last Updated:** December 2025  
**Tested On:** Ubuntu 20.04.4 LTS, Apache 2.4.41, Python 3.8.10
