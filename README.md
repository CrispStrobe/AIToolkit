# KI Suite - Deployment Guide

This documentation covers the deployment of a (mostly) **GDPR-compliant Multi-Provider AI Suite** with:
- 💬 **Chat** (Multiple LLM providers with context-aware history)
- 🎙️ **Audio Transcription** (Gladia, Whisper, Mistral, Deepgram, AssemblyAI; Speaker Diarization; Chunking)
- 📄 **Content Extraction** for uploads (text extraction from PPTX, DOCX, XLSX, PDF; OCR for images/scans)
- 📦 **Cloud Storage** (Hetzner Storage Box integration via SMB)
- 🔄 **Resumable Jobs** (Job tracking for large files)
- 👁️ **Vision Analysis** (Image understanding)
- 🎨 **Image Generation** (e.g. FLUX models via Nebius)
- 💾 **Database Integration** (SQLite with user authentication)
- 📱 **PWA Support** (Progressive Web App for mobile installation)
- 🛡️ **Security** (Fail2Ban protection)

The application runs as a Python/Gradio service on `localhost:7860`, managed by **systemd**, and exposed via **Apache reverse proxy** with SSL.

---

## 📋 Prerequisites

- **OS:** Ubuntu 20.04 LTS or newer
- **RAM:** Minimum 8GB (for handling audio files)
- **Root/Sudo Access**
- **Domain:** DNS A-Record pointing to server IP (e.g., `ai.yourdomain.de`)
- **Storage:** Hetzner Storage Box with sub-account access
- **API Keys:** Supported providers (stored in `.env` file):
  - Mistral (multipurpose)
  - Scaleway (e.g. for transcription)
  - Groq (e.g. for Whisper transcription)
  - Gladia (esp. for long-form transcription)
  - Nebius (e.g. for FLUX image generation)
  - Poe (optional, for additional models)
  - OpenRouter (optional, for additional models)

---

## 🛠️ Step 1: System Dependencies & Security


**CRITICAL:** FFmpeg must be in PATH. `cifs-utils` is required for Storage Box. OCR and document tools are required for the Content Extractor.

```bash
sudo apt update
sudo apt upgrade -y

# Install all required system packages
sudo apt install -y \
    wget \
    python3-pip \
    ffmpeg \
    apache2 \
    certbot \
    python3-certbot-apache \
    sqlite3 \
    fail2ban \
    cifs-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    poppler-utils \
    pandoc
```

### Firewall Configuration (UFW)

**CRITICAL:** You must allow SSH before enabling the firewall, or you will lock yourself out.

```bash
# 1. Allow incoming SSH connections
sudo ufw allow OpenSSH

# 2. Allow Web Traffic (HTTP/HTTPS)
sudo ufw allow 'Apache Full'

# 3. Enable the Firewall
sudo ufw enable

# 4. Verify Status
sudo ufw status
# Output should look like:
# Status: active
# To                         Action      From
# --                         ------      ----
# OpenSSH                    ALLOW       Anywhere
# Apache Full                ALLOW       Anywhere
```

### Configure Swap Space (prevent OOM crashes)

Create some swap space to act as a safety net.

```bash
# Allocate 4GB
sudo fallocate -l 4G /swapfile

# Secure permissions
sudo chmod 600 /swapfile

# Mark as swap
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```


**Verify critical installations:**

```bash
which ffmpeg                    # Must show: /usr/bin/ffmpeg
sudo systemctl status fail2ban  # Must be: active (running)
```

---

## 📦 Step 2: Storage Box Mounting (Hetzner)

Mount e.g. the Hetzner Storage Box e.g. to `/mnt/storage` for direct file access.

### 2.1 Create Mount Point

```bash
sudo mkdir -p /mnt/storage
```

### 2.2 Create Credentials File

```bash
sudo nano /etc/cifs-credentials
```

**Content:**

```ini
username=u12345-sub1
password=YOUR_SUBACCOUNT_PASSWORD
```

**Secure it:**

```bash
sudo chmod 600 /etc/cifs-credentials
```

### 2.3 Configure Auto-Mount via fstab

```bash
sudo nano /etc/fstab
```

**Add (single line):**

```text
//u12345-sub1.your-storagebox.de/u12345-sub1 /mnt/storage cifs credentials=/etc/cifs-credentials,uid=0,gid=0,file_mode=0770,dir_mode=0770,nounix,vers=3.0,x-systemd.automount,x-systemd.idle-timeout=60 0 0
```

> **Note:** `uid=0` assumes app runs as root. Change to `1000` for non-root user.

### 2.4 Mount and Verify

```bash
sudo systemctl daemon-reload
sudo systemctl restart remote-fs.target
ls -la /mnt/storage  # Should list files from Storage Box
```

---

## 🐍 Step 3: Application Setup

### 3.1 Create Directory Structure

```bash
sudo mkdir -p /var/www/transkript_app
sudo mkdir -p /var/www/transkript_app/static
sudo mkdir -p /var/www/transkript_app/generated_images
sudo mkdir -p /var/www/transkript_app/jobs
cd /var/www/transkript_app
```

### 3.2 Clone Repository or Copy Files

**Option A: Clone from GitHub**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .
```

**Option B: Manual File Upload**
Upload these files from your repository:
- `app.py` - Main application
- `crypto_utils.py` - Encryption logic & Key Wrapping
- `requirements.txt` - Python dependencies
- `static/` folder containing:
  - `custom.css` - PWA styling
  - `manifest.json` - PWA manifest
  - `pwa.js` - PWA logic & install prompt
  - `service-worker.js` - Offline caching
  - `icon-192.png` - App icon (192x192)
  - `icon-512.png` - App icon (512x512)

### 3.3 Create Environment File

**CRITICAL:** Store all API keys in a secured `.env` file.

```bash
sudo nano /var/www/transkript_app/.env
```

**Required content:**

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
SCALEWAY_API_KEY=...
GROQ_API_KEY=gsk_...
GLADIA_API_KEY=...
NEBIUS_API_KEY=...
POE_API_KEY=...
ASSEMBLYAI_API_KEY=...
DEEPGRAM_API_KEY=...
GRADIO_ANALYTICS_ENABLED=False
STORAGE_BOX_PATH=/mnt/storage
```

**Secure it:**

```bash
sudo chmod 600 /var/www/transkript_app/.env
```

### 3.4 Create Virtual Environment (Miniconda Method)

We use **Miniconda** to ensure a stable Python 3.10 environment, bypassing system repo issues.

```bash
# 1. Download and Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda

# 2. Initialize Conda
/opt/miniconda/bin/conda init bash
source ~/.bashrc

# Accept TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. Create the 'ak_suite' environment with Python 3.10
/opt/miniconda/bin/conda create -n ak_suite python=3.10 -y

# 4. Link it to the app directory (Standardizes systemd paths)
cd /var/www/transkript_app
rm -rf venv
ln -s /opt/miniconda/envs/ak_suite ./venv
conda activate ak_suite
```

### 3.5 Install Python Dependencies

```bash
# Activate the environment
conda activate ak_suite

# Install Core Dependencies
pip install --upgrade pip wheel
pip install \
    "openai>=1.0" \
    "gradio>=5.0" \
    "yt-dlp>=2024.11" \
    requests \
    pillow \
    pydub \
    sqlalchemy \
    bcrypt \
    fastapi-poe \
    python-pptx \
    python-docx \
    pypdf \
    pymupdf \
    pdf2image \
    pytesseract \
    pandas \
    openpyxl \
    lxml \
    pycryptodome \
    pqcrypto \
    cryptography
```

We use pycryptodome for AES-GCM and cryptography for Fernet/PBKDF2.

Alternatively, for single installs without activation of the environment:
```/var/www/transkript_app/venv/bin/pip install [package]```

### 3.6 Initialize Permissions

```bash
# Create log file
sudo touch /var/www/transkript_app/app.log
sudo chmod 666 /var/www/transkript_app/app.log

# Set directory permissions
sudo chmod -R 755 /var/www/transkript_app

# PWA files need Apache ownership
sudo chown -R www-data:www-data /var/www/transkript_app/static
sudo chmod -R 755 /var/www/transkript_app/static
```

---

## 🌐 Step 4: Apache Configuration

This configuration serves PWA files directly via Apache and proxies the Gradio app with proper HTTPS headers.

### 4.1 Enable Required Modules

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite headers ssl
```

### 4.2 SSL Certificate Setup

```bash
sudo certbot --apache -d ai.yourdomain.de
```

### 4.3 HTTP Config (Port 80 - HTTPS Redirect)

Edit `/etc/apache2/sites-available/transkript.conf`:

```apache
<VirtualHost *:80>
    ServerName ai.yourdomain.com
    
    # Block OpenAPI on HTTP too
    RewriteEngine On
    RewriteRule ^/openapi\.json$ - [F,L]
    RewriteRule ^/docs/?$ - [F,L]
    RewriteRule ^/redoc/?$ - [F,L]
    RewriteRule ^/api(/.*)?$ - [F,L]
    RewriteRule ^/gradio_api/openapi\.json$ - [F,L]
    
    # Websocket support
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://127.0.0.1:7860/$1 [P,L]
    
    RewriteCond %{HTTP:Upgrade} !=websocket [NC]
    RewriteRule /(.*)           http://127.0.0.1:7860/$1 [P,L]
    
    ProxyPass / http://127.0.0.1:7860/
    ProxyPassReverse / http://127.0.0.1:7860/
    
    # Force HTTPS for everything else
    RewriteCond %{SERVER_NAME} =ki.akademie-rs.de
    RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]
</VirtualHost>
```

### 4.4 HTTPS Config (Port 443 - Main Config)

Edit `/etc/apache2/sites-available/transkript-le-ssl.conf`:

```apache
<IfModule mod_ssl.c>
<VirtualHost *:443>
    ServerName ai.yourdomain.com
    
    # =================================================
    # 🔒 BLOCK OPENAPI - FIRST PRIORITY
    # =================================================
    
    RewriteEngine On
    
    # Block all OpenAPI endpoints with [F,L] - Forbidden + Last
    RewriteRule ^/openapi\.json$ - [F,L]
    RewriteRule ^/docs/?$ - [F,L]
    RewriteRule ^/redoc/?$ - [F,L]
    RewriteRule ^/api(/.*)?$ - [F,L]
    RewriteRule ^/gradio_api/openapi\.json$ - [F,L]
    
    # =================================================
    # 1. STATIC FILES (Served by Apache)
    # =================================================
    
    Alias /static /var/www/transkript_app/static
    <Directory /var/www/transkript_app/static>
        Require all granted
        Options -Indexes
        AddType text/css .css
        AddType application/javascript .js
        AddType image/png .png
        Header set Cache-Control "public, max-age=31536000, immutable"
    </Directory>
    
    Alias /manifest.json /var/www/transkript_app/static/manifest.json
    <Files "manifest.json">
        Require all granted
        Header set Content-Type "application/manifest+json"
        Header set Cache-Control "no-cache"
    </Files>
    
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
    
    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Forwarded-Port "443"
    RequestHeader set X-Forwarded-Host "ki.akademie-rs.de"
    
    # Websockets (must come AFTER OpenAPI blocks)
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://127.0.0.1:7860/$1 [P,L]
    
    # Exclude static files from proxy
    ProxyPass /static !
    ProxyPass /manifest.json !
    ProxyPass /service-worker.js !
    
    # Proxy everything else to Gradio
    ProxyPass / http://127.0.0.1:7860/
    ProxyPassReverse / http://127.0.0.1:7860/
    
    # =================================================
    # 3. SSL CONFIGURATION
    # =================================================
    
    SSLCertificateFile /etc/letsencrypt/live/ki.akademie-rs.de/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/ki.akademie-rs.de/privkey.pem
    Include /etc/letsencrypt/options-ssl-apache.conf
    
    # Upload Limits (1GB for audio files)
    LimitRequestBody 1048576000
    ProxyTimeout 600
    TimeOut 600
</VirtualHost>
</IfModule>
```

This way, Gradio API, no matter if active by itself or not, will be strictly not reachable.

### 4.5 Enable Sites and Restart Apache

```bash
sudo a2ensite transkript.conf
sudo a2ensite transkript-le-ssl.conf
sudo apache2ctl configtest  # Should show "Syntax OK"
sudo systemctl restart apache2
```

---

## ⚙️ Step 5: Systemd Service

Create `/etc/systemd/system/transkript.service`:

```ini
[Unit]
Description=Gradio App
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/transkript_app
Environment="PATH=/var/www/transkript_app/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="GRADIO_ANALYTICS_ENABLED=False"
EnvironmentFile=/var/www/transkript_app/.env
ExecStart=/var/www/transkript_app/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Key Points:**
- `EnvironmentFile=/var/www/transkript_app/.env` - Loads API keys from .env file
- `User=root` - Runs as root (change if using non-root deployment)
- `Restart=always` - Auto-restart on crashes

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable transkript
sudo systemctl start transkript
```

**Check status:**

```bash
sudo systemctl status transkript
```

---

## 🛡️ Step 6: Fail2Ban Configuration

Protect against brute-force attacks and malicious bots.

### 6.1 Create Local Configuration

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

### 6.2 Enable Apache Protections

Ensure these sections are enabled:

```ini
[sshd]
enabled = true

[apache-auth]
enabled = true

[apache-badbots]
enabled = true

[apache-noscript]
enabled = true

[apache-overflows]
enabled = true
```

### 6.3 Restart Fail2Ban

```bash
sudo systemctl restart fail2ban
sudo fail2ban-client status  # Verify active jails
```

## 🔄 Step 7: Log Rotation

We prevent `app.log` from consuming too much disk space.

### 7.1 Create Logrotate Config

```bash
sudo nano /etc/logrotate.d/transkript_app
```

**Content:**

```text
/var/www/transkript_app/app.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 666 root root
    copytruncate
}
```

  * `daily`: Rotate logs every day.
  * `rotate 14`: Keep 14 days of logs.
  * `compress`: Compress old logs to save space.
  * `copytruncate`: Crucial for Python apps—truncates the active file in place so the app doesn't need restarting.

**Test the configuration:**

```bash
sudo logrotate -d /etc/logrotate.d/transkript_app
```

---

## 📁 Final File Structure

```
/var/www/transkript_app/
├── app.py                      # Main application (from repo)
├── crypto_utils.py             # Encryption Logic
├── requirements.txt            # Python dependencies (from repo)
├── .env                        # API keys (NOT in repo - created manually)
├── .master_key                 # Global Encryption Key (CRITICAL)
├── .pq_keypair                 # (OPTIONAL) Post-Quantum Keys
├── suite.db                    # SQLite database (auto-created on first run)
├── app.log                     # Application logs
├── venv/                       # Python virtual environment
├── jobs/                       # Resume job manifests (auto-created)
├── generated_images/           # AI-generated images (auto-created)
└── static/                     # PWA assets (from repo, owned by www-data)
    ├── custom.css              # PWA styling
    ├── pwa.js                  # PWA registration & install prompt
    ├── manifest.json           # PWA manifest
    ├── service-worker.js       # Offline caching
    ├── icon-192.png            # App icon (192x192)
    └── icon-512.png            # App icon (512x512)

/mnt/storage/          # Hetzner Storage Box (mounted via CIFS)
├── shared/                     # Shared files across users
└── [username]/                 # User-specific folders
```

**Files from GitHub Repository:**
- `app.py`, `requirements.txt`, `static/*` - Version controlled
- `.env` - **NOT in repo** (contains secrets, create manually on server)
- `suite.db`, `app.log`, `jobs/`, `generated_images/` - Auto-generated

---

## 🔍 Troubleshooting

### Service Not Starting

```bash
# Check service status
sudo systemctl status transkript

# View detailed logs
sudo journalctl -u transkript -f

# Check application logs
tail -f /var/www/transkript_app/app.log

# Common issues:
# - Missing .env file → Create /var/www/transkript_app/.env
# - Missing FFmpeg → sudo apt install ffmpeg
# - Port 7860 in use → sudo lsof -i :7860
```

### PWA Not Installing on Mobile

```bash
# Test URLs (all should return 200 OK)
curl -I https://ai.yourdomain.de/manifest.json
curl -I https://ai.yourdomain.de/service-worker.js
curl -I https://ai.yourdomain.de/static/icon-192.png

# Common fixes:
# 1. Clear browser cache/data on mobile
# 2. Verify HTTPS is working
# 3. Check Apache logs: sudo tail -f /var/log/apache2/error.log
# 4. Verify permissions: ls -la /var/www/transkript_app/static
```

### Storage Box Not Mounting

```bash
# Check mount status
mount | grep storage

# Test credentials
sudo cat /etc/cifs-credentials

# Manual mount test
sudo mount -t cifs \
    //u12345-sub1.your-storagebox.de/u513542-sub1 \
    /mnt/storage \
    -o credentials=/etc/cifs-credentials,uid=0,gid=0

# Check mount in real-time
sudo mount -v -t cifs //u12345-sub1.your-storagebox.de/u12345-sub1 /mnt/storage -o credentials=/etc/cifs-credentials
```

### Apache Configuration Issues

```bash
# Test configuration
sudo apache2ctl configtest

# Check which sites are enabled
sudo apache2ctl -S

# View Apache logs
sudo tail -f /var/log/apache2/error.log
sudo tail -f /var/log/apache2/access.log

# Restart Apache
sudo systemctl restart apache2
```

### Permission Issues

```bash
# Fix static files (must be owned by www-data)
sudo chown -R www-data:www-data /var/www/transkript_app/static
sudo chmod -R 755 /var/www/transkript_app/static

# Fix log file
sudo chmod 666 /var/www/transkript_app/app.log

# Fix app directory
sudo chmod -R 755 /var/www/transkript_app
```

### Missing API Keys

```bash
# Verify .env file exists
cat /var/www/transkript_app/.env

# Test if service can read it
sudo systemctl restart transkript
sudo journalctl -u transkript -n 50  # Look for API key errors, or:
sudo journalctl -u transkript -f # watch continuously
```

---

## 📝 Maintenance Tasks

### SSL Certificate Renewal

Certbot auto-renews, but test it:

```bash
sudo certbot renew --dry-run
```

### Update Application

```bash
cd /var/www/transkript_app

# Pull latest changes (if using Git)
git pull

# Or upload new files manually

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart transkript
```

### Update Python Dependencies

```bash
cd /var/www/transkript_app
source venv/bin/activate
pip list --outdated
pip install --upgrade [package_name]
sudo systemctl restart transkript
```

### Backup Database & Keys

You must backup the .master_key file along with the database. Without it, the database cannot be decrypted.

```bash
# Create backup directory
mkdir -p /var/backups/transkript

# Backup DB and Key
cp /var/www/transkript_app/akademie_suite.db \
   /var/backups/transkript/akademie_suite.db.backup-$(date +%Y%m%d)

cp /var/www/transkript_app/.master_key \
   /var/backups/transkript/master_key.backup-$(date +%Y%m%d)

# Optional: Backup PQ Keypair if it exists
if [ -f /var/www/transkript_app/.pq_keypair ]; then
   cp /var/www/transkript_app/.pq_keypair \
      /var/backups/transkript/pq_keypair.backup-$(date +%Y%m%d)
fi
```

### Monitor Disk Space

```bash
# Check Storage Box usage
df -h /mnt/storage

# Check local disk
df -h /var/www/transkript_app
```

---

## 🚀 Quick Reference

### Service Management
```bash
sudo systemctl start transkript      # Start service
sudo systemctl stop transkript       # Stop service
sudo systemctl restart transkript    # Restart service
sudo systemctl status transkript     # Check status
```

### View Logs
```bash
sudo journalctl -u transkript -f     # Follow systemd logs
tail -f /var/www/transkript_app/app.log  # Follow app logs
sudo tail -f /var/log/apache2/error.log  # Apache errors
```

### Critical Files
- **Application:** `/var/www/transkript_app/app.py`
- **Environment:** `/var/www/transkript_app/.env` (secrets)
- **Service:** `/etc/systemd/system/transkript.service`
- **Apache SSL:** `/etc/apache2/sites-available/transkript-le-ssl.conf`
- **Storage Credentials:** `/etc/cifs-credentials`

---

## ⚠️ Security Checklist

- ✅ `.env` file permissions set to `600` (only root can read)
- ✅ `.master_key` file permissions set to `600` (CRITICAL)
- ✅ `/etc/cifs-credentials` permissions set to `600`
- ✅ Fail2Ban enabled and running
- ✅ SSL certificate valid and auto-renewing
- ✅ Firewall configured (only ports `22, 80, 443` open)
- ✅ Regular backups of `suite.db` and `.master_key`
- ✅ Apache upload limits configured (1GB)
- ✅ Storage Box mounted with restricted permissions

---

**Last Updated:** December 2025
