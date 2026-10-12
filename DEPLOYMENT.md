# Deploying AI Room Designer on cPanel

This guide puts the app live on a cPanel host (your own hosting or the School of Computing server).

## 0. Check your cPanel can run it

Log in to cPanel and look in the **Software** section for **Setup Python App**.

| You see… | Meaning |
|---|---|
| **Setup Python App** with Python **3.10 or newer** | ✅ You can deploy. Continue below. |
| Only PHP / WordPress / "Website Builder" tools | ❌ This plan can't run Python. Ask your host to enable it, or use a VPS. |
| Python 3.9 or older only | ❌ Too old. Ask your host for 3.11 or 3.12. |

Also check **Terminal** (or SSH access) under **Advanced**. It makes installing much easier.

**Memory:** the AI models need about **2 GB RAM**. Many shared plans allow only 1 GB. If yours does,
use **Lite mode** (step 4b). Everything still works except detecting objects in photos.

## 1. Create a free MongoDB database (cPanel doesn't include MongoDB)

1. Sign up at **mongodb.com/atlas** and create a free **M0** cluster (choose a region near you, e.g. London).
2. **Database Access** → add a database user with a strong password.
3. **Network Access** → add your server's IP address (cPanel shows it on the right-hand side as
   "Shared IP Address"). Use `0.0.0.0/0` only if your host's IP keeps changing.
4. **Connect → Drivers** → copy the connection string. It looks like
   `mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`

## 2. Build the upload package (on your laptop)

Decide the address first. For `https://yourdomain.com/roomai/`, the base is `/roomai/`.
For a subdomain such as `https://roomai.yourdomain.com/`, the base is `/`.

In **PowerShell** (not Git Bash) inside the `RoomAI` folder:

```powershell
.\backend\.venv\Scripts\python.exe make_deploy_package.py --base /roomai/
```

This creates `deploy\RoomAI-deploy.zip` (≈20 MB, AI models included) and
`deploy\SECRETS-keep-private.txt` (new secret keys; never upload or share this file).

## 3. Upload

1. cPanel → **File Manager** → open your home folder (e.g. `/home/yourname`), **not** `public_html`.
2. **Upload** `RoomAI-deploy.zip`, then right-click → **Extract**. You get `/home/yourname/roomai/`.

## 4. Create the Python app

cPanel → **Setup Python App** → **Create Application**:

| Field | Value |
|---|---|
| Python version | 3.11 or 3.12 (highest available) |
| Application root | `roomai/backend` |
| Application URL | your domain + `roomai` (must match the `--base` from step 2) |
| Application startup file | `passenger_wsgi.py` |
| Application Entry point | `application` |

Click **Create**. At the top, cPanel shows a command to **enter the virtual environment**. Copy it.

### Environment variables (same page → "Environment variables" → Add)

| Name | Value |
|---|---|
| `ENVIRONMENT` | `production` |
| `JWT_SECRET` | from `SECRETS-keep-private.txt` |
| `STORAGE_KEY` | from `SECRETS-keep-private.txt` |
| `MONGO_URI` | your Atlas connection string from step 1 |
| `MONGO_DB` | `roomai` |
| `HF_HUB_OFFLINE` | `1` (use the bundled models; no internet needed) |

Leave `BASE_PATH` empty. cPanel already handles the `/roomai` part of the URL.
Optional: `STABILITY_API_KEY` (photoreal renders), `GOOGLE_CLIENT_ID` (Google sign-in).

### 4a. Install packages: full mode (needs ~2 GB RAM)

cPanel → **Terminal**, paste the "enter virtual environment" command, then:

```bash
cd ~/roomai/backend
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4b. Install packages: Lite mode (1 GB plans)

```bash
cd ~/roomai/backend
pip install -r requirements-lite.txt
```

Also add the environment variable `PRELOAD_MODELS` = `0`.

No Terminal? In **Setup Python App** type `requirements-lite.txt` (or `requirements.txt`) in
"Configuration files" and click **Run Pip Install**. Full mode usually times out this way, so use the Terminal for it.

## 5. Start and check

1. **Setup Python App** → **Restart**.
2. Visit `https://yourdomain.com/roomai/api/health`. You should see `"status":"ok","database":true`.
   About 40 seconds after a restart, `"detector":{"loaded":true}` appears in full mode.
3. Visit `https://yourdomain.com/roomai/`. You should see the landing page. Sign up and create a design.
4. Optional demo account: in the Terminal (inside the virtual environment) run
   `python scripts/create_demo_user.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Incomplete response" / 503 | Read the error log: `stderr.log` in `~/roomai/backend` if present, otherwise cPanel → **Metrics → Errors**. Or run `python passenger_wsgi.py` in the Terminal (inside the virtual environment) to see the error directly. |
| `Unsafe production configuration` | `JWT_SECRET` or `STORAGE_KEY` is missing or too short. |
| `database: false` | Wrong `MONGO_URI`, or your server's IP isn't in Atlas → Network Access. |
| Blank page, files 404 | The `--base` you packaged with doesn't match the Application URL. Re-package and re-upload `frontend/dist`. |
| Process killed / out of memory in full mode | Your plan's RAM is too low. Switch to Lite mode (4b). |
| `pip install torch` fails (disk/inode quota) | Use Lite mode, or ask the host for a bigger quota. |
| First design very slow | Normal for 30–40 s after a restart while models load (`PRELOAD_MODELS=1` warms them up). |
| Upload fails for large photos | Your host's request size limit. Photos under ~8 MB work on most hosts. |

## Updating the live app

Re-run step 2, upload the new zip, extract over the old folder (your data is in MongoDB and
`roomai/backend/storage`, which the package never overwrites), then **Restart** the app.
Keep the **same** `STORAGE_KEY`, or older photos can't be decrypted.

## Security checklist before sharing the link

- [ ] Site loads over **https://** (cPanel → SSL/TLS Status → run AutoSSL if needed)
- [ ] `ENVIRONMENT=production` (hides the API docs, turns on HSTS, refuses weak secrets)
- [ ] `SECRETS-keep-private.txt` is **not** uploaded anywhere and is backed up privately
- [ ] Atlas user has a strong password; network access limited to your server's IP
- [ ] Ethics approval is in place before real users upload photos of their homes
