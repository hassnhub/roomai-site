# AI Room Designer
### AI-Powered Personalized Interior Design and Room Transformation System

Final-year computing project artefact. A user photographs a room, chooses a style, colour scheme,
room dimensions and budget, and the system:

1. **Detects furniture** with a COCO-pretrained **YOLOv8** model (sofa, bed, chairs, tables, TV, plants…).
2. **Segments the room** with **SegFormer (ADE20K)** — walls, floor, ceiling, windows, doors and furniture
   surfaces at pixel level — and measures the layout (visible floor, windows, natural light, wall/floor colours).
3. **Reads the colour palette** (k-means in CIELAB) and infers the room type from what it detected.
4. **Recommends furniture & décor** from a 450-item catalogue, scored on style match, colour harmony (CIE ΔE),
   physical fit and value, within budget and a 45% floor-space limit — every pick explained.
5. **Produces three costed options** (Best Match, Budget Saver, Alternative Look) with **Swap** for any item.
6. **Visualises the redesign on the user's own photo**: walls repainted, floor refinished and furniture
   recoloured using the segmentation masks while keeping the photo's lighting — or photoreal renders via
   Stability AI / Replicate when an API key is configured.
7. **Design Explorer**: change style, colours or budget and see the cost of all three options update live.

## Screens

| # | Screen | Route |
|---|---|---|
| 1 | Landing (before/after hero, Start Transformation) | `/` (signed out) |
| 2 | Sign up / Log in (email + Google) | `/signup`, `/login` |
| 3 | Upload Room (drag-and-drop or camera, room type, L × W × H) | `/create` step 1 |
| 4 | Personalization (6 styles + 3 more, colour schemes, budget slider + input) | `/create` step 2 |
| 5 | Room Analysis (loading steps, then labelled furniture/windows/walls/floor) | `/designs/:id/analysis` |
| 6 | Redesign Result (before/after slider, 3 options) | `/designs/:id` |
| 7 | Recommendations (product cards, Swap, sticky cost summary) | `/designs/:id/products` |
| 8 | Design Explorer (live cost, regenerate) | `/designs/:id/explore` |
| 9 | My Designs (grid, favourites, search) | `/designs` |
| 10 | Profile & Settings (name, password, privacy, export, delete) | `/profile` |
| – | Design system reference | `/design-system` |

Mobile-first with a bottom tab bar (Home · Create · Designs · Profile); Poppins type, warm neutrals with a
sage accent, all text meeting WCAG AA contrast, 44 px+ tap targets, and empty/loading/error states throughout.

## Quick start (Windows)

Double-click **`START_ROOMAI.bat`** — it starts both servers and opens Chrome at http://localhost:5173.

```bash
# Terminal 1 — API (http://127.0.0.1:8000, interactive docs at /docs)
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# Terminal 2 — Web app (http://localhost:5173)
cd frontend
npm run dev
```

Demo login: see `backend/scripts/create_demo_user.py` (run it to create/reset the demo account).
The first analysis after starting the API takes ~40 s while the two AI models load; later ones take a few seconds.

## Fresh install

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # then set JWT_SECRET (and optionally AI / Google keys)

cd ..\frontend
npm install
```

Requires Python 3.11+, Node 18+ and MongoDB on `localhost:27017`. The catalogue seeds itself on first start;
YOLO (6 MB) and SegFormer (15 MB) weights download automatically on first use.

## Optional integrations (`backend/.env`)

| Setting | Enables |
|---|---|
| `STABILITY_API_KEY` | Photoreal redesigns via Stability AI "Structure" control (keeps the room layout) |
| `REPLICATE_API_TOKEN` | Photoreal redesigns via the `adirik/interior-design` ControlNet model |
| `GOOGLE_CLIENT_ID` | "Continue with Google" (OAuth 2.0 Web client; authorised origin `http://localhost:5173`) |

Without an AI key the free offline renderer is used; if a cloud call fails it falls back automatically.
Without a Google client ID the Google button explains that it isn't configured.

## Tests

```bash
cd backend
.venv\Scripts\python.exe -m pytest -v
```

Automated tests cover the recommender (budget, fit, floor space, style rules, option ordering), colour science,
the realistic renderer, secure upload handling, YOLO detection, room segmentation, authentication, access control,
swap/preview endpoints, encryption at rest and GDPR export/erasure.

`scripts/reprocess_designs.py` re-analyses designs created by older versions (adds segmentation, re-renders).

## Architecture

```
React (Vite) SPA ──/api──► FastAPI ──► MongoDB (users, designs, products)
                              ├── services/detection.py     YOLOv8 furniture detection (CPU)
                              ├── services/segmentation.py  SegFormer walls/floor/windows/furniture masks
                              ├── services/colours.py       k-means palette, CIELAB ΔE
                              ├── services/recommender.py   knowledge-based content filtering + swap
                              ├── services/visualiser.py    mask-based offline renderer / Stability / Replicate
                              ├── services/imaging.py       validation, EXIF strip, person blurring
                              └── storage.py                Fernet-encrypted image storage
```

## Security & ethics

- bcrypt password hashing, JWT sessions, login rate limiting, generic login errors; verified Google ID tokens
- Ownership enforced in every database query (users cannot access each other's designs)
- Uploads validated by decoding, size/dimension limits, decompression-bomb guard
- EXIF metadata (incl. GPS) stripped; people detected in photos are automatically blurred
- Photos and segmentation masks encrypted at rest; served only through authenticated endpoints
- Explicit consent at sign-up; data export (UK GDPR Art. 15) and full account erasure (Art. 17)
- Security headers; regex-safe search

## Notes

The product catalogue is **synthetic demonstration data** with indicative UK prices — generated, not scraped,
to avoid retailer terms-of-service and copyright issues. Product images are illustrations.
