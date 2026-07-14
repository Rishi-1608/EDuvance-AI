# EDuvance — Glassmorphic Frontend

A premium glassmorphic React (Vite) frontend for the Multi-Modal Academic Intelligence System v2.0.0.

## Stack
- **React 18** + **Vite 5**
- **Pure CSS** glassmorphism design system (no UI library)
- **Syne** (display) + **Lora** (body) + **DM Mono** (code) fonts
- Component-based architecture with clean API separation

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. (Optional) Set your API base URL
echo "VITE_API_URL=http://127.0.0.1:8000" > .env

# 3. Start dev server
npm run dev

# 4. Build for production
npm run build
```

Then open http://localhost:5173 in your browser.

## Project Structure

```
src/
├── lib/
│   └── api.js              # All backend API calls (single source of truth)
├── hooks/
│   ├── useStatus.js        # Auto-polling /status every 3s
│   └── useTheme.js         # Dark/light mode toggle
├── components/
│   └── layout/
│       ├── Navbar.jsx      # Sticky glass navbar + mobile nav
│       └── Navbar.css
├── pages/
│   ├── UploadPage.jsx      # Drag-and-drop upload for video/image/audio
│   ├── DashboardPage.jsx   # Live pipeline status & per-video progress
│   ├── NotesPage.jsx       # Markdown study notes viewer with search
│   ├── FlashcardsPage.jsx  # Flip-card Q&A with progress tracking
│   ├── QuizPage.jsx        # Interactive MCQ with scoring & explanations
│   └── DiagnosticsPage.jsx # System health & endpoint reference
├── App.jsx                 # Root + tab routing
├── App.css
├── index.css               # Full glassmorphism design system
└── main.jsx
```

## API Endpoints Integrated

| Page          | Endpoints used |
|---------------|----------------|
| Upload        | POST /upload/video, /upload/image, /upload/audio |
| Dashboard     | GET /status, POST /stop, DELETE /results |
| Notes         | GET /results/notes/{stem}, GET /results/pdf/{stem} |
| Flashcards    | GET /results/flashcards/{stem} |
| Quiz          | GET /results/quiz/{stem} |
| Diagnostics   | GET /diagnostics |

## Design System

The entire design system lives in `src/index.css` via CSS custom properties:

- **Glass cards** — `backdrop-filter: blur(18px)`, translucent backgrounds
- **Nebula palette** — violet, blue, teal, gold, rose on cosmic void dark
- **Light mode** — auto-switches all variables via `[data-theme="light"]`
- **Micro-animations** — `fadeUp`, `shimmer` skeletons, `pulse` dots, card flips
- **Typography** — Syne (display headings), Lora (body), DM Mono (code/labels)

## Environment Variables

| Variable        | Default                    | Description          |
|-----------------|----------------------------|----------------------|
| VITE_API_URL    | http://127.0.0.1:8000      | FastAPI backend URL  |
