# Atrophy Advisor Frontend

React + TypeScript + Vite frontend for the Atrophy Advisor OCT analysis application.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The frontend runs on `http://localhost:3000` and proxies API requests to `http://localhost:8000/api`.

Make sure the backend API server is running before starting the frontend.

## Architecture

- **React 18** with TypeScript for type safety
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Axios** for API communication

## Components

- `ImageUpload`: Drag-and-drop file upload with date picker
- `ImageCanvas`: Canvas for displaying images with annotations
- `ResultsPanel`: Display progression analysis results
- `App`: Main application orchestrator

## Workflow

1. User uploads "before" image → automatic disc/fovea/GA detection
2. User uploads "after" image → automatic disc/fovea/GA detection
3. System automatically calculates progression and displays prediction
4. User can click GA regions to select different regions for distance measurement
