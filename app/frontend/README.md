# Frontend Documentation

This is the frontend portion of the HackCMU2025 project built with React and Vite.

## Prerequisites

- Node.js (v14 or higher)
- npm (comes with Node.js)

## Installation

1. Navigate to the frontend directory:
   ```bash
   cd app/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Development Server

To start the development server:

```bash
npm run dev
```

This will start the Vite development server. By default, it will be available at `http://localhost:3000`.

## Building for Production

To create a production build:

```bash
npm run build
```

This will create a `dist` directory with the compiled assets ready for deployment.

To preview the production build locally:

```bash
npm run preview
```

## Project Structure

- `src/` - Contains the source code
  - `components/` - React components
  - `App.jsx` - Main application component
  - `main.jsx` - Application entry point
- `public/` - Static assets
- `index.html` - HTML entry point
- `vite.config.js` - Vite configuration