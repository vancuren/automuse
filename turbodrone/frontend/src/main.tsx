import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import TestPage from "./pages/TestPage";

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* basename follows Vite's base, so a build served under a sub-path (vite build --base /ui/) still routes */}
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "")}>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/test" element={<TestPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
