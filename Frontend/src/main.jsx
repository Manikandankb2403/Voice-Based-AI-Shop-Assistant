// main.jsx (or index.js)
import React from 'react';
import ReactDOM from 'react-dom/client';  // Import createRoot from react-dom/client
import App from './App';
import './index.css';  // or the appropriate CSS file


const root = ReactDOM.createRoot(document.getElementById('root'));  // Create root using createRoot
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);