// App.jsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';  // Use Routes instead of Switch
import LoginPage from './Components/LoginForm';
import ChatPage from './Components/ChatWindow';

const App = () => {
  return (
    <Router>
      <div className="app">
        <Routes>  {/* Use Routes to wrap Route components */}
          <Route path="/" element={<LoginPage />} />  {/* Update to use 'element' instead of 'component' */}
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
