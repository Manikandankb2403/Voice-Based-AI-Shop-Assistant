// App.jsx
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import LoginPage from './LoginPage';
import ChatPage from './ChatPage';

const App = () => {
  return (
    <Router>
      <div className="app">
        <Switch>
          <Route exact path="/" component={LoginPage} />
          <Route path="/chat" component={ChatPage} />
        </Switch>
      </div>
    </Router>
  );
};

export default App;
