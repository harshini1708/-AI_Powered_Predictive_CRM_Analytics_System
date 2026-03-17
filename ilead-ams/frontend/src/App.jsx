import React from 'react';
import { AuthProvider } from './context/AuthContext';
import { Router } from './Router';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <Router />
    </AuthProvider>
  );
}

export default App;
