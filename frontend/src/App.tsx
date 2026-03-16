import React, { ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import HomePage  from './pages/HomePage';
import ChatPage  from './pages/ChatPage';
import TripPage  from './pages/TripPage';

function Guard({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh' }}>
      <div style={{ width:36, height:36, border:'3px solid #2a3347', borderTopColor:'#6c8eff', borderRadius:'50%', animation:'spin 0.8s linear infinite' }}/>
    </div>
  );
  return user ? <>{children}</> : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login"        element={<LoginPage />} />
          <Route path="/"             element={<Guard><HomePage /></Guard>} />
          <Route path="/chat/:tripId" element={<Guard><ChatPage /></Guard>} />
          <Route path="/trip/:tripId" element={<Guard><TripPage /></Guard>} />
          <Route path="*"             element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
