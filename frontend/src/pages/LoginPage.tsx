import React, { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { MapPin, Plane, Mountain, Compass } from 'lucide-react';
import './LoginPage.css';

type Mode = 'login' | 'register';

export default function LoginPage() {
  const [mode,      setMode]      = useState<Mode>('login');
  const [email,     setEmail]     = useState<string>('');
  const [password,  setPassword]  = useState<string>('');
  const [firstName, setFirstName] = useState<string>('');
  const [lastName,  setLastName]  = useState<string>('');
  const [error,     setError]     = useState<string>('');
  const [loading,   setLoading]   = useState<boolean>(false);
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const switchMode = (m: Mode) => { setMode(m); setError(''); };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      if (mode === 'login') await login(email, password);
      else                  await register(email, password, firstName, lastName);
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.error || 'Something went wrong');
    } finally { setLoading(false); }
  };

  return (
    <div className="login-root">
      <div className="login-left">
        <div className="login-brand">
          <Compass size={32} color="#6c8eff" />
          <span>TravelCopilot</span>
        </div>
        <div className="login-hero">
          <h1>Plan your perfect trip with AI</h1>
          <p>Chat naturally, get personalised day-by-day itineraries, and travel smarter.</p>
          <div className="login-features">
            <div className="feat"><Plane size={18}/><span>Smart itinerary generation</span></div>
            <div className="feat"><MapPin size={18}/><span>Real places, real distances</span></div>
            <div className="feat"><Mountain size={18}/><span>Tailored to your budget</span></div>
          </div>
        </div>
        <div className="login-orbs">
          <div className="orb orb1"/><div className="orb orb2"/><div className="orb orb3"/>
        </div>
      </div>

      <div className="login-right">
        <div className="login-card">
          <div className="login-tabs">
            <button className={mode==='login'?'active':''} onClick={()=>switchMode('login')}>Sign in</button>
            <button className={mode==='register'?'active':''} onClick={()=>switchMode('register')}>Sign up</button>
          </div>
          <form onSubmit={submit} className="login-form">
            {mode==='register' && (
              <div className="name-row">
                <div className="field">
                  <label>First name</label>
                  <input value={firstName} onChange={e=>setFirstName(e.target.value)} placeholder="John" required />
                </div>
                <div className="field">
                  <label>Last name</label>
                  <input value={lastName} onChange={e=>setLastName(e.target.value)} placeholder="Doe" />
                </div>
              </div>
            )}
            <div className="field">
              <label>Email</label>
              <input type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@email.com" required />
            </div>
            <div className="field">
              <label>Password</label>
              <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="••••••••" required />
            </div>
            {error && <div className="login-error">{error}</div>}
            <button type="submit" className="btn-primary login-btn" disabled={loading}>
              {loading ? 'Please wait…' : mode==='login' ? 'Sign in' : 'Create account'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
