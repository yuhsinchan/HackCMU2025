import React, { useState } from 'react';

function LandingPage({ onStart }) {
  const [hover, setHover] = useState(false);

  return (
    <div style={{ minHeight: '100vh', background: '#0b0f12', color: '#e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <div style={{ width: '100%', maxWidth: 960 }}>
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: '3rem', letterSpacing: 0.5 }}>
            <span style={{ color: '#00FF95' }}>Gymaster</span>
            <span style={{ color: '#e5e7eb' }}> – Your AI Form Coach</span>
          </h1>
          <p style={{ marginTop: 8, color: '#9ca3af', fontSize: '1.1rem' }}>
            Real‑time exercise form analysis with live feedback and coaching tips.
          </p>
        </div>

        <div style={{ background: '#111827', borderRadius: 16, padding: 24, boxShadow: '0 0 0 1px rgba(0,255,149,0.15), 0 20px 60px rgba(0,0,0,0.45)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            <div>
              <div style={{ border: '2px solid #00FF95', borderRadius: 12, aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.85 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ width: 56, height: 56, borderRadius: '50%', border: '3px solid #00FF95', margin: '0 auto 12px', boxShadow: '0 0 20px rgba(0,255,149,0.35)' }} />
                  <p style={{ margin: 0, color: '#9ca3af' }}>Your video feed will appear here</p>
                </div>
              </div>
            </div>
            <div>
              <h2 style={{ marginTop: 0, marginBottom: 12, fontSize: '1.4rem' }}>What you’ll get</h2>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 10 }}>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#00FF95', boxShadow: '0 0 10px #00FF95', marginTop: 7 }} />
                  <div>
                    <strong>Live skeleton overlay</strong>
                    <div style={{ color: '#9ca3af' }}>Visualize joints and posture while you move.</div>
                  </div>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#00FF95', boxShadow: '0 0 10px #00FF95', marginTop: 7 }} />
                  <div>
                    <strong>Action recognition</strong>
                    <div style={{ color: '#9ca3af' }}>Automatically detects squats, deadlifts, and more.</div>
                  </div>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#00FF95', boxShadow: '0 0 10px #00FF95', marginTop: 7 }} />
                  <div>
                    <strong>Coach feedback</strong>
                    <div style={{ color: '#9ca3af' }}>Instant, actionable tips to improve each rep.</div>
                  </div>
                </li>
              </ul>

              <div style={{ marginTop: 20 }}>
                <button
                  onClick={onStart}
                  onMouseEnter={() => setHover(true)}
                  onMouseLeave={() => setHover(false)}
                  style={{
                    appearance: 'none',
                    border: 'none',
                    background: '#00FF95',
                    color: '#0b0f12',
                    fontWeight: 700,
                    fontSize: '1rem',
                    padding: '14px 22px',
                    borderRadius: 9999,
                    cursor: 'pointer',
                    boxShadow: hover ? '0 0 30px rgba(0,255,149,0.6), 0 10px 24px rgba(0,0,0,0.5)' : '0 0 18px rgba(0,255,149,0.45), 0 6px 16px rgba(0,0,0,0.4)',
                    transform: hover ? 'translateY(-2px)' : 'translateY(0)',
                    transition: 'all 180ms ease'
                  }}
                >
                  Start Exercise Analysis
                </button>
                <div style={{ marginTop: 8, color: '#6b7280', fontSize: 12 }}>
                  Works with your webcam. No data saved.
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: 16, color: '#6b7280', fontSize: 12 }}>
          Tip: Make sure your whole body is visible for the best results.
        </div>
      </div>
    </div>
  );
}

export default LandingPage;