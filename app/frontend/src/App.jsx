import { useState, useEffect, useRef } from 'react'
import './App.css'
import LandingPage from './components/LandingPage'

function App() {
  const [isStarted, setIsStarted] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [error, setError] = useState(null)
  const [lastFrameTime, setLastFrameTime] = useState(null)
  const [connectionQuality, setConnectionQuality] = useState('checking')
  const [suggestion, setSuggestion] = useState('')
  const imgRef = useRef(null)
  const frameCountRef = useRef(0)
  const lastFrameTimeRef = useRef(Date.now())

  const handleStart = async () => {
    try {
      console.log('Sending start request to backend...');
      const response = await fetch('http://localhost:8000/start', {
        method: 'POST',
      });
      
      if (response.ok) {
        console.log('Video feed started successfully');
        setIsStarted(true);
      } else {
        const errorData = await response.json();
        console.error('Failed to start video feed:', errorData);
        setError('Failed to start video feed: ' + JSON.stringify(errorData));
      }
    } catch (err) {
      console.error('Error starting video feed:', err);
      setError('Error starting video feed: ' + err.message);
    }
  }

  useEffect(() => {
    if (!isStarted) return

    console.log('Setting up WebSocket connection...')
    // Use the current host for WebSocket connection
    const ws = new WebSocket(`ws://localhost:8000/ws`)
    
    ws.onopen = () => {
      console.log('WebSocket connection established')
      setConnectionStatus('Connected')
      setError(null)
      // Send initial ping to start the interaction
      ws.send('ping')
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        console.log('Received message type:', msg.type)
        if (msg.type === 'frame') {
          if (msg.image) {
            console.log('Received frame at:', new Date().toISOString())
            console.log('Frame data starts with:', msg.image.substring(0, 50))
            if (imgRef.current) {
              imgRef.current.src = `data:image/jpeg;base64,${msg.image}`
              setLastFrameTime(new Date().toLocaleTimeString())
            } else {
              console.warn('Image ref is not available')
            }
          } else {
            console.warn('Frame message missing image data')
          }
        } else if (msg.type === 'suggestion') {
          console.log('Received suggestion:', msg.message)
          setSuggestion(msg.message)
        }
      } catch (e) {
        console.error('Error processing message:', e)
        setError(`Error processing message: ${e.message}`)
      }
    }

    ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      setConnectionStatus('Error')
      setError('WebSocket error occurred')
    }

    ws.onclose = () => {
      console.log('WebSocket connection closed')
      setConnectionStatus('Disconnected')
    }

    // Keep connection alive
    const keepAlive = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping')
      }
    }, 30000)

    return () => {
      clearInterval(keepAlive)
      ws.close()
    }
  }, [isStarted])

  // Calculate connection quality
  useEffect(() => {
    if (connectionStatus === 'Connected') {
      const interval = setInterval(() => {
        const now = Date.now()
        const fps = frameCountRef.current / ((now - lastFrameTimeRef.current) / 1000)
        frameCountRef.current = 0
        lastFrameTimeRef.current = now
        
        if (fps > 20) {
          setConnectionQuality('excellent')
        } else if (fps > 10) {
          setConnectionQuality('good')
        } else if (fps > 5) {
          setConnectionQuality('fair')
        } else {
          setConnectionQuality('poor')
        }
      }, 1000)
      return () => clearInterval(interval)
    }
  }, [connectionStatus])

  return (
    <div className="app-container">
      {!isStarted ? (
        <LandingPage onStart={handleStart} />
      ) : (
        <div style={{ minHeight: '100vh', background: '#0b0f12', color: '#e5e7eb', padding: 16 }}>
          <div style={{ maxWidth: 1280, margin: '0 auto' }}>
            <div style={{ marginBottom: 24, textAlign: 'center' }}>
              <h1 style={{ margin: 0, fontSize: '2.5rem', color: '#00FF95' }}>Gymaster</h1>
              <p style={{ margin: '8px 0 0', color: '#9ca3af' }}>Real-time exercise form analysis and feedback</p>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
              <div style={{ background: '#111827', borderRadius: 12, padding: 8 }}>
                <div style={{ position: 'relative' }}>
                  <img 
                    ref={imgRef} 
                    alt="Video feed" 
                    style={{ width: '100%', borderRadius: 8, display: 'block' }} 
                  />
                  <div style={{ 
                    position: 'absolute', 
                    inset: 0, 
                    border: '2px solid #00FF95',
                    borderRadius: 8,
                    pointerEvents: 'none',
                    opacity: 0.3
                  }}/>
                </div>
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ background: '#111827', borderRadius: 12, padding: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                    <div style={{ 
                      width: 8, 
                      height: 8, 
                      borderRadius: '50%', 
                      background: '#00FF95',
                      boxShadow: '0 0 8px #00FF95',
                      animation: 'pulse 1.5s infinite'
                    }}/>
                    <h2 style={{ margin: 0, fontSize: '1.5rem' }}>Coach Feedback</h2>
                  </div>
                  <p style={{ 
                    margin: 0, 
                    color: '#d1d5db',
                    fontSize: '1.1rem',
                    lineHeight: 1.5
                  }}>
                    {suggestion || 'Analyzing your form...'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;