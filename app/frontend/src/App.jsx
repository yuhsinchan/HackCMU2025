import { useState, useEffect, useRef } from 'react'
import './App.css'
import LandingPage from './components/LandingPage'

function App() {
  const [isStarted, setIsStarted] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [error, setError] = useState(null)
  const [lastFrameTime, setLastFrameTime] = useState(null)
  const [connectionQuality, setConnectionQuality] = useState('checking')
  const [counter, setCounter] = useState(0)
  const [pred, setPred] = useState(-1)
  const imgRef = useRef(null)
  const currentAudioRef = useRef(null)
  const lastCounterRef = useRef(-1)
  const lastPredRef = useRef(-1)

  // Celebration state + streak + confetti
  const [celebrate, setCelebrate] = useState(false)
  const celebrateTimeoutRef = useRef(null)
  const [streak, setStreak] = useState(0)
  const videoContainerRef = useRef(null)
  const confettiCanvasRef = useRef(null)
  const confettiRafRef = useRef(null)

  const triggerCelebrate = () => {
    if (celebrateTimeoutRef.current) {
      clearTimeout(celebrateTimeoutRef.current)
    }
    setCelebrate(true)
    celebrateTimeoutRef.current = setTimeout(() => setCelebrate(false), 1200)
  }

  const triggerConfettiBurst = () => {
    const container = videoContainerRef.current
    const canvas = confettiCanvasRef.current
    if (!container || !canvas) return

    // Size canvas to container
    const { clientWidth: w, clientHeight: h } = container
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const colors = ['#00FF95', '#34D399', '#A7F3D0', '#ffffff']
    const count = Math.min(140, Math.max(60, Math.floor((w * h) / 7000)))
    const particles = new Array(count).fill(0).map(() => {
      const angle = (Math.random() * Math.PI) - Math.PI / 2
      const speed = 6 + Math.random() * 7
      return {
        x: w * 0.5,
        y: h * 0.4,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 2,
        g: 0.25 + Math.random() * 0.35,
        size: 4 + Math.random() * 6,
        rot: Math.random() * Math.PI,
        vr: (Math.random() - 0.5) * 0.3,
        color: colors[(Math.random() * colors.length) | 0],
        life: 900 + Math.random() * 500
      }
    })

    const start = performance.now()
    const draw = (t) => {
      const elapsed = t - start
      ctx.clearRect(0, 0, w, h)
      particles.forEach(p => {
        p.vy += p.g
        p.x += p.vx
        p.y += p.vy
        p.rot += p.vr
        p.life -= 16
        ctx.save()
        ctx.translate(p.x, p.y)
        ctx.rotate(p.rot)
        ctx.globalAlpha = Math.max(0, Math.min(1, p.life / 400))
        ctx.fillStyle = p.color
        ctx.fillRect(-p.size * 0.5, -p.size * 0.5, p.size, p.size)
        ctx.restore()
      })

      if (elapsed < 1200) {
        confettiRafRef.current = requestAnimationFrame(draw)
      } else {
        ctx.clearRect(0, 0, w, h)
        if (confettiRafRef.current) cancelAnimationFrame(confettiRafRef.current)
        confettiRafRef.current = null
      }
    }

    if (confettiRafRef.current) cancelAnimationFrame(confettiRafRef.current)
    confettiRafRef.current = requestAnimationFrame(draw)
  }

  // Audio mapping function
  const getAudioFile = (prediction) => {
    const audioFiles = {
      0: '/audio/Correct.mp3',
      1: '/audio/Error0.mp3',
      2: '/audio/Error1.mp3',
      3: '/audio/Error2.mp3',
      4: '/audio/Error3.mp3'
    }
    return audioFiles[prediction]
  }

  // Handle audio playback
  const playAudio = (prediction) => {
    if (prediction === -1) return
    
    const audioFile = getAudioFile(prediction)
    if (!audioFile) return

    // Stop any current audio before playing new one
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current.currentTime = 0
    }

    // Create and play new audio
    const audio = new Audio(audioFile)
    audio.play().catch(e => console.error('Error playing audio:', e))
    currentAudioRef.current = audio
  }

  // Mapping function to convert prediction numbers to feedback messages
  const getCoachWords = (prediction) => {
    const messages = {
      0: 'Nice work! Your stance and movement look solid—keep that same control and depth. Great form, keep it up!',
      1: 'Your feet are a bit too wide—try bringing them in closer to about shoulder-width. That’ll give you better balance and let you drive more power through your legs.',
      2: 'Bring your feet a bit wider apart—right now they’re too close, which limits balance and depth. A shoulder-width stance will give you more stability and let your hips move naturally through the squat.',
      3: 'Watch your knees—they’re caving in (knee valgus). Focus on pushing them outward in line with your toes to keep your joints safe and maintain proper squat mechanics.',
      4: 'You’re not squatting deep enough. Aim to lower until your thighs are at least parallel to the ground for full range of motion and better results.'
    }
    return messages[prediction] || 'Analyzing your form...'
  }

  // Mapping function to convert prediction numbers to feedback messages
  const getErrorType = (prediction) => {
    const messages = {
      0: 'Correct',
      1: 'Feet too wide',
      2: 'Feet too close',
      3: 'Knee too close',
      4: 'Squat shallow'
    }
    return 'Coach Feedback: ' + (messages[prediction] || '')
  }

  const frameCountRef = useRef(0)
  const lastFrameTimeRef = useRef(Date.now())

  const handleStart = async () => {
    try {
      console.log('Sending start request to backend...');
      const response = await fetch('http://172.26.203.22:8000/start', {
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
    const ws = new WebSocket(`ws://172.26.203.22:8000/ws`)
    
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
                } else if (msg.type === 'counter') {
          const newCount = msg.count
          const newPred = msg.prediction
          // Only play audio when counter changes (new rep)
          if (newCount !== lastCounterRef.current) {
            playAudio(newPred)
            if (newPred === 0) {
              triggerCelebrate()
              triggerConfettiBurst()
              setStreak((s) => s + 1)
            } else {
              setStreak(0)
            }
            lastCounterRef.current = newCount
          }
          if (newPred !== -1) {
            setCounter(newCount)
            setPred(newPred)
          }
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
      // Clean up audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause()
        currentAudioRef.current = null
      }
      if (celebrateTimeoutRef.current) {
        clearTimeout(celebrateTimeoutRef.current)
        celebrateTimeoutRef.current = null
      }
      if (confettiRafRef.current) {
        cancelAnimationFrame(confettiRafRef.current)
        confettiRafRef.current = null
      }
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
              <h1 style={{ margin: 0, fontSize: '2.5rem', color: '#00FF95' }}>RepCheck</h1>
              <p style={{ margin: '8px 0 0', color: '#9ca3af' }}>Real-time exercise form analysis and feedback</p>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
              <div style={{ background: '#111827', borderRadius: 12, padding: 8 }}>
                <div style={{ position: 'relative' }} ref={videoContainerRef}>
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
                  <div className={`shine-sweep ${celebrate ? 'show' : ''}`} />
                  <canvas ref={confettiCanvasRef} className="confetti-canvas" />
                  <div className={`streak-badge ${streak >= 2 ? 'show' : ''}`}>Streak ×{streak}</div>
                  <div className={`perfect-toast ${celebrate ? 'show' : ''}`}>Perfect rep!</div>
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
                    <h2 style={{ margin: 0, fontSize: '1.5rem' }}>Counter</h2>
                  </div>
                  <p style={{ 
                    margin: 0, 
                    color: '#d1d5db',
                    fontSize: '2.5rem',
                    lineHeight: 1.5,
                    textAlign: 'center',
                    fontWeight: 'bold'
                  }}>
                    {counter}
                  </p>
                </div>

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
                    <h2 style={{ margin: 0, fontSize: '1.5rem' }}>{getErrorType(pred)}</h2>
                  </div>
                  <p style={{ 
                    margin: 0, 
                    color: '#d1d5db',
                    fontSize: '1.1rem',
                    lineHeight: 1.5
                  }}>
                    {getCoachWords(pred)}
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