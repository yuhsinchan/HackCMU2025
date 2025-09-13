import { useState, useEffect, useRef } from 'react'
import './App.css'
import LandingPage from './components/LandingPage'

function App() {
  const [isStarted, setIsStarted] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [error, setError] = useState(null)
  const [lastFrameTime, setLastFrameTime] = useState(null)
  const [connectionQuality, setConnectionQuality] = useState('checking')
  const [counter, setCounter] = useState(-1)
  const [pred, setPred] = useState(-1)
  const imgRef = useRef(null)
  const currentAudioRef = useRef(null)
  const lastPredRef = useRef(-1)

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
    if (prediction === -1) {
      console.log('Skipping audio for initial state')
      return
    }
    
    const audioFile = getAudioFile(prediction)
    console.log('Attempting to play audio:', audioFile)
    if (!audioFile) {
      console.log('No audio file found for prediction:', prediction)
      return
    }

    // If it's the same prediction and audio is still playing, don't play again
    if (prediction === lastPredRef.current && currentAudioRef.current) {
      const isPlaying = !currentAudioRef.current.paused && 
                       !currentAudioRef.current.ended && 
                       currentAudioRef.current.currentTime > 0
      if (isPlaying) {
        console.log('Same prediction and audio still playing, skipping')
        return
      }
    }

    // Stop any current audio before playing new one
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current.currentTime = 0
    }

    // Create and play new audio
    console.log('Creating new audio for file:', audioFile)
    const audio = new Audio(audioFile)
    
    // Add event listeners for debugging
    audio.addEventListener('playing', () => {
      console.log('Audio started playing:', audioFile)
    })
    audio.addEventListener('error', (e) => {
      console.error('Audio error:', e)
    })
    audio.addEventListener('ended', () => {
      console.log('Audio finished playing:', audioFile)
    })

    audio.play()
      .then(() => console.log('Audio play promise resolved'))
      .catch(e => console.error('Error playing audio:', e, audioFile))
    
    currentAudioRef.current = audio
    lastPredRef.current = prediction
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
          console.log('Received counter update:', msg.count)
          console.log('Received prediction update:', msg.prediction)
          setCounter(msg.count)
          setPred(msg.prediction)
          playAudio(msg.prediction)
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