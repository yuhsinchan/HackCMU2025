import asyncio
import base64
import argparse
from typing import Dict, Any, List

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['zed', 'webcam'], 
                   default='webcam', help='Camera mode: zed or webcam')
args = parser.parse_args()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Store global counter
exercise_counter: int = 0
prediction: int = 0

COACH_SUGGESTION = "Keep your back straight and maintain good form!"

async def broadcast_counter():
    """Send current counter value to all clients"""
    counter_data = {
        'type': 'counter',
        'count': exercise_counter,
        'prediction': prediction
    }
    await broadcast_frame(counter_data)

async def broadcast_frame(frame_data: Dict[str, Any]):
    """Send frame data to all connected clients"""
    if not active_connections:
        return  # No clients connected
        
    dead_connections = []
    for connection in active_connections:
        try:
            await connection.send_json(frame_data)
        except Exception as e:
            print(f"Error sending frame: {e}")
            dead_connections.append(connection)
    
    # Remove dead connections
    for dead in dead_connections:
        print("Removing dead connection")
        active_connections.remove(dead)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global exercise_counter  # Declare global at the start of the function
    global prediction
    print("New client connecting...")
    try:
        await websocket.accept()
        print("Client connected successfully")
        active_connections.append(websocket)
        print(f"Active connections: {len(active_connections)}")
        
        while True:
            # Keep connection alive
            msg = await websocket.receive_text()
            print(f"Received from client: {msg}")
            if msg == 'ping':
                print("Received ping, sending pong")
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        print("Client disconnected")
        if websocket in active_connections:
            active_connections.remove(websocket)
            # Reset counter if this was the last client
            if len(active_connections) == 0:
                print("All clients disconnected, reset counter to 0")
                exercise_counter = 0
                prediction = 0
            print(f"Active connections after disconnect: {len(active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
            # Reset counter if this was the last client
            if len(active_connections) == 0:
                print("All clients disconnected, reset counter to 0")
                exercise_counter = 0
                prediction = 0

async def webcam_feed():
    """Capture and broadcast frames from webcam"""
    global exercise_counter  # Declare global at the start of the function
    global prediction
    print("Initializing webcam feed...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                await asyncio.sleep(0.1)
                continue
            
            # Flip frame horizontally for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Show local preview window
            cv2.imshow("Local Webcam View", frame)
            
            # Handle window events (required for window to work properly)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            # Convert to base64
            frame_b64 = base64.b64encode(buffer).decode('ascii')
            
            # Prepare frame data
            frame_data = {
                'type': 'frame',
                'image': frame_b64
            }
            
            # Broadcast frame to all clients
            await broadcast_frame(frame_data)
            frame_count += 1
            
            # Every 3 seconds (90 frames at 30 FPS), increment and broadcast the counter
            if frame_count % 90 == 0:
                exercise_counter += 1
                prediction = (prediction + 1) % 7  # Cycle through predictions 0-6
            await broadcast_counter()

            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"Sent {frame_count} frames, current active connections: {len(active_connections)}")
            
            # Control frame rate (30 FPS)
            await asyncio.sleep(1/30)
    except Exception as e:
        print(f"Error in webcam feed: {e}")
    finally:
        cv2.destroyAllWindows()
        cap.release()

async def zed_feed():
    """Capture and broadcast frames from ZED camera with body tracking"""
    try:
        # Import ZED SDK and related modules
        import pyzed.sl as sl
        import cv_viewer.tracking_viewer as cv_viewer
        import ogl_viewer.viewer as gl
    except ImportError as e:
        print(f"Error: Required ZED packages not available: {e}")
        print("Please install ZED SDK and required packages for ZED mode")
        return
        
    # Initialize ZED camera
    zed = sl.Camera()
    
    # Create initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # Open camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error: ZED camera failed to open: {err}")
        return
    
    # Enable positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # Configure body tracking
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True
    body_param.enable_body_fitting = False
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_34
    
    # Enable body tracking
    zed.enable_body_tracking(body_param)
    
    # Runtime parameters
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    
    # Image and bodies containers
    image = sl.Mat()
    bodies = sl.Bodies()
    
    # Initialize 3D OpenGL viewer
    camera_info = zed.get_camera_information()
    
    # Setup display resolution and scaling
    display_resolution = sl.Resolution(
        min(camera_info.camera_configuration.resolution.width, 1280),
        min(camera_info.camera_configuration.resolution.height, 720)
    )
    image_scale = [
        display_resolution.width / camera_info.camera_configuration.resolution.width,
        display_resolution.height / camera_info.camera_configuration.resolution.height
    ]
    
    # Initialize OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(
        camera_info.camera_configuration.calibration_parameters.left_cam,
        body_param.enable_tracking,
        body_param.body_format
    )
    
    print("ZED camera and body tracking initialized successfully")
    
    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve image and bodies
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                zed.retrieve_bodies(bodies, body_runtime_param)
                
                # Get OpenCV image
                image_ocv = image.get_data()
                
                # Render skeletons on the image
                cv_viewer.render_2D(
                    image_ocv,
                    image_scale,  # Use calculated image scale
                    bodies.body_list,
                    body_param.enable_tracking,
                    body_param.body_format
                )
                
                # Show local OpenCV window
                cv2.imshow("ZED Body Tracking", image_ocv)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
                
                # Update 3D viewer with image and bodies data
                viewer.update_view(image, bodies)
                
                # Break the loop if viewer is closed
                if not viewer.is_available():
                    break
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', image_ocv, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                # Convert to base64
                frame_b64 = base64.b64encode(buffer).decode('ascii')
                
                # Broadcast frame to all clients
                await broadcast_frame({
                    'type': 'frame',
                    'image': frame_b64
                })

                # TODO: Implement counter logic based on body tracking data
                # exercise_counter += 1
                # await broadcast_counter()
                
                # Control frame rate (30 FPS)
                await asyncio.sleep(1/30)
    except Exception as e:
        print(f"Error in ZED feed: {e}")
    finally:
        cv2.destroyAllWindows()
        viewer.exit()
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()

# Store video feed tasks
video_tasks = {}

@app.post("/start")
async def start_video_feed():
    """Start video feed on request"""
    print("Received start video feed request")
    try:
        if not video_tasks.get('feed'):
            if args.mode == 'zed':
                print("Starting in ZED camera mode")
                video_tasks['feed'] = asyncio.create_task(zed_feed())
            else:
                print("Starting in webcam mode")
                video_tasks['feed'] = asyncio.create_task(webcam_feed())
            print("Video feed task created successfully")
        else:
            print("Video feed task already exists")
        return {"status": "Video feed started", "mode": args.mode}
    except Exception as e:
        print(f"Error starting video feed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def read_root():
    return {"status": "Video streaming server is running", "mode": args.mode}

@app.get("/suggestion")
async def get_suggestion():
    """Get the current coach's suggestion"""
    return {"suggestion": COACH_SUGGESTION}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server in {args.mode} mode...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    