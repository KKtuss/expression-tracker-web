import asyncio
import json
import base64
import logging
import uuid
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Import our detection modules (exact copies from original project)
from detection_core import ExpressionDetector
from image_manager import ImageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Expression Tracker Web API", version="2.0.0")

# Add CORS middleware (allow all origins for web deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections and their detection sessions
active_connections: Dict[str, WebSocket] = {}
detection_sessions: Dict[str, Dict] = {}

class ConnectionManager:
    """Manages WebSocket connections and detection sessions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.detection_sessions: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and initialize detection session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Create isolated detection session
        try:
            expression_detector = ExpressionDetector()
            image_manager = ImageManager(expression_detector.images)
            
            self.detection_sessions[session_id] = {
                'detector': expression_detector,
                'image_manager': image_manager,
                'frame_count': 0,
                'last_expression': None
            }
            
            logger.info(f"Session {session_id} initialized with ExpressionDetector and ImageManager")
            
            # Send initial status
            await self.send_message(session_id, {
                "type": "status",
                "message": "Connected and ready",
                "session_id": session_id
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize session {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Initialization failed: {str(e)}"
            })
    
    def disconnect(self, session_id: str):
        """Clean up session resources when client disconnects"""
        if session_id in self.detection_sessions:
            session = self.detection_sessions[session_id]
            
            # Clean up detection resources
            try:
                if 'detector' in session:
                    session['detector'].cleanup()
                logger.info(f"Cleaned up resources for session {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
            
            # Remove from active sessions
            del self.detection_sessions[session_id]
        
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        logger.info(f"Session {session_id} disconnected and cleaned up")
    
    async def send_message(self, session_id: str, message: dict):
        """Send message to specific WebSocket connection"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to session {session_id}: {e}")
                self.disconnect(session_id)
    
    async def process_frame(self, session_id: str, frame_data: str) -> dict:
        """Process frame for specific session"""
        if session_id not in self.detection_sessions:
            return {"type": "error", "message": "Session not found"}
        
        session = self.detection_sessions[session_id]
        detector = session['detector']
        image_manager = session['image_manager']
        
        try:
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"type": "error", "message": "Could not decode frame"}
            
            # Process frame using exact original detection logic
            detection_result = detector.process_frame(frame)
            
            # Get current expression image
            current_expression = detection_result.get("expression")
            expression_image_base64 = None
            
            if current_expression and current_expression != session['last_expression']:
                expression_image_base64 = image_manager.get_image_base64(current_expression)
                session['last_expression'] = current_expression
            
            # Update frame count
            session['frame_count'] += 1
            
            # Return comprehensive result
            result = {
                "type": "detection_result",
                "session_id": session_id,
                "frame_count": session['frame_count'],
                "expression": detection_result.get("expression"),
                "success": detection_result.get("success", False),
                "debug": detection_result.get("debug", {}),
                "frame_with_overlay": detection_result.get("frame_with_overlay"),
                "expression_image": expression_image_base64
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame for session {session_id}: {e}")
            return {"type": "error", "message": f"Processing error: {str(e)}"}

# Initialize connection manager
manager = ConnectionManager()

@app.get("/")
async def main_app():
    """Serve the main application"""
    try:
        with open("frontend.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(html_content)
    except FileNotFoundError:
        return {
            "message": "Expression Tracker Web API",
            "version": "2.0.0",
            "status": "healthy",
            "active_sessions": len(manager.active_connections),
            "websocket_endpoint": "/ws/{session_id}",
            "test_page": "/test",
            "note": "Main frontend not found, serving API info"
        }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "active_sessions": len(manager.active_connections),
        "supported_features": [
            "real_time_detection",
            "face_detection",
            "hand_gesture_recognition",
            "facial_expression_analysis",
            "gaze_tracking",
            "image_upload"
        ]
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time detection"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Process frame
                frame_data = message.get("data")
                if frame_data:
                    result = await manager.process_frame(session_id, frame_data)
                    await manager.send_message(session_id, result)
                else:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": "No frame data provided"
                    })
            
            elif message.get("type") == "upload_image":
                # Handle image upload for specific expression
                expression_type = message.get("expression_type")
                image_data = message.get("image_data")
                
                if expression_type and image_data:
                    try:
                        session = manager.detection_sessions.get(session_id)
                        if session:
                            image_manager = session['image_manager']
                            success = image_manager.set_image_from_base64(expression_type, image_data)
                            
                            if success:
                                await manager.send_message(session_id, {
                                    "type": "upload_success",
                                    "expression_type": expression_type,
                                    "message": f"Image uploaded for {expression_type}"
                                })
                            else:
                                await manager.send_message(session_id, {
                                    "type": "error",
                                    "message": f"Failed to upload image for {expression_type}"
                                })
                        else:
                            await manager.send_message(session_id, {
                                "type": "error",
                                "message": "Session not found"
                            })
                    except Exception as e:
                        await manager.send_message(session_id, {
                            "type": "error",
                            "message": f"Upload failed: {str(e)}"
                        })
            
            elif message.get("type") == "clear_images":
                # Clear all images for session
                try:
                    session = manager.detection_sessions.get(session_id)
                    if session:
                        image_manager = session['image_manager']
                        image_manager.clear_all_images()
                        
                        await manager.send_message(session_id, {
                            "type": "clear_success",
                            "message": "All images cleared"
                        })
                    else:
                        await manager.send_message(session_id, {
                            "type": "error",
                            "message": "Session not found"
                        })
                except Exception as e:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": f"Clear failed: {str(e)}"
                    })
            
            elif message.get("type") == "get_expressions":
                # Get available expressions
                try:
                    session = manager.detection_sessions.get(session_id)
                    if session:
                        image_manager = session['image_manager']
                        expressions = image_manager.get_all_expressions()
                        loaded_expressions = image_manager.get_loaded_expressions()
                        
                        await manager.send_message(session_id, {
                            "type": "expressions",
                            "expressions": expressions,
                            "loaded_expressions": loaded_expressions
                        })
                    else:
                        await manager.send_message(session_id, {
                            "type": "error",
                            "message": "Session not found"
                        })
                except Exception as e:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": f"Failed to get expressions: {str(e)}"
                    })
            
            elif message.get("type") == "ping":
                # Respond to ping
                await manager.send_message(session_id, {
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                })
            
            else:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message.get('type')}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)

@app.get("/sessions")
async def get_active_sessions():
    """Get information about active sessions (for monitoring)"""
    return {
        "active_sessions": len(manager.active_connections),
        "session_ids": list(manager.active_connections.keys()),
        "session_details": {
            session_id: {
                "frame_count": session.get("frame_count", 0),
                "last_expression": session.get("last_expression"),
                "initialized": "detector" in session
            }
            for session_id, session in manager.detection_sessions.items()
        }
    }

@app.post("/benchmark")
async def run_benchmark():
    """Run performance benchmark (for testing instance performance)"""
    try:
        import subprocess
        import sys
        import os
        
        # Check if benchmark script exists
        if not os.path.exists("benchmark.py"):
            return {
                "success": False,
                "error": "benchmark.py not found"
            }
        
        # Run benchmark in background
        result = subprocess.run(
            [sys.executable, "benchmark.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            "success": True,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Benchmark timed out after 5 minutes"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Benchmark failed: {str(e)}"
        }

@app.get("/presets")
async def get_presets():
    """Get list of available presets"""
    try:
        # Create a temporary ImageManager to access preset methods
        temp_images = {}
        temp_manager = ImageManager(temp_images)
        presets = temp_manager.list_presets()
        
        return {
            "success": True,
            "presets": presets
        }
    except Exception as e:
        logger.error(f"Error getting presets: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/presets/{preset_name}")
async def save_preset(preset_name: str, request: dict):
    """Save current expression images as a preset"""
    try:
        # Get the session_id from request or create a default one
        session_id = request.get("session_id", "default")
        
        # Get session's image manager, create if doesn't exist
        session = manager.detection_sessions.get(session_id)
        if not session:
            # Create a temporary session for preset operations
            await manager.initialize_session(session_id)
            session = manager.detection_sessions.get(session_id)
            if not session:
                return {
                    "success": False,
                    "error": "Failed to create session"
                }
        
        image_manager = session['image_manager']
        success = image_manager.save_preset(preset_name)
        
        if success:
            return {
                "success": True,
                "message": f"Preset '{preset_name}' saved successfully"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to save preset '{preset_name}'"
            }
            
    except Exception as e:
        logger.error(f"Error saving preset: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/presets/{preset_name}/load")
async def load_preset(preset_name: str, request: dict):
    """Load a preset"""
    try:
        # Get the session_id from request
        session_id = request.get("session_id", "default")
        
        # Get session's image manager, create if doesn't exist
        session = manager.detection_sessions.get(session_id)
        if not session:
            # Create a temporary session for preset operations
            await manager.initialize_session(session_id)
            session = manager.detection_sessions.get(session_id)
            if not session:
                return {
                    "success": False,
                    "error": "Failed to create session"
                }
        
        image_manager = session['image_manager']
        success = image_manager.load_preset(preset_name)
        
        if success:
            return {
                "success": True,
                "message": f"Preset '{preset_name}' loaded successfully"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to load preset '{preset_name}'"
            }
            
    except Exception as e:
        logger.error(f"Error loading preset: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/presets/{preset_name}")
async def delete_preset(preset_name: str):
    """Delete a preset"""
    try:
        # Create a temporary ImageManager to access preset methods
        temp_images = {}
        temp_manager = ImageManager(temp_images)
        success = temp_manager.delete_preset(preset_name)
        
        if success:
            return {
                "success": True,
                "message": f"Preset '{preset_name}' deleted successfully"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to delete preset '{preset_name}'"
            }
            
    except Exception as e:
        logger.error(f"Error deleting preset: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Frontend HTML (simple test interface)
@app.get("/test", response_class=HTMLResponse)
async def test_frontend():
    """Simple test frontend for WebSocket connection"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Expression Tracker Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #000; color: #0f0; }
            .container { max-width: 800px; margin: 0 auto; }
            video, canvas { border: 1px solid #0f0; margin: 10px 0; }
            button { background: #000; color: #0f0; border: 1px solid #0f0; padding: 10px 20px; margin: 5px; cursor: pointer; }
            button:hover { background: #0f0; color: #000; }
            .status { padding: 10px; border: 1px solid #0f0; margin: 10px 0; }
            .debug { font-family: monospace; font-size: 12px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Expression Tracker Web Test</h1>
            <div id="status" class="status">Connecting...</div>
            <video id="video" width="640" height="480" autoplay muted></video>
            <canvas id="canvas" width="640" height="480"></canvas>
            <br>
            <button id="startCamera">Start Camera</button>
            <button id="startDetection">Start Detection</button>
            <button id="stopDetection">Stop Detection</button>
            <div id="debug" class="debug"></div>
        </div>
        
        <script>
            const sessionId = Math.random().toString(36).substr(2, 9);
            let ws = null;
            let video = null;
            let canvas = null;
            let ctx = null;
            let isProcessing = false;
            let processingInterval = null;
            
            document.getElementById('startCamera').onclick = async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video = document.getElementById('video');
                    canvas = document.getElementById('canvas');
                    ctx = canvas.getContext('2d');
                    video.srcObject = stream;
                    document.getElementById('status').textContent = 'Camera started';
                } catch (error) {
                    document.getElementById('status').textContent = 'Camera error: ' + error.message;
                }
            };
            
            document.getElementById('startDetection').onclick = () => {
                if (!ws) {
                    ws = new WebSocket(`wss://expression-tracker-web.onrender.com/ws/${sessionId}`);
                    ws.onopen = () => {
                        document.getElementById('status').textContent = 'Connected to WebSocket';
                        startProcessing();
                    };
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.type === 'detection_result') {
                            document.getElementById('debug').textContent = JSON.stringify(data, null, 2);
                        }
                    };
                    ws.onclose = () => {
                        document.getElementById('status').textContent = 'Disconnected';
                        ws = null;
                    };
                } else {
                    startProcessing();
                }
            };
            
            document.getElementById('stopDetection').onclick = () => {
                stopProcessing();
            };
            
            function startProcessing() {
                if (!isProcessing && video) {
                    isProcessing = true;
                    processingInterval = setInterval(processFrame, 100);
                    document.getElementById('status').textContent = 'Processing frames...';
                }
            }
            
            function stopProcessing() {
                if (isProcessing) {
                    isProcessing = false;
                    clearInterval(processingInterval);
                    document.getElementById('status').textContent = 'Processing stopped';
                }
            }
            
            function processFrame() {
                if (!video || !ws) return;
                
                ctx.drawImage(video, 0, 0);
                const frameData = canvas.toDataURL('image/jpeg').split(',')[1];
                
                ws.send(JSON.stringify({
                    type: 'frame',
                    data: frameData
                }));
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get("PORT", 8001))
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload for production
        log_level="info"
    )

