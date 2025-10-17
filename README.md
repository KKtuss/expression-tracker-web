# Expression Tracker Web Deployment

This is the web deployment version of the Expression Tracker, providing real-time facial expression and hand gesture detection via WebSockets.

## Features

- **Real-time detection** via WebSocket connections
- **Isolated user sessions** - each WebSocket connection gets its own detection instance
- **Full resolution processing** - maintains 640×480 input resolution
- **Complete feature parity** with the local version
- **Multi-user support** with proper resource management
- **Auto-scaling** support for cloud deployment

## Architecture

- **FastAPI backend** with WebSocket support
- **Session isolation** - each user gets their own ExpressionDetector instance
- **Resource cleanup** - automatic cleanup when users disconnect
- **JSON messaging** for real-time communication

## Deployment Options

### Option 1: Railway (Recommended)
1. Push this folder to a new GitHub repository
2. Connect Railway to your GitHub repository
3. Railway will auto-detect the deployment configuration
4. The app will be available at `https://your-app-name.railway.app`

### Option 2: Render
1. Push this folder to a new GitHub repository
2. Create a new Web Service on Render
3. Connect to your GitHub repository
4. Use Docker deployment mode
5. Render will use the `render.yaml` configuration

### Option 3: Google Cloud Run
1. Build and push the Docker image to Google Container Registry
2. Deploy using Cloud Run with the provided Dockerfile

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health information
- `GET /sessions` - Active session information
- `WS /ws/{session_id}` - WebSocket endpoint for real-time detection
- `GET /test` - Simple test frontend

## WebSocket Protocol

### Client → Server Messages
```json
{
  "type": "frame",
  "data": "base64_encoded_frame"
}

{
  "type": "upload_image",
  "expression_type": "smiling",
  "image_data": "base64_encoded_image"
}

{
  "type": "clear_images"
}

{
  "type": "get_expressions"
}

{
  "type": "ping",
  "timestamp": 1234567890
}
```

### Server → Client Messages
```json
{
  "type": "detection_result",
  "session_id": "abc123",
  "frame_count": 42,
  "expression": "smiling",
  "success": true,
  "debug": {...},
  "frame_with_overlay": "base64_encoded_frame",
  "expression_image": "base64_encoded_image"
}

{
  "type": "status",
  "message": "Connected and ready",
  "session_id": "abc123"
}

{
  "type": "error",
  "message": "Error description"
}
```

## Performance

- **Full resolution**: 640×480 input frames
- **Real-time processing**: ~10 FPS per session
- **Multi-user**: Supports multiple concurrent users
- **Resource efficient**: Automatic cleanup on disconnect
- **Scalable**: Auto-scaling support for cloud platforms

## Requirements

- Python 3.12
- OpenCV with MediaPipe
- FastAPI with WebSocket support
- 1GB+ RAM per concurrent user
- Modern web browser with WebSocket support

## Security

- CORS enabled for web access
- Session isolation between users
- Automatic resource cleanup
- Non-root user in Docker container
- Health check endpoints for monitoring
