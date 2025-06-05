from fastapi import FastAPI, File, UploadFile, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
import numpy as np
import io
import json
import cv2
import tempfile
import os
from infer import infer_image
from editing import edit

app = FastAPI()

# Allow CORS for your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported file types
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
SUPPORTED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mov", "video/mkv", "video/webm"]

@app.post("/api/remove-background")
async def remove_background(
    file: UploadFile = File(...),
    settings: str = Form(...),
    backgroundImage: Optional[UploadFile] = File(None)
):
    # Check file type
    if file.content_type in SUPPORTED_IMAGE_TYPES:
        return await process_image(file, settings, backgroundImage)
    elif file.content_type in SUPPORTED_VIDEO_TYPES:
        return await process_video(file, settings, backgroundImage)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Supported: {SUPPORTED_IMAGE_TYPES + SUPPORTED_VIDEO_TYPES}"
        )

async def process_image(
    image: UploadFile,
    settings: str,
    backgroundImage: Optional[UploadFile] = None
):
    """Process single image"""
    try:
        # Parse settings JSON
        settings_dict = json.loads(settings)

        # Load original image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Load optional background image
        bg_img = None
        if backgroundImage:
            bg_bytes = await backgroundImage.read()
            bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGB")
        
        alpha = infer_image(img)
        alpha = Image.fromarray(((alpha * 255).astype('uint8')), mode='L')

        result_img = edit(img, alpha, settings_dict, bg_img)

        # Save result to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Image processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

async def process_video(
    video: UploadFile,
    settings: str,
    backgroundImage: Optional[UploadFile] = None
):
    """Process video file frame by frame"""
    try:
        # Parse settings JSON
        settings_dict = json.loads(settings)
        
        # Load optional background image
        bg_img = None
        if backgroundImage:
            bg_bytes = await backgroundImage.read()
            bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGB")

        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                
                # Save uploaded video to temp file
                video_bytes = await video.read()
                temp_input.write(video_bytes)
                temp_input.flush()
                
                # Process video
                process_video_file(temp_input.name, temp_output.name, settings_dict, bg_img)
                
                # Read processed video
                with open(temp_output.name, 'rb') as f:
                    result_bytes = f.read()
                
                # Cleanup temp files
                os.unlink(temp_input.name)
                os.unlink(temp_output.name)
                
                return Response(content=result_bytes, media_type="video/mp4")

    except Exception as e:
        print(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

def process_video_file(input_path: str, output_path: str, settings_dict: dict, bg_img: Optional[Image.Image] = None):
    """Process video file frame by frame"""
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:  # Progress logging
                print(f"Processing frame {frame_count}/{total_frames}")
            
            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            # Process frame through background removal
            try:
                alpha = infer_image(pil_frame)
                alpha = Image.fromarray(((alpha * 255).astype('uint8')), mode='L')
                
                # Resize background image to match frame if needed
                frame_bg_img = bg_img
                if bg_img and bg_img.size != pil_frame.size:
                    frame_bg_img = bg_img.resize(pil_frame.size, Image.LANCZOS)
                
                result_frame = edit(pil_frame, alpha, settings_dict, frame_bg_img)
                
                # Convert back to OpenCV format (BGR)
                result_array = np.array(result_frame.convert('RGB'))
                result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(result_bgr)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write original frame if processing fails
                out.write(frame)
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"Video processing complete: {frame_count} frames processed")

@app.post("/api/remove-background-batch")
async def remove_background_batch(
    files: list[UploadFile] = File(...),
    settings: str = Form(...),
    backgroundImage: Optional[UploadFile] = File(None)
):
    """Process multiple files at once"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    try:
        settings_dict = json.loads(settings)
        
        # Load optional background image
        bg_img = None
        if backgroundImage:
            bg_bytes = await backgroundImage.read()
            bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGB")
        
        results = []
        
        for file in files:
            if file.content_type in SUPPORTED_IMAGE_TYPES:
                # Process image
                image_bytes = await file.read()
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                alpha = infer_image(img)
                alpha = Image.fromarray(((alpha * 255).astype('uint8')), mode='L')
                
                result_img = edit(img, alpha, settings_dict, bg_img)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                result_img.save(img_byte_arr, format='PNG')
                
                results.append({
                    "filename": file.filename,
                    "type": "image",
                    "data": img_byte_arr.getvalue()
                })
            
            elif file.content_type in SUPPORTED_VIDEO_TYPES:
                # For batch processing, we might want to handle videos differently
                # or skip them for now
                results.append({
                    "filename": file.filename,
                    "type": "video",
                    "error": "Video batch processing not implemented yet"
                })
        
        # For now, return JSON with base64 encoded images
        # In production, you might want to return a ZIP file
        import base64
        
        response_data = []
        for result in results:
            if "error" not in result:
                response_data.append({
                    "filename": result["filename"],
                    "type": result["type"],
                    "data": base64.b64encode(result["data"]).decode()
                })
            else:
                response_data.append({
                    "filename": result["filename"],
                    "type": result["type"],
                    "error": result["error"]
                })
        
        return {"results": response_data}
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/api/supported-formats")
async def get_supported_formats():
    """Return supported file formats"""
    return {
        "images": SUPPORTED_IMAGE_TYPES,
        "videos": SUPPORTED_VIDEO_TYPES
    }

@app.get("/")
async def root():
    return {"message": "Background Removal API", "endpoints": ["/api/remove-background", "/api/remove-background-batch", "/api/supported-formats"]}
