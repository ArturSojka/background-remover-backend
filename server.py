from fastapi import FastAPI, File, UploadFile, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
import numpy as np
import io
import json
from infer import infer_image
from editing import edit

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# settings format:
# {
#   settings: {
#     aiModel: 'original' | 'custom',
#     background: {
#       type: 'original' | 'image' | 'color' | 'none',
#       blur: boolean,
#       color: '#ffffff',
#       image: File | null
#     },
#     effect: {
#       type: 'none' | 'border',
#       color: '#000000'
#     },
#     blendingMethod: 'simple' | 'advanced'
#   }
# }

@app.post("/api/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    settings: str = Form(...),
    backgroundImage: Optional[UploadFile] = File(None)
):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    try:
        settings_dict = json.loads(settings)

        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        bg_img = None
        if backgroundImage:
            bg_bytes = await backgroundImage.read()
            bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGB")
        
        alpha = infer_image(img, settings_dict['aiModel']=='custom')
        alpha = Image.fromarray(((alpha * 255).astype('uint8')), mode='L')

        result_img = edit(img, alpha, settings_dict, bg_img)

        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
