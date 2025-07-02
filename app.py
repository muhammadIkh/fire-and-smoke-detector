# app.py
# Import library yang dibutuhkan
import cv2
import uvicorn
import numpy as np
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import logging
import torch  # <-- Diperlukan untuk mengecek ketersediaan GPU

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Konfigurasi CORS (Cross-Origin Resource Sharing)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEMILIH DEVICE (GPU/CPU) ---
# Secara otomatis memilih GPU (CUDA) jika tersedia, jika tidak, gunakan CPU
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Deteksi akan dijalankan menggunakan device: {device.upper()}")
    if device == 'cpu':
        logger.warning("CUDA tidak ditemukan. Model akan berjalan di CPU, performa mungkin lebih lambat.")
except Exception as e:
    logger.error(f"Terjadi error saat memeriksa device: {e}. Menggunakan CPU sebagai default.")
    device = 'cpu'


# --- MEMUAT MODEL YOLOv8 ---
# Model hanya dimuat sekali saat aplikasi dimulai
try:
    model_path = 'best.pt'
    model = YOLO(model_path)
    logger.info(f"Model YOLOv8 berhasil dimuat dari: {model_path}")
except Exception as e:
    logger.error(f"Error fatal saat memuat model: {e}")
    model = None

# Definisikan endpoint untuk deteksi
@app.post("/detect")
async def detect(request: Request):
    """
    Endpoint untuk menerima gambar, melakukan deteksi di device yang dipilih,
    dan mengirimkan kembali hasilnya.
    """
    if not model:
        return {"error": "Model tidak berhasil dimuat, server tidak bisa melakukan deteksi."}

    # Ambil data JSON dari request
    data = await request.json()
    # Decode gambar dari base64
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # --- LAKUKAN DETEKSI DENGAN DEVICE YANG DIPILIH (GPU/CPU) ---
    # `verbose=False` ditambahkan untuk mengurangi output log yang tidak perlu di setiap frame
    results = model(img, device=device, verbose=False)

    detections = []
    # Loop melalui hasil deteksi
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(confidence),
                "class_name": class_name,
            })
    
    # Tidak perlu log setiap frame agar tidak memenuhi konsol
    # logger.info(f"Deteksi selesai. Ditemukan {len(detections)} objek.")
    return {"detections": detections}

# Jalankan server menggunakan uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
