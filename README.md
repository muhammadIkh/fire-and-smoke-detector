Real-Time Smoke and Fire Detection with YOLOv8
<!-- Replace with a URL to your application screenshot -->

This project is a full-stack web application capable of detecting the presence of fire and smoke in real-time using a webcam feed. The application is built on a client-server architecture, combining an interactive frontend with a powerful AI backend based on the YOLOv8 model.

✨ Key Features
Real-Time Detection: Analyzes video feeds directly from a webcam for instant detection.

High Accuracy: Utilizes a custom-trained YOLOv8 model on a smoke and fire detection dataset.

Responsive Interface: A clean and modern frontend built with Tailwind CSS, accessible from various devices.

GPU Acceleration: The backend automatically leverages a GPU (CUDA) if available for significantly faster detection performance.

Clear Visualization: Detection results are visually displayed with bounding boxes and confidence scores directly over the video feed.

🏗️ System Architecture
The application operates on a client-server model:

Frontend (Client): Built with HTML, CSS (Tailwind), and Vanilla JavaScript. It is responsible for accessing the webcam, capturing video frames, and sending them to the backend.

Backend (Server): Built with Python using the FastAPI framework. This server receives images from the frontend, processes them using the pre-loaded YOLOv8 model, and sends back the detection coordinates in JSON format.

🚀 How to Run the Project
Follow these steps to run this project on your local machine.

Prerequisites
Python 3.8 or newer

A webcam connected to your computer

(Optional but highly recommended) An NVIDIA GPU with CUDA drivers installed for the best performance.

1. Backend Setup
First, set up the backend server.

# 1. Clone this repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install all required libraries
pip install -r requirements.txt

# 4. Place your model file
# Make sure the 'best.pt' file is in the same directory as 'app.py'

# 5. Run the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000

The backend server is now running at http://127.0.0.1:8000.

2. Frontend Setup
Simply open the index.html file in a modern web browser like Google Chrome or Firefox.

Double-click the index.html file, or

Open your browser and navigate to the file path.

3. Start Detection
Open the index.html page.

Allow the browser to access your camera when prompted.

Click the "Start Detection" button.

🧠 Model Performance
The model used is a YOLOv8n custom-trained on a smoke and fire detection dataset. Here is a summary of the model's performance on the validation data:

<!-- Replace with a URL to your model evaluation curves image -->

mAP@0.5: 0.726 (72.6%)

Smoke Class Accuracy (AP): 78.8%

Fire Class Accuracy (AP): 46.4%

Optimal Confidence Threshold: 0.307

The model shows excellent performance in detecting smoke but needs further improvement for fire detection. Using a confidence threshold of around 0.3 is highly recommended to get the best balance between precision and recall.

🛠️ Technology Stack
Frontend:

HTML5

Tailwind CSS

Vanilla JavaScript

Backend:

Python

FastAPI

Uvicorn

AI / Machine Learning:

Ultralytics YOLOv8

PyTorch

OpenCV

NumPy

requirements.txt File
Create a new file named requirements.txt in your project directory and fill it with the following text. This will simplify the backend installation process.

fastapi
uvicorn[standard]
torch
torchvision
ultralytics
opencv-python-headless
numpy
python-multipart
