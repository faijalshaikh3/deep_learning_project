# Real-Time Object Detection and Counting Using YOLO

## 1. Project Title
Real-Time Object Detection and Counting Using YOLO

## 2. Project Overview
This project is a beginner-friendly deep learning mini-project that demonstrates how a pretrained YOLO model can be used to detect objects in real time through a web application. The system can process live webcam feeds, uploaded images, and uploaded videos. It draws bounding boxes around objects, labels them with class names and confidence scores, and counts the number of detected objects by class.

The goal is to build a simple but practical application that can be explained clearly in a viva and easily demonstrated in a college setting.

## 3. Features
- Real-time webcam detection using OpenCV and YOLO
- Live pose estimation for hands raised, arms spread, and crouching
- Bounding boxes around detected objects
- Object labels and confidence percentage
- Counting of detected objects by category
- Total object count summary
- FPS display for the live detection feed
- Image upload detection
- Video upload detection
- Clean and responsive web interface
- Error handling for invalid files and unsupported formats

## 4. Technologies Used
- Python 3.12
- Flask
- Ultralytics YOLO
- OpenCV
- NumPy
- HTML5
- CSS3
- JavaScript
- Bootstrap (optional, not required here)

## 5. System Architecture
The system follows a simple client-server flow:

Browser
↓
HTML/CSS/JavaScript frontend
↓
Flask server
↓
OpenCV frame processing
↓
YOLO object detection model
↓
Bounding boxes, labels, counting, and statistics
↓
Response sent back to browser

This architecture is suitable for a mini-project because it is simple, understandable, and easy to explain.

## 6. Installation Instructions
1. Open a terminal in the project folder.
2. Create a Python virtual environment if you want to isolate dependencies.
3. Install the required dependencies.

Command:

```bash
pip install -r requirements.txt
```

## 7. How to Run the Project
From the project root, run:

```bash
python app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000/
```

## 8. How to Use Webcam Detection
1. Click the Start Camera button.
2. Allow webcam access if prompted by the browser.
3. The webcam feed will begin streaming in the browser.
4. YOLO will process each frame and draw detection boxes.
5. The system shows the detected object names and confidence values.
6. The activity panel estimates poses for each visible person.
7. The count panel updates automatically.
8. Click Stop Camera to stop the webcam feed.

Live pose estimation uses the pretrained Ultralytics `yolov8n-pose.pt` model. On first use, Ultralytics may download its weights. The pose labels are based on detected body keypoints and simple pose rules; they are not a separately trained activity classifier. For best results, keep one person fully visible in good lighting. Running pose estimation alongside object detection may reduce webcam FPS, especially on a CPU.

## 9. How to Upload an Image
1. Click the Upload Image section.
2. Select an image file.
3. Press the Process Image button.
4. The system will process the uploaded image and display the output image with bounding boxes and labels.
5. Detection counts are shown in the statistics area.

## 10. How to Upload a Video
1. Select a video file in the video upload section.
2. Click the Process Video button.
3. The server processes the video frame by frame using YOLO.
4. The output video is displayed with bounding boxes and labels.
5. Object counts are computed across the full video.

## 11. Explanation of YOLO
YOLO stands for You Only Look Once. It is a real-time object detection algorithm that detects objects in a single forward pass through the neural network. Unlike older methods that may scan the image multiple times, YOLO processes the full image in one go, which makes it fast enough for real-time applications.

In this project, a pretrained YOLO model is used. The model is already trained on a large dataset and can detect common objects such as people, cars, bottles, laptops, and many others.

## 12. Project Folder Structure

```text
RealTimeObjectDetection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── uploads/
├── outputs/
└── docs/
```

## 13. Future Improvements
- Add support for selecting a custom YOLO model
- Add multi-camera support
- Add count summaries by time intervals
- Add deployment on a local server or cloud platform
- Add support for tracking moving objects across frames
- Add more advanced analytics dashboard

## 14. Notes
This project is intentionally designed to remain simple, understandable, and suitable for a B.E./B.Tech mini-project. It focuses on the core concepts of real-time object detection, counting, and frontend-backend integration without adding unnecessary complexity.
