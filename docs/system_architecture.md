# System Architecture

The application uses a simple layered architecture for easier understanding and implementation.

## Frontend Layer
The frontend is built using HTML, CSS, and JavaScript. It provides the UI for:
- starting and stopping the webcam
- uploading images
- uploading videos
- displaying detection statistics

## Backend Layer
The backend is built using Flask. It contains routes for:
- serving the homepage
- starting and stopping the webcam
- streaming video frames
- handling image uploads
- handling video uploads
- returning detection statistics

## Detection Layer
The detection layer uses OpenCV and YOLO:
- OpenCV reads frames from webcam or files
- YOLO processes each frame
- bounding boxes are drawn on the image
- object labels and confidence values are displayed
- counts are calculated and sent to the frontend

## Flow
Browser -> Flask server -> OpenCV + YOLO -> detection results -> browser

This architecture is efficient for a mini-project and easy to explain in a viva.
