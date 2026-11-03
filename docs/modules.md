# Modules

## app.py
This is the main application file. It contains:
- Flask app configuration
- YOLO model loading
- pose model loading and live pose classification
- webcam stream generation
- detection and counting logic
- file upload handling
- routes for frontend communication

## templates/index.html
This file contains the dashboard layout and UI elements.

## static/css/style.css
This file contains styles for the clean, modern interface.

## static/js/script.js
This file handles frontend logic such as:
- starting/stopping the webcam
- polling detection statistics
- sending image upload requests
- sending video upload requests
- displaying results

## uploads/
Stores uploaded files temporarily.

## outputs/
Stores processed output files such as annotated images and videos.

## docs/
Contains documentation for the project.
