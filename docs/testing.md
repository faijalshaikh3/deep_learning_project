# Testing

This project can be tested in a simple step-by-step manner.

## 1. Dependency Installation
Run:

```bash
pip install -r requirements.txt
```

## 2. Run the Application
Run:

```bash
python app.py
```

## 3. Check the Homepage
Visit:

```text
http://127.0.0.1:5000/
```

## 4. Webcam Testing
- Click Start Camera
- Check if the webcam feed appears
- Confirm bounding boxes and labels are visible
- Ensure FPS updates
- Confirm object counts update correctly

## 5. Image Testing
- Upload a valid image
- Confirm the processed image is shown
- Check labels and counts

## 6. Video Testing
- Upload a valid video
- Confirm the processed output is displayed
- Check object counts after processing

## 7. Error Handling Testing
- Upload an unsupported file type
- Upload an empty file
- Check that error messages appear clearly
- Ensure the app does not crash

## 8. Expected Result
The application should run without syntax errors and should detect common objects using the YOLO model.
