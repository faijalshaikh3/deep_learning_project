from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yt_dlp
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
MODEL_FOLDER = BASE_DIR / "models"

for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_CONFIG = {
    "model_name": "yolov8n.pt",
    "confidence": 0.25,
    "iou": 0.45,
    "img_size": 640,
}
POSE_MODEL_CONFIG = {
    "model_name": "yolov8n-pose.pt",
    "confidence": 0.35,
    "img_size": 320,
}

MAX_YOUTUBE_SECONDS = 20
MAX_YOUTUBE_HEIGHT = 720

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}

model = None
pose_model = None
CAMERA_ACTIVE = {"enabled": False}
LATEST_STATS = {"fps": 0, "total_objects": 0, "counts": {}, "activities": []}


def load_model():
    global model
    if model is None:
        try:
            model = YOLO(MODEL_CONFIG["model_name"])
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(f"YOLO model could not be loaded: {exc}") from exc
    return model


def load_pose_model():
    global pose_model
    if pose_model is None:
        try:
            pose_model = YOLO(POSE_MODEL_CONFIG["model_name"])
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(f"YOLO pose model could not be loaded: {exc}") from exc
    return pose_model


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions


def normalize_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def is_valid_youtube_url(url: str) -> bool:
    if not isinstance(url, str):
        return False

    candidate = url.strip()
    if not candidate:
        return False

    try:
        parsed = urlparse(candidate)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False

    host = parsed.netloc.lower().replace("www.", "")
    if host not in YOUTUBE_HOSTS and not host.endswith(".youtube.com"):
        return False

    if host == "youtu.be":
        return bool(parsed.path.strip("/"))

    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        return "v" in query and bool(query["v"][0])

    return parsed.path.startswith(("/shorts/", "/embed/", "/live/"))


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")

    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0]

    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        video_id = query.get("v", [""])[0]
        if video_id:
            return video_id

    match = re.search(r"(?:/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})", parsed.path)
    if match:
        return match.group(1)

    return ""


def download_youtube_video(video_url: str, target_directory: Path) -> Path:
    if not is_valid_youtube_url(video_url):
        raise ValueError("Please provide a valid public YouTube video URL.")

    video_id = extract_youtube_video_id(video_url)
    if not video_id:
        raise ValueError("The YouTube URL is valid but the video ID could not be extracted.")

    temp_directory = target_directory / f"youtube_tmp_{uuid.uuid4().hex}"
    temp_directory.mkdir(parents=True, exist_ok=True)
    output_template = str(temp_directory / "%(title)s.%(ext)s")

    format_candidates = [
        "bestvideo[height<=720]",
        "bestvideo[height<=720][ext=mp4]",
        "bestvideo[ext=mp4]",
        "bestvideo",
    ]

    last_error = None
    try:
        for format_option in format_candidates:
            try:
                with yt_dlp.YoutubeDL(
                    {
                        "format": format_option,
                        "outtmpl": output_template,
                        "noplaylist": True,
                        "quiet": True,
                        "no_warnings": True,
                        "restrictfilenames": True,
                        "nocheckcertificate": True,
                    }
                ) as downloader:
                    downloader.download([video_url])

                downloaded_files = [file for file in temp_directory.iterdir() if file.is_file()]
                if not downloaded_files:
                    raise FileNotFoundError("The YouTube video could not be downloaded.")

                chosen_file = sorted(downloaded_files, key=lambda file: file.stat().st_size, reverse=True)[0]
                final_path = target_directory / f"youtube_{video_id}_{uuid.uuid4().hex}{chosen_file.suffix.lower()}"
                chosen_file.rename(final_path)
                return final_path
            except yt_dlp.utils.DownloadError as exc:
                last_error = exc
                message = str(exc).lower()
                if "requested format is not available" in message:
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise FileNotFoundError("No compatible YouTube stream could be downloaded for this video.")
    finally:
        if temp_directory.exists():
            for child in temp_directory.iterdir():
                if child.is_file():
                    child.unlink()
            temp_directory.rmdir()


def count_detections(results) -> dict:
    class_counts = Counter()
    result = results[0] if len(results) > 0 else None
    if result is None or result.boxes is None:
        return {"counts": {}, "total_objects": 0}

    names = result.names
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = names.get(class_id, f"class_{class_id}")
        class_counts[normalize_class_name(str(class_name))] += 1

    return {"counts": dict(sorted(class_counts.items())), "total_objects": sum(class_counts.values())}


def draw_detection_boundaries(frame: np.ndarray, results) -> tuple[np.ndarray, dict]:
    annotated_frame = frame.copy()
    detection_result = results[0] if len(results) > 0 else None
    if detection_result is None or detection_result.boxes is None:
        return annotated_frame, {"counts": {}, "total_objects": 0}

    names = detection_result.names
    for box in detection_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0]) * 100
        class_id = int(box.cls[0])
        label_name = names.get(class_id, f"class_{class_id}")
        label_text = f"{normalize_class_name(str(label_name))} {confidence:.0f}%"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            label_text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return annotated_frame, count_detections(results)


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    yolo_model = load_model()
    results = yolo_model(
        frame,
        verbose=False,
        conf=MODEL_CONFIG["confidence"],
        iou=MODEL_CONFIG["iou"],
        imgsz=MODEL_CONFIG["img_size"],
    )
    annotated_frame, stats = draw_detection_boundaries(frame, results)
    return annotated_frame, stats


def _joint_angle(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> float:
    first_vector = first - middle
    last_vector = last - middle
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(last_vector)
    if denominator == 0:
        return 180.0
    cosine = np.dot(first_vector, last_vector) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def classify_pose(keypoints: np.ndarray, confidence: np.ndarray) -> str:
    if keypoints.shape != (17, 2) or confidence.shape != (17,):
        return "Pose not clear"

    visible = confidence >= POSE_MODEL_CONFIG["confidence"]

    def has_joints(*indices: int) -> bool:
        return all(visible[index] for index in indices)

    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_wrist, right_wrist = keypoints[9], keypoints[10]
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

    if has_joints(5, 6, 9, 10) and shoulder_width > 0:
        if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
            return "Hands raised"

        image_left_shoulder, image_right_shoulder = sorted(
            ((left_shoulder[0], left_wrist[0]), (right_shoulder[0], right_wrist[0])),
            key=lambda joint: joint[0],
        )
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        wrist_y_close = (
            abs(left_wrist[1] - shoulder_y) < shoulder_width * 0.6
            and abs(right_wrist[1] - shoulder_y) < shoulder_width * 0.6
        )
        arms_out = (
            image_left_shoulder[1] < image_left_shoulder[0] - shoulder_width * 0.5
            and image_right_shoulder[1] > image_right_shoulder[0] + shoulder_width * 0.5
        )
        if wrist_y_close and arms_out:
            return "Arms spread"

    if has_joints(11, 13, 15, 12, 14, 16):
        left_knee_angle = _joint_angle(keypoints[11], keypoints[13], keypoints[15])
        right_knee_angle = _joint_angle(keypoints[12], keypoints[14], keypoints[16])
        if left_knee_angle < 135 and right_knee_angle < 135:
            return "Crouching"

    return "No target pose"


POSE_SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def annotate_poses(frame: np.ndarray, pose_result) -> tuple[np.ndarray, list[str]]:
    keypoints_result = pose_result.keypoints
    if keypoints_result is None or keypoints_result.xy is None:
        return frame, ["No person detected"]

    points_by_person = keypoints_result.xy.cpu().numpy()
    confidences_by_person = (
        keypoints_result.conf.cpu().numpy()
        if keypoints_result.conf is not None
        else np.ones(points_by_person.shape[:2], dtype=np.float32)
    )
    if len(points_by_person) == 0:
        return frame, ["No person detected"]

    activities = []

    for person_index, (points, confidences) in enumerate(zip(points_by_person, confidences_by_person)):
        activity = classify_pose(points, confidences)
        activities.append(activity)

        visible = confidences >= POSE_MODEL_CONFIG["confidence"]
        for first_index, second_index in POSE_SKELETON:
            if visible[first_index] and visible[second_index]:
                first_point = tuple(points[first_index].astype(int))
                second_point = tuple(points[second_index].astype(int))
                cv2.line(frame, first_point, second_point, (255, 190, 0), 2, cv2.LINE_AA)

        for point, is_visible in zip(points, visible):
            if is_visible:
                cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)

        if pose_result.boxes is not None and person_index < len(pose_result.boxes):
            x1, y1, _, _ = map(int, pose_result.boxes.xyxy[person_index].tolist())
        else:
            x1, y1 = 10, 55 + person_index * 28
        cv2.putText(
            frame,
            activity,
            (x1, max(22, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 190, 0),
            2,
            cv2.LINE_AA,
        )

    return frame, activities


def generate_camera_feed():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Webcam is not available or permission was denied.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    CAMERA_ACTIVE["enabled"] = True

    try:
        while CAMERA_ACTIVE["enabled"]:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue

            detect_start = time.perf_counter()
            annotated_frame, stats = process_frame(frame)
            pose_results = load_pose_model()(
                frame,
                verbose=False,
                conf=POSE_MODEL_CONFIG["confidence"],
                imgsz=POSE_MODEL_CONFIG["img_size"],
                max_det=3,
            )
            annotated_frame, activities = annotate_poses(annotated_frame, pose_results[0])
            elapsed_seconds = time.perf_counter() - detect_start
            fps = 1.0 / max(elapsed_seconds, 1e-6)

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            LATEST_STATS["fps"] = round(fps)
            LATEST_STATS["total_objects"] = stats["total_objects"]
            LATEST_STATS["counts"] = stats["counts"]
            LATEST_STATS["activities"] = activities

            ret, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        camera.release()
        CAMERA_ACTIVE["enabled"] = False


def save_uploaded_file(file_storage, directory: Path, allowed_extensions: set[str], field_name: str) -> Path:
    if file_storage is None or file_storage.filename == "":
        raise ValueError(f"Please select a valid {field_name} file.")
    if not allowed_file(file_storage.filename, allowed_extensions):
        raise ValueError(f"Unsupported file type. Allowed {field_name} types: {', '.join(sorted(allowed_extensions))}.")

    filename = secure_filename(file_storage.filename)
    file_path = directory / filename
    file_storage.save(file_path)
    return file_path


def process_image_file(image_path: Path) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded image could not be read. Please upload a valid image file.")
    annotated_image, stats = process_frame(image)
    return annotated_image, stats


def process_video_file(video_path: Path, max_seconds: int | None = None) -> tuple[Path, dict]:
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise ValueError("The uploaded video could not be opened. Please upload a valid video file.")

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    output_name = f"output_{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_FOLDER / output_name
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        video_capture.release()
        raise RuntimeError("The output video writer could not be initialized.")

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    max_frame_limit = None if max_seconds is None else max(1, int(fps * max_seconds))
    processed_frames = 0
    aggregate_counts = Counter()
    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            if max_frame_limit is not None and processed_frames >= max_frame_limit:
                break
            annotated_frame, stats = process_frame(frame)
            writer.write(annotated_frame)
            aggregate_counts.update(stats["counts"])
            processed_frames += 1
            if total_frames and processed_frames >= total_frames:
                break
    finally:
        video_capture.release()
        writer.release()

    final_counts = dict(sorted(aggregate_counts.items()))
    total_objects = sum(aggregate_counts.values())
    stats = {"counts": final_counts, "total_objects": total_objects, "fps": int(fps)}
    LATEST_STATS["fps"] = int(fps)
    LATEST_STATS["total_objects"] = total_objects
    LATEST_STATS["counts"] = final_counts
    return output_path, stats


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start_camera")
def start_camera():
    CAMERA_ACTIVE["enabled"] = True
    return jsonify({"status": "camera_started", "message": "Webcam stream started."})


@app.route("/api/stop_camera")
def stop_camera():
    CAMERA_ACTIVE["enabled"] = False
    LATEST_STATS["fps"] = 0
    LATEST_STATS["total_objects"] = 0
    LATEST_STATS["counts"] = {}
    LATEST_STATS["activities"] = []
    return jsonify({"status": "camera_stopped", "message": "Webcam stream stopped."})


@app.route("/api/stats")
def api_stats():
    return jsonify(LATEST_STATS)


@app.route("/video_feed")
def video_feed():
    try:
        return Response(generate_camera_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/upload/image", methods=["POST"])
def upload_image():
    try:
        uploaded_file = request.files.get("image")
        if uploaded_file is None:
            return jsonify({"error": "Please choose an image file to upload."}), 400

        image_path = save_uploaded_file(uploaded_file, UPLOAD_FOLDER, SUPPORTED_IMAGE_EXTENSIONS, "image")
        annotated_image, stats = process_image_file(image_path)

        output_file_name = f"annotated_{uuid.uuid4().hex}.jpg"
        output_path = OUTPUT_FOLDER / output_file_name
        success = cv2.imwrite(str(output_path), annotated_image)
        if not success:
            return jsonify({"error": "The processed image could not be saved."}), 500

        LATEST_STATS["fps"] = 0
        LATEST_STATS["total_objects"] = stats["total_objects"]
        LATEST_STATS["counts"] = stats["counts"]

        return jsonify(
            {
                "status": "success",
                "image_url": f"/outputs/{output_file_name}",
                "stats": stats,
                "message": "Image processed successfully.",
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@app.route("/process/youtube", methods=["POST"])
def process_youtube_video():
    try:
        youtube_url = request.form.get("youtube_url") or (request.get_json(silent=True) or {}).get("youtube_url")
        if not youtube_url:
            return jsonify({"error": "Please enter a valid YouTube video URL."}), 400

        youtube_url = youtube_url.strip()
        if not is_valid_youtube_url(youtube_url):
            return jsonify({"error": "This does not look like a valid public YouTube URL."}), 400

        downloaded_video = download_youtube_video(youtube_url, UPLOAD_FOLDER)
        output_path, stats = process_video_file(downloaded_video, max_seconds=MAX_YOUTUBE_SECONDS)
        video_url = f"/outputs/{output_path.name}"

        return jsonify(
            {
                "status": "success",
                "video_url": video_url,
                "stats": stats,
                "message": "YouTube video processed successfully.",
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@app.route("/upload/video", methods=["POST"])
def upload_video():
    try:
        uploaded_file = request.files.get("video")
        if uploaded_file is None:
            return jsonify({"error": "Please choose a valid video file to upload."}), 400

        video_path = save_uploaded_file(uploaded_file, UPLOAD_FOLDER, SUPPORTED_VIDEO_EXTENSIONS, "video")
        output_path, stats = process_video_file(video_path)

        video_url = f"/outputs/{output_path.name}"
        return jsonify(
            {
                "status": "success",
                "video_url": video_url,
                "stats": stats,
                "message": "Video processed successfully.",
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(str(OUTPUT_FOLDER), filename)


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({"error": "The requested page was not found."}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "An internal server error occurred. Please try again later."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
