document.addEventListener('DOMContentLoaded', () => {
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const liveVideo = document.getElementById('liveVideo');
    const fpsValue = document.getElementById('fpsValue');
    const totalObjects = document.getElementById('totalObjects');
    const countList = document.getElementById('countList');
    const activityList = document.getElementById('activityList');
    const imageUploadForm = document.getElementById('imageUploadForm');
    const youtubeVideoForm = document.getElementById('youtubeVideoForm');
    const videoUploadForm = document.getElementById('videoUploadForm');
    const imageInput = document.getElementById('imageInput');
    const videoInput = document.getElementById('videoInput');
    const youtubeUrlInput = document.getElementById('youtubeUrlInput');
    const imageStatus = document.getElementById('imageStatus');
    const videoStatus = document.getElementById('videoStatus');
    const cameraStatus = document.getElementById('cameraStatus');
    const processedImage = document.getElementById('processedImage');
    const processedVideo = document.getElementById('processedVideo');

    let cameraStarted = false;
    let statsTimer = null;

    function updateCountList(counts) {
        if (!counts || Object.keys(counts).length === 0) {
            countList.innerHTML = '<li>No objects detected yet.</li>';
            return;
        }

        const items = Object.entries(counts).map(([label, value]) => {
            return `<li>${label}: ${value}</li>`;
        });

        countList.innerHTML = items.join('');
    }

    function updateActivityList(activities) {
        if (!activities || activities.length === 0) {
            activityList.innerHTML = '<li>Start the webcam to recognize poses.</li>';
            return;
        }

        activityList.replaceChildren(...activities.map((activity, index) => {
            const item = document.createElement('li');
            item.textContent = `Person ${index + 1}: ${activity}`;
            return item;
        }));
    }

    function refreshStats() {
        fetch('/api/stats')
            .then((res) => res.json())
            .then((data) => {
                fpsValue.textContent = data.fps ?? 0;
                totalObjects.textContent = data.total_objects ?? 0;
                updateCountList(data.counts || {});
                updateActivityList(data.activities || []);
            })
            .catch(() => {
                fpsValue.textContent = '0';
                totalObjects.textContent = '0';
                updateCountList({});
                updateActivityList([]);
            });
    }

    function startCamera() {
        if (cameraStarted) {
            return;
        }

        fetch('/api/start_camera')
            .then((res) => res.json())
            .then((data) => {
                if (data.status === 'camera_started') {
                    cameraStatus.textContent = 'Camera started successfully.';
                    cameraStatus.style.color = '#34d399';
                    liveVideo.src = '/video_feed?ts=' + new Date().getTime();
                    cameraStarted = true;
                    if (statsTimer) {
                        clearInterval(statsTimer);
                    }
                    statsTimer = setInterval(refreshStats, 250);
                } else {
                    cameraStatus.textContent = 'Unable to start camera.';
                    cameraStatus.style.color = '#f87171';
                }
            })
            .catch((error) => {
                cameraStatus.textContent = 'Error starting camera: ' + error;
                cameraStatus.style.color = '#f87171';
            });
    }

    function stopCamera() {
        fetch('/api/stop_camera')
            .then((res) => res.json())
            .then((data) => {
                cameraStatus.textContent = data.message || 'Camera stopped.';
                cameraStatus.style.color = '#fbbf24';
                if (statsTimer) {
                    clearInterval(statsTimer);
                    statsTimer = null;
                }
                if (liveVideo) {
                    liveVideo.src = '';
                }
                cameraStarted = false;
                refreshStats();
            })
            .catch(() => {
                cameraStatus.textContent = 'Could not stop the camera.';
                cameraStatus.style.color = '#f87171';
            });
    }

    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);

    imageUploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const file = imageInput.files[0];

        if (!file) {
            imageStatus.textContent = 'Please select an image.';
            imageStatus.style.color = '#f87171';
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        imageStatus.textContent = 'Processing image...';
        imageStatus.style.color = '#fbbf24';

        fetch('/upload/image', {
            method: 'POST',
            body: formData
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    imageStatus.textContent = data.error;
                    imageStatus.style.color = '#f87171';
                    return;
                }

                processedImage.src = data.image_url;
                imageStatus.textContent = data.message || 'Image processed successfully.';
                imageStatus.style.color = '#34d399';
                if (data.stats) {
                    fpsValue.textContent = data.stats.fps ?? 0;
                    totalObjects.textContent = data.stats.total_objects ?? 0;
                    updateCountList(data.stats.counts || {});
                }
            })
            .catch((error) => {
                imageStatus.textContent = 'Image upload failed: ' + error;
                imageStatus.style.color = '#f87171';
            });
    });

    youtubeVideoForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const youtubeUrl = youtubeUrlInput.value.trim();

        if (!youtubeUrl) {
            videoStatus.textContent = 'Please paste a valid YouTube URL.';
            videoStatus.style.color = '#f87171';
            return;
        }

        const formData = new FormData();
        formData.append('youtube_url', youtubeUrl);

        videoStatus.textContent = 'Downloading and processing YouTube video...';
        videoStatus.style.color = '#fbbf24';

        fetch('/process/youtube', {
            method: 'POST',
            body: formData
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    videoStatus.textContent = data.error;
                    videoStatus.style.color = '#f87171';
                    return;
                }

                processedVideo.src = data.video_url;
                videoStatus.textContent = data.message || 'YouTube video processed successfully.';
                videoStatus.style.color = '#34d399';
                if (data.stats) {
                    fpsValue.textContent = data.stats.fps ?? 0;
                    totalObjects.textContent = data.stats.total_objects ?? 0;
                    updateCountList(data.stats.counts || {});
                }
            })
            .catch((error) => {
                videoStatus.textContent = 'YouTube processing failed: ' + error;
                videoStatus.style.color = '#f87171';
            });
    });

    videoUploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const file = videoInput.files[0];

        if (!file) {
            videoStatus.textContent = 'Please select a local video.';
            videoStatus.style.color = '#f87171';
            return;
        }

        const formData = new FormData();
        formData.append('video', file);

        videoStatus.textContent = 'Processing local video...';
        videoStatus.style.color = '#fbbf24';

        fetch('/upload/video', {
            method: 'POST',
            body: formData
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    videoStatus.textContent = data.error;
                    videoStatus.style.color = '#f87171';
                    return;
                }

                processedVideo.src = data.video_url;
                videoStatus.textContent = data.message || 'Video processed successfully.';
                videoStatus.style.color = '#34d399';
                if (data.stats) {
                    fpsValue.textContent = data.stats.fps ?? 0;
                    totalObjects.textContent = data.stats.total_objects ?? 0;
                    updateCountList(data.stats.counts || {});
                }
            })
            .catch((error) => {
                videoStatus.textContent = 'Video upload failed: ' + error;
                videoStatus.style.color = '#f87171';
            });
    });

    refreshStats();
});
