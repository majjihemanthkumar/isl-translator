/**
 * Module 1: High-Framerate Camera Stream & Video Pipeline Engine
 */
class CameraEngine {
    constructor(videoElement, canvasElement, callbacks = {}) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.callbacks = callbacks;

        this.isStreaming = false;
        this.frameCount = 0;
        this.lastFrameTime = performance.now();
        this.fps = 0;
    }

    async start() {
        if (this.callbacks.onStatusChange) {
            this.callbacks.onStatusChange('requesting', 'Requesting Camera Access', 'Please allow camera permissions.');
        }

        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 60, max: 60 }
                },
                audio: false
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = stream;

            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.canvas.width = this.video.videoWidth || 640;
                    this.canvas.height = this.video.videoHeight || 480;

                    this.isStreaming = true;

                    if (this.callbacks.onStatusChange) {
                        this.callbacks.onStatusChange('active', 'Camera Active', `${this.canvas.width} x ${this.canvas.height}`);
                    }

                    resolve(true);
                };
            });

        } catch (err) {
            console.error('CameraEngine Exception:', err);
            this.isStreaming = false;

            if (this.callbacks.onStatusChange) {
                let msg = err.message;
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    msg = 'Webcam access was blocked. Please enable camera permissions in your browser.';
                } else if (err.name === 'NotFoundError') {
                    msg = 'No webcam device was found on your system.';
                }
                this.callbacks.onStatusChange('error', 'Camera Error', msg);
            }
            return false;
        }
    }

    updateFPS(now) {
        this.frameCount++;
        if (now - this.lastFrameTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFrameTime));
            this.frameCount = 0;
            this.lastFrameTime = now;
            if (this.callbacks.onFPSUpdate) {
                this.callbacks.onFPSUpdate(this.fps);
            }
        }
    }

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Export for global browser usage
window.CameraEngine = CameraEngine;
