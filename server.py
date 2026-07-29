<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ISL Real-Time Translator</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: #0f172a;
      color: #f8fafc;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
    }

    h1 {
      font-size: 1.8rem;
      margin-bottom: 20px;
      letter-spacing: -0.02em;
      color: #38bdf8;
    }

    .container {
      position: relative;
      width: 100%;
      max-width: 640px;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
      border: 1px solid #1e293b;
      background-color: #000;
    }

    video {
      width: 100%;
      height: auto;
      display: block;
      transform: scaleX(-1); /* Mirror camera stream */
    }

    canvas { display: none; }

    .hud-overlay {
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 12px 20px;
      border-radius: 12px;
      min-width: 240px;
    }

    .sign-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #94a3b8;
    }

    .sign-word {
      font-size: 1.8rem;
      font-weight: 800;
      color: #22c55e;
      margin: 2px 0;
    }

    .confidence-meter {
      font-size: 0.85rem;
      color: #cbd5e1;
    }

    .status-badge {
      position: absolute;
      bottom: 16px;
      right: 16px;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
      background: #1e293b;
      color: #94a3b8;
    }

    .status-active {
      background: rgba(34, 197, 94, 0.2);
      color: #4ade80;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }
  </style>
</head>
<body>

  <h1>🤟 Indian Sign Language Translator</h1>

  <div class="container">
    <video id="webcam" autoplay playsinline muted></video>
    <canvas id="canvas"></canvas>

    <div class="hud-overlay">
      <div class="sign-title">Detected Sign</div>
      <div id="signWord" class="sign-word">Waiting...</div>
      <div id="confidence" class="confidence-meter">Confidence: --</div>
    </div>

    <div id="statusBadge" class="status-badge">Connecting...</div>
  </div>

  <script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const signWord = document.getElementById('signWord');
    const confidenceEl = document.getElementById('confidence');
    const statusBadge = document.getElementById('statusBadge');

    // Setup WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws/translate`);

    ws.onopen = () => {
      statusBadge.textContent = "● LIVE SYSTEM ACTIVE";
      statusBadge.classList.add("status-active");
      startStreaming();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.status === "detected") {
        signWord.textContent = data.word;
        signWord.style.color = data.confidence > 50 ? "#22c55e" : "#f97316";
        confidenceEl.textContent = `Confidence: ${data.confidence}%`;
      } else {
        signWord.textContent = "Waiting...";
        signWord.style.color = "#94a3b8";
        confidenceEl.textContent = "Confidence: --";
      }
    };

    ws.onclose = () => {
      statusBadge.textContent = "Disconnected";
      statusBadge.classList.remove("status-active");
    };

    // Request Webcam Stream
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then(stream => {
        video.srcObject = stream;
      })
      .catch(err => {
        console.error("Camera access denied:", err);
        alert("Webcam access is required for real-time translation.");
      });

    // Capture & send frames over WebSocket (~20 FPS)
    function startStreaming() {
      setInterval(() => {
        if (video.readyState === video.HAVE_ENOUGH_DATA && ws.readyState === WebSocket.OPEN) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Export JPEG frame base64 string
          const base64Image = canvas.toDataURL('image/jpeg', 0.6);
          ws.send(base64Image);
        }
      }, 50); // 50ms interval = ~20 FPS
    }
  </script>
</body>
</html>