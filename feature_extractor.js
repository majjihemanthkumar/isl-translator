/**
 * Module 2: 234-Feature Normalization Engine & Motion Tracing Visualizer
 */
class FeatureExtractor {
    constructor() {
        this.MAX_TRAIL_LENGTH = 10;
        this.leftHandTrail = [];
        this.rightHandTrail = [];

        this.lastKnownLH = null;
        this.lastKnownRH = null;
        this.lhMissCount = 0;
        this.rhMissCount = 0;
    }

    /**
     * Extract 234 Float32 features matching Python implementation with exact math parity
     * @param {Object} results - MediaPipe Holistic results object
     * @returns {Float32Array} 234-element normalized feature vector
     */
    extract(results) {
        const vector = new Float32Array(234);
        let idx = 0;

        // 1. Pose Landmarks (33 x 3 = 99 features)
        const poseArray = [];
        if (results.poseLandmarks && results.poseLandmarks.length === 33) {
            for (let i = 0; i < 33; i++) {
                const lm = results.poseLandmarks[i];
                poseArray.push([lm.x, lm.y, lm.z]);
            }
        } else {
            for (let i = 0; i < 33; i++) {
                poseArray.push([0, 0, 0]);
            }
        }

        const lShoulder = poseArray[11];
        const rShoulder = poseArray[12];
        const shoulderCenter = [
            (lShoulder[0] + rShoulder[0]) / 2.0,
            (lShoulder[1] + rShoulder[1]) / 2.0,
            (lShoulder[2] + rShoulder[2]) / 2.0
        ];

        const dxS = lShoulder[0] - rShoulder[0];
        const dyS = lShoulder[1] - rShoulder[1];
        const dzS = lShoulder[2] - rShoulder[2];
        const shoulderWidth = Math.sqrt(dxS * dxS + dyS * dyS + dzS * dzS) + 1e-6;

        const nose = poseArray[0];

        for (let i = 0; i < 33; i++) {
            vector[idx++] = (poseArray[i][0] - shoulderCenter[0]) / shoulderWidth;
            vector[idx++] = (poseArray[i][1] - shoulderCenter[1]) / shoulderWidth;
            vector[idx++] = (poseArray[i][2] - shoulderCenter[2]) / shoulderWidth;
        }

        // Cache update for landmark persistence
        if (results.leftHandLandmarks) {
            this.lastKnownLH = results.leftHandLandmarks;
            this.lhMissCount = 0;
        } else {
            this.lhMissCount++;
        }

        if (results.rightHandLandmarks) {
            this.lastKnownRH = results.rightHandLandmarks;
            this.rhMissCount = 0;
        } else {
            this.rhMissCount++;
        }

        const activeLH = results.leftHandLandmarks || (this.lhMissCount < 3 ? this.lastKnownLH : null);
        const activeRH = results.rightHandLandmarks || (this.rhMissCount < 3 ? this.lastKnownRH : null);

        // 2. Left Hand (21 x 3 = 63 features)
        const lhArray = [];
        let lhWrist = [0, 0, 0];
        let lhPosRel = [0, 0, 0];

        if (activeLH && activeLH.length === 21) {
            for (let i = 0; i < 21; i++) {
                const lm = activeLH[i];
                lhArray.push([lm.x, lm.y, lm.z]);
            }
            lhWrist = lhArray[0];
            const mcp9 = lhArray[9];
            
            const dxLH = mcp9[0] - lhWrist[0];
            const dyLH = mcp9[1] - lhWrist[1];
            const dzLH = mcp9[2] - lhWrist[2];
            const lhSpan = Math.sqrt(dxLH * dxLH + dyLH * dyLH + dzLH * dzLH) + 1e-6;

            for (let i = 0; i < 21; i++) {
                vector[idx++] = (lhArray[i][0] - lhWrist[0]) / lhSpan;
                vector[idx++] = (lhArray[i][1] - lhWrist[1]) / lhSpan;
                vector[idx++] = (lhArray[i][2] - lhWrist[2]) / lhSpan;
            }

            lhPosRel = [
                (lhWrist[0] - nose[0]) / shoulderWidth,
                (lhWrist[1] - nose[1]) / shoulderWidth,
                (lhWrist[2] - nose[2]) / shoulderWidth
            ];
        } else {
            for (let i = 0; i < 63; i++) {
                vector[idx++] = 0.0;
            }
        }

        // 3. Right Hand (21 x 3 = 63 features)
        const rhArray = [];
        let rhWrist = [0, 0, 0];
        let rhPosRel = [0, 0, 0];

        if (activeRH && activeRH.length === 21) {
            for (let i = 0; i < 21; i++) {
                const lm = activeRH[i];
                rhArray.push([lm.x, lm.y, lm.z]);
            }
            rhWrist = rhArray[0];
            const mcp9 = rhArray[9];

            const dxRH = mcp9[0] - rhWrist[0];
            const dyRH = mcp9[1] - rhWrist[1];
            const dzRH = mcp9[2] - rhWrist[2];
            const rhSpan = Math.sqrt(dxRH * dxRH + dyRH * dyRH + dzRH * dzRH) + 1e-6;

            for (let i = 0; i < 21; i++) {
                vector[idx++] = (rhArray[i][0] - rhWrist[0]) / rhSpan;
                vector[idx++] = (rhArray[i][1] - rhWrist[1]) / rhSpan;
                vector[idx++] = (rhArray[i][2] - rhWrist[2]) / rhSpan;
            }

            rhPosRel = [
                (rhWrist[0] - nose[0]) / shoulderWidth,
                (rhWrist[1] - nose[1]) / shoulderWidth,
                (rhWrist[2] - nose[2]) / shoulderWidth
            ];
        } else {
            for (let i = 0; i < 63; i++) {
                vector[idx++] = 0.0;
            }
        }

        // 4. Hand-to-Nose Distances (6 features)
        vector[idx++] = lhPosRel[0];
        vector[idx++] = lhPosRel[1];
        vector[idx++] = lhPosRel[2];

        vector[idx++] = rhPosRel[0];
        vector[idx++] = rhPosRel[1];
        vector[idx++] = rhPosRel[2];

        // 5. Wrist-to-Wrist Distance (3 features)
        if (activeLH && activeRH) {
            vector[idx++] = (lhWrist[0] - rhWrist[0]) / shoulderWidth;
            vector[idx++] = (lhWrist[1] - rhWrist[1]) / shoulderWidth;
            vector[idx++] = (lhWrist[2] - rhWrist[2]) / shoulderWidth;
        } else {
            vector[idx++] = 0.0;
            vector[idx++] = 0.0;
            vector[idx++] = 0.0;
        }

        return {
            vector: vector,
            activeLH: activeLH,
            activeRH: activeRH
        };
    }

    /**
     * Render motion trajectory trails and skeleton overlays
     */
    drawOverlay(ctx, width, height, results, activeLH, activeRH) {
        // Motion Trail Updates
        if (activeLH) {
            const tip = activeLH[8]; // Index finger tip
            this.leftHandTrail.push({ x: tip.x * width, y: tip.y * height });
            if (this.leftHandTrail.length > this.MAX_TRAIL_LENGTH) this.leftHandTrail.shift();
        } else {
            this.leftHandTrail = [];
        }

        if (activeRH) {
            const tip = activeRH[8]; // Index finger tip
            this.rightHandTrail.push({ x: tip.x * width, y: tip.y * height });
            if (this.rightHandTrail.length > this.MAX_TRAIL_LENGTH) this.rightHandTrail.shift();
        } else {
            this.rightHandTrail = [];
        }

        // Render Motion Trails
        this._drawTrail(ctx, this.leftHandTrail, 'rgba(16, 185, 129, ');
        this._drawTrail(ctx, this.rightHandTrail, 'rgba(245, 158, 11, ');

        // Render Pose Skeletons
        if (results.poseLandmarks) {
            this._drawPose(ctx, width, height, results.poseLandmarks);
        }

        // Render Hand Skeletons
        if (activeLH) {
            this._drawHand(ctx, width, height, activeLH, '#10b981', '#059669');
        }
        if (activeRH) {
            this._drawHand(ctx, width, height, activeRH, '#f59e0b', '#d97706');
        }
    }

    _drawTrail(ctx, points, colorPrefix) {
        if (points.length < 2) return;
        for (let i = 1; i < points.length; i++) {
            const p1 = points[i - 1];
            const p2 = points[i];
            const alpha = (i / points.length) * 0.85;
            const w = (i / points.length) * 6 + 2;

            ctx.strokeStyle = `${colorPrefix}${alpha})`;
            ctx.lineWidth = w;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    }

    _drawPose(ctx, width, height, landmarks) {
        const poseConnections = [[11,12],[11,13],[13,15],[12,14],[14,16]];
        ctx.strokeStyle = 'rgba(37, 99, 235, 0.4)';
        ctx.lineWidth = 3;

        poseConnections.forEach(([i, j]) => {
            const p1 = landmarks[i];
            const p2 = landmarks[j];
            if (p1 && p2 && p1.visibility > 0.4 && p2.visibility > 0.4) {
                ctx.beginPath();
                ctx.moveTo(p1.x * width, p1.y * height);
                ctx.lineTo(p2.x * width, p2.y * height);
                ctx.stroke();
            }
        });
    }

    _drawHand(ctx, width, height, landmarks, strokeColor, jointColor) {
        const connections = [
            [0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]
        ];

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        connections.forEach(([i, j]) => {
            const p1 = landmarks[i];
            const p2 = landmarks[j];
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        });

        landmarks.forEach(lm => {
            ctx.beginPath();
            ctx.arc(lm.x * width, lm.y * height, 4.5, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            ctx.strokeStyle = jointColor;
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }
}

// Export for global browser usage
window.FeatureExtractor = FeatureExtractor;
