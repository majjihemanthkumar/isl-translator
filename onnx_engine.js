/**
 * Module 3: In-Browser ONNX Runtime Web Execution Engine
 */
class ONNXEngine {
    constructor(modelPath = './isl_model.onnx', labelMapPath = './processed_data/label_map.json') {
        this.modelPath = modelPath;
        this.labelMapPath = labelMapPath;

        this.session = null;
        this.idToWord = {};
        this.isLoaded = false;
        this.inputName = 'float_input';
    }

    /**
     * Load ONNX Model & Label Map JSON
     */
    async init() {
        try {
            console.log('🚀 Loading ONNX Runtime Web Model...');

            // 1. Fetch Label Map JSON
            const mapRes = await fetch(this.labelMapPath);
            if (!mapRes.ok) {
                throw new Error(`Failed to load label map from ${this.labelMapPath}`);
            }
            const labelMap = await mapRes.json();
            
            // Build ID to Word reverse lookup map
            this.idToWord = {};
            for (const [word, id] of Object.entries(labelMap)) {
                this.idToWord[id] = word;
            }
            console.log('✅ Label Map Loaded:', this.idToWord);

            // 2. Initialize ONNX Inference Session with WASM Backend
            ort.env.wasm.numThreads = 1;
            
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['wasm']
            });

            if (this.session.inputNames && this.session.inputNames.length > 0) {
                this.inputName = this.session.inputNames[0];
            }

            this.isLoaded = true;
            console.log(`✅ ONNX Model Loaded! Input Name: '${this.inputName}', Outputs:`, this.session.outputNames);
            return true;

        } catch (err) {
            console.error('❌ ONNX Engine Load Error:', err);
            this.isLoaded = false;
            return false;
        }
    }

    /**
     * Execute inference on 234 Float32 feature vector
     * @param {Float32Array} features234 - 234-element feature vector
     * @returns {Object} { word: string, confidence: number, classId: number }
     */
    async predict(features234) {
        if (!this.isLoaded || !this.session) {
            return { word: '--', confidence: 0, classId: -1 };
        }

        try {
            // Create Float32 Tensor with shape [1, 234]
            const tensorInput = new ort.Tensor('float32', features234, [1, 234]);

            // Execute model session
            const feeds = {};
            feeds[this.inputName] = tensorInput;
            const outputMap = await this.session.run(feeds);

            const outputNames = Object.keys(outputMap);
            let bestClassId = -1;
            let confidence = 0.0;

            // 1. Extract predicted class ID from label output
            const labelKey = outputNames.find(name => name.toLowerCase().includes('label')) || outputNames[0];
            if (outputMap[labelKey] && outputMap[labelKey].data) {
                const labelVal = outputMap[labelKey].data[0];
                bestClassId = typeof labelVal === 'bigint' ? Number(labelVal) : Number(labelVal);
            }

            // 2. Extract probability distribution tensor
            const probKey = outputNames.find(name => name.toLowerCase().includes('prob')) || outputNames[1];
            let probs = [];

            if (probKey && outputMap[probKey] && outputMap[probKey].data) {
                probs = Array.from(outputMap[probKey].data);
            }

            // Determine confidence percentage
            if (probs.length > 0) {
                if (bestClassId >= 0 && bestClassId < probs.length) {
                    confidence = probs[bestClassId];
                } else {
                    let maxP = -1;
                    for (let i = 0; i < probs.length; i++) {
                        if (probs[i] > maxP) {
                            maxP = probs[i];
                            bestClassId = i;
                        }
                    }
                    confidence = maxP;
                }
            } else if (bestClassId !== -1) {
                confidence = 0.95; // High confidence if model output label without probability vector
            }

            if (bestClassId === -1) {
                bestClassId = 0;
            }

            const word = this.idToWord[bestClassId] || `Class ${bestClassId}`;

            return {
                word: word,
                confidence: Math.min(Math.max(confidence, 0.01), 0.999),
                classId: bestClassId,
                rawProbs: probs
            };

        } catch (err) {
            console.error('Inference Execution Error:', err);
            return { word: 'Error', confidence: 0, classId: -1 };
        }
    }
}

// Export for global browser usage
window.ONNXEngine = ONNXEngine;
