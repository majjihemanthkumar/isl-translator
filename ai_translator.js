/**
 * Module 6: AI Sentence Refiner & Contextual Natural Language Translator
 * Powered by Groq Ultra-Fast LLM (Qwen / Llama 3)
 */
class AITranslator {
    constructor(options = {}) {
        this.storageKey = 'isl_groq_api_key';
        this.defaultKey = '';
        this.apiKey = localStorage.getItem(this.storageKey) || options.apiKey || '';
        this.model = options.model || 'qwen/qwen3.8-27b';
        this.debounceDelayMs = options.debounceDelayMs || 1500;
        this.autoTranslate = options.autoTranslate !== undefined ? options.autoTranslate : true;

        this.debounceTimer = null;
        this.countdownInterval = null;
        this.lastRawSentence = '';
        this.lastTranslation = '';
        this.isTranslating = false;

        // Callbacks
        this.onStateChange = options.onStateChange || null;
        this.onSuccess = options.onSuccess || null;
        this.onError = options.onError || null;
        this.onClear = options.onClear || null;
    }

    /**
     * Called whenever SentenceEngine commits or modifies recognized words
     */
    handleSentenceUpdate(sentenceString, wordsArray) {
        this.clearTimers();

        if (!wordsArray || wordsArray.length === 0 || !sentenceString.trim()) {
            this.lastRawSentence = '';
            this.lastTranslation = '';
            if (this.onClear) this.onClear();
            return;
        }

        const trimmed = sentenceString.trim();

        // If the sentence changed, schedule auto-translate if enabled
        if (this.autoTranslate) {
            let remainingMs = this.debounceDelayMs;
            if (this.onStateChange) {
                this.onStateChange({
                    status: 'pending',
                    message: `Signing paused... refining in ${(remainingMs / 1000).toFixed(1)}s`,
                    raw: trimmed
                });
            }

            this.countdownInterval = setInterval(() => {
                remainingMs -= 200;
                if (remainingMs > 0 && this.onStateChange) {
                    this.onStateChange({
                        status: 'pending',
                        message: `Signing paused... refining in ${(remainingMs / 1000).toFixed(1)}s`,
                        raw: trimmed
                    });
                }
            }, 200);

            this.debounceTimer = setTimeout(() => {
                this.clearTimers();
                this.translate(trimmed);
            }, this.debounceDelayMs);
        } else {
            if (this.onStateChange) {
                this.onStateChange({
                    status: 'idle',
                    message: 'Click "✨ Refine with AI" to generate sentence',
                    raw: trimmed
                });
            }
        }
    }

    clearTimers() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
            this.countdownInterval = null;
        }
    }

    /**
     * Manual or debounced execution of AI translation
     */
    async translate(rawSentence) {
        const sentence = (rawSentence || '').trim();
        if (!sentence) return;

        this.clearTimers();
        this.isTranslating = true;
        this.lastRawSentence = sentence;

        if (this.onStateChange) {
            this.onStateChange({
                status: 'translating',
                message: 'AI is formulating natural sentence...',
                raw: sentence
            });
        }

        try {
            // 1. Try local server endpoint first (prevents CORS, handles API key securely)
            let result = null;
            try {
                const srvResp = await fetch('/api/ai-translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sentence: sentence,
                        apiKey: this.apiKey,
                        model: this.model
                    })
                });

                if (srvResp.ok) {
                    result = await srvResp.json();
                }
            } catch (err) {
                console.warn('Local server /api/ai-translate unreachable, trying direct Groq fallback...', err);
            }

            // 2. Client-side direct Groq API fallback if server returned error or was unavailable
            if (!result || !result.success) {
                result = await this.callGroqDirect(sentence);
            }

            if (result && result.success && result.translation) {
                this.lastTranslation = result.translation;
                this.isTranslating = false;

                if (this.onSuccess) {
                    this.onSuccess({
                        raw: sentence,
                        translation: result.translation,
                        model: result.model || this.model
                    });
                }
            } else {
                throw new Error((result && result.error) || 'Translation was empty');
            }

        } catch (error) {
            console.error('AI Translation Error:', error);
            this.isTranslating = false;
            if (this.onError) {
                this.onError(error.message || 'AI translation failed');
            }
        }
    }

    /**
     * Direct browser fetch to Groq API
     */
    async callGroqDirect(sentence) {
        const systemPrompt = 
            "You are an expert Indian Sign Language (ISL) interpreter. " +
            "You receive a sequence of recognized ISL sign keywords/glosses. " +
            "Convert them into a single, natural, fluent, and grammatically complete English sentence representing the intended meaning in everyday conversation. " +
            "Example: 'WORRY ALOT WATER' -> \"Don't worry, there is plenty of water.\" " +
            "Example: 'WANT WATER' -> \"I would like some water.\" " +
            "Example: 'WHERE WATER' -> \"Where can I find water?\" " +
            "Example: 'WHO WEAR' -> \"Who is wearing that?\" " +
            "Output ONLY the single final English sentence with no quotation marks, labels, markdown, or explanation.";

        const models = [this.model, 'qwen/qwen3.8-27b', 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'openai/gpt-oss-20b'];
        const seen = new Set();
        const uniqueModels = models.filter(m => !seen.has(m) && seen.add(m));

        let lastErr = null;
        for (const mod of uniqueModels) {
            try {
                const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model: mod,
                        messages: [
                            { role: 'system', content: systemPrompt },
                            { role: 'user', content: sentence }
                        ],
                        temperature: 0.2,
                        max_tokens: 80
                    })
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    lastErr = errData.error ? errData.error.message : `HTTP ${response.status}`;
                    if (response.status === 404) continue; // Try fallback model
                    throw new Error(lastErr);
                }

                const data = await response.json();
                let translated = data.choices && data.choices[0] && data.choices[0].message ? data.choices[0].message.content.trim() : '';
                if ((translated.startsWith('"') && translated.endsWith('"')) || (translated.startsWith("'") && translated.endsWith("'"))) {
                    translated = translated.slice(1, -1).trim();
                }

                return {
                    success: true,
                    raw: sentence,
                    translation: translated,
                    model: mod
                };
            } catch (e) {
                lastErr = e.message;
            }
        }

        return { success: false, error: lastErr || 'Failed to call Groq API' };
    }

    /**
     * Browser Text-to-Speech (TTS)
     */
    speak(text) {
        if (!('speechSynthesis' in window)) {
            alert('Text-to-speech is not supported by your browser.');
            return;
        }

        window.speechSynthesis.cancel(); // Stop any active speech

        const utterance = new SpeechSynthesisUtterance(text || this.lastTranslation);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        utterance.lang = 'en-US';

        // Select an English voice if available
        const voices = window.speechSynthesis.getVoices();
        const englishVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.default));
        if (englishVoice) {
            utterance.voice = englishVoice;
        }

        window.speechSynthesis.speak(utterance);
    }

    setApiKey(newKey) {
        if (newKey && newKey.trim()) {
            this.apiKey = newKey.trim();
            localStorage.setItem(this.storageKey, this.apiKey);
            return true;
        }
        return false;
    }

    getApiKey() {
        return this.apiKey;
    }

    toggleAutoTranslate(enabled) {
        this.autoTranslate = Boolean(enabled);
    }
}

// Export for global browser usage
window.AITranslator = AITranslator;
