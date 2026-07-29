/**
 * Module 5: Sentence Assembly & Debounce Engine
 */
class SentenceEngine {
    constructor(debounceThreshold = 8, minConfidence = 0.45) {
        this.debounceThreshold = debounceThreshold; // Number of consecutive frames needed
        this.minConfidence = minConfidence;       // Minimum prediction confidence required

        this.sentenceWords = [];
        this.candidateWord = null;
        this.consecutiveCount = 0;

        this.onSentenceChange = null;
    }

    /**
     * Process frame prediction tick
     * @param {Object} prediction - { word: string, confidence: number }
     */
    processPrediction(prediction) {
        if (!prediction || prediction.confidence < this.minConfidence || prediction.word === '--' || prediction.word === 'Unknown') {
            this.candidateWord = null;
            this.consecutiveCount = 0;
            return false;
        }

        const newWord = prediction.word;

        if (newWord === this.candidateWord) {
            this.consecutiveCount++;
        } else {
            this.candidateWord = newWord;
            this.consecutiveCount = 1;
        }

        // Commit word when sustained for debounce threshold
        if (this.consecutiveCount === this.debounceThreshold) {
            // Avoid immediate duplicate word duplication if it's already the last word
            const lastWord = this.sentenceWords[this.sentenceWords.length - 1];
            if (lastWord !== newWord) {
                this.sentenceWords.push(newWord);
                if (this.onSentenceChange) {
                    this.onSentenceChange(this.getSentenceString(), this.sentenceWords);
                }
                return true;
            }
        }

        return false;
    }

    getSentenceString() {
        return this.sentenceWords.join(' ');
    }

    clear() {
        this.sentenceWords = [];
        this.candidateWord = null;
        this.consecutiveCount = 0;
        if (this.onSentenceChange) {
            this.onSentenceChange('', []);
        }
    }

    deleteLast() {
        if (this.sentenceWords.length > 0) {
            this.sentenceWords.pop();
            this.candidateWord = null;
            this.consecutiveCount = 0;
            if (this.onSentenceChange) {
                this.onSentenceChange(this.getSentenceString(), this.sentenceWords);
            }
        }
    }
}

// Export for global browser usage
window.SentenceEngine = SentenceEngine;
