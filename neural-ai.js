/**
 * NeuralAI — Policy Gradient (REINFORCE) AI for Coup
 */

// ── 1. Load TensorFlow.js synchronously (no changes to index.html needed) ──
if (typeof tf === 'undefined') {
    // document.write during initial parse is synchronous: the browser fetches
    // and executes the injected script before continuing to parse the document.
    document.write(
        '<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"><\/script>'
    );
}

// ── 2. Constants ────────────────────────────────────────────────────────────
const NEURAL_CFG = {
    STATE_DIM:      73,
    ACTION_DIM:     16,
    HIDDEN_1:       64,
    HIDDEN_2:       32,
    LEARNING_RATE:  0.001, // change
    GAMMA:          0.99,  // change
    ENTROPY_COEFF:  0.01,  // change
    MAX_PLAYERS:    5,
    MAX_OPPONENTS:  4,
    CHARACTERS:     ['duke', 'assassin', 'captain', 'ambassador', 'contessa'],
    ACTIONS:        ['income', 'foreign-aid', 'tax', 'exchange', 'coup', 'assassinate', 'steal'],
    TRAIN_EVERY:    10,
    MAX_TRAJ_STEPS: 60,

    // ── Reward shaping ──────────────────────────────────────────────────────
    REWARD_WIN:         1.0,
    REWARD_LOSS:       -1.0,
    REWARD_COIN_COEFF:  0.02,   // per coin held at game end
    REWARD_TURN_COEFF:  0.01,   // per decision/turn survived
};



// Which characters can block each action
const NEURAL_BLOCK_CHARS = {
    'foreign-aid': ['duke'],
    'steal':       ['captain', 'ambassador'],
    'assassinate': ['contessa'],
};

// ── 3. NeuralAI class ───────────────────────────────────────────────────────
class NeuralAI extends AIEngine {

    // hidden1 and hidden2 are optional — defaults to NEURAL_CFG values (64/32)
    constructor(playerId, aiType, hidden1 = NEURAL_CFG.HIDDEN_1, hidden2 = NEURAL_CFG.HIDDEN_2) {
        super(playerId, aiType);
        this.hidden1     = hidden1;
        this.hidden2     = hidden2;
        this.weightsKey  = `coup_neural_ai_weights_v3_${hidden1}_${hidden2}`;
        this.model       = this._buildModel();
        this.optimizer   = tf.train.adam(NEURAL_CFG.LEARNING_RATE);
        this.trajectory  = [];
        this._buffer     = [];
        this.training    = true;
        this.gamesPlayed = 0;
        this.wins        = 0;
        this._loadWeights();
    }

    // ── Model ────────────────────────────────────────────────────────────────
    _buildModel() {
        const cfg   = NEURAL_CFG;
        const input = tf.input({ shape: [cfg.STATE_DIM] });

        const h1 = tf.layers.dense({ units: this.hidden1, activation: 'relu', name: 'shared_h1' }).apply(input);
        const h2 = tf.layers.dense({ units: this.hidden2, activation: 'relu', name: 'shared_h2' }).apply(h1);

        const headAction    = tf.layers.dense({ units: cfg.ACTION_DIM, activation: 'linear', name: 'head_action'    }).apply(h2);
        const headChallenge = tf.layers.dense({ units: 2,              activation: 'linear', name: 'head_challenge'  }).apply(h2);
        const headBlock     = tf.layers.dense({ units: 2,              activation: 'linear', name: 'head_block'      }).apply(h2);
        const headClaim     = tf.layers.dense({ units: 5,              activation: 'linear', name: 'head_claim'      }).apply(h2);

        return tf.model({ inputs: input, outputs: [headAction, headChallenge, headBlock, headClaim] });
    }

    // ── State encoding ───────────────────────────────────────────────────────
    //  decisionType : 0=chooseAction  1=challenge  2=block  3=blockClaim
    //  actionStr    : pending action string, or null for chooseAction
    //  actorId      : player index of the claimant / actor
    //  targetId     : player index of the target, or null
    _encodeState(gameState, gameHistory, decisionType, actionStr, actorId, targetId) {
        const cfg     = NEURAL_CFG;
        const vec     = new Float32Array(cfg.STATE_DIM);
        const players = gameState.players;
        let   i       = 0;

        // Per-player features  (5 × 8 = 40)
        for (let p = 0; p < cfg.MAX_PLAYERS; p++) {
            if (p < players.length) {
                const pl  = players[p];
                const revealedCards = gameHistory.revealedCards;
                const rev = revealedCards instanceof Map
                    ? (revealedCards.get(pl.id) || [])
                    : (revealedCards[pl.id] || []);
                vec[i++] = pl.coins / 12.0;
                vec[i++] = pl.cards.filter(c => !c.revealed).length / 2.0;
                vec[i++] = pl.eliminated ? 1 : 0;
                for (const ch of cfg.CHARACTERS) {
                    vec[i++] = rev.filter(c => c === ch).length;
                }
            } else {
                i += 8;   // zero-pad unused player slots
            }
        }

        // Own hand  (2 × 5 = 10)
        const me = players[this.playerId];
        for (let c = 0; c < 2; c++) {
            const card = me ? me.cards[c] : null;
            for (const ch of cfg.CHARACTERS) {
                vec[i++] = (card && !card.revealed && card.character === ch) ? 1 : 0;
            }
        }

        // Is-acting-player  (1)
        vec[i++] = (gameState.currentPlayerIndex === this.playerId) ? 1 : 0;

        // Pending-action context  (7 + 5 + 5 + 1 = 18)
        const aIdx = actionStr ? cfg.ACTIONS.indexOf(actionStr) : -1;
        for (let a = 0; a < 7; a++) vec[i++] = (a === aIdx)    ? 1 : 0;
        for (let p = 0; p < 5; p++) vec[i++] = (p === actorId) ? 1 : 0;
        for (let p = 0; p < 5; p++) vec[i++] = (targetId != null && p === targetId) ? 1 : 0;
        vec[i++] = (targetId != null) ? 1 : 0;

        // Decision-type one-hot  (4)
        for (let d = 0; d < 4; d++) vec[i++] = (d === decisionType) ? 1 : 0;

        return vec;  // i === 73
    }

    // ── Forward pass + categorical sample ────────────────────────────────────
    //  Returns { chosen: number }
    //  validMask: boolean[] — invalid slots get logit −1e9 before softmax
    _sampleFromHead(stateVec, headIdx, validMask) {
        return tf.tidy(() => {
            const cfg   = NEURAL_CFG;
            const input = tf.tensor2d([Array.from(stateVec)], [1, cfg.STATE_DIM]);
            const outs  = this.model.predict(input);
            let logits  = outs[headIdx].squeeze();

            if (validMask) {
                const penalty = tf.tensor1d(validMask.map(v => v ? 0.0 : -1e9));
                logits = logits.add(penalty);
            }

            const probs = tf.softmax(logits).arraySync();
            let r = Math.random();
            let chosen = probs.length - 1;
            for (let k = 0; k < probs.length; k++) {
                r -= probs[k];
                if (r <= 0) { chosen = k; break; }
            }
            return { chosen };
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Non-self, non-eliminated opponents sorted by id (up to MAX_OPPONENTS = 4)
    _opponents(gameState) {
        return gameState.players
            .filter(p => !p.eliminated && p.id !== this.playerId)
            .sort((a, b) => a.id - b.id)
            .slice(0, NEURAL_CFG.MAX_OPPONENTS);
    }

    // Build the 16-slot action list and valid mask for Head 0
    _buildActionList(player, gameState) {
        const cfg      = NEURAL_CFG;
        const actions  = new Array(cfg.ACTION_DIM).fill(null);
        const valid    = new Array(cfg.ACTION_DIM).fill(false);
        const opps     = this._opponents(gameState);
        const mustCoup = player.coins >= 10;
        const canCoup  = player.coins >= 7;
        const canAss   = player.coins >= 3;

        // Non-targeted slots 0-3
        const noTarget = ['income', 'foreign-aid', 'tax', 'exchange'];
        for (let s = 0; s < 4; s++) {
            actions[s] = { action: noTarget[s], targetId: null };
            if (!mustCoup) valid[s] = true;
        }

        // Targeted slots  4-15  (coup | assassinate | steal) × 4 opponent slots
        const targeted = ['coup', 'assassinate', 'steal'];
        for (let ti = 0; ti < 3; ti++) {
            const act = targeted[ti];
            for (let s = 0; s < NEURAL_CFG.MAX_OPPONENTS; s++) {
                const slot = 4 + ti * NEURAL_CFG.MAX_OPPONENTS + s;
                if (s < opps.length) {
                    actions[slot] = { action: act, targetId: opps[s].id };
                    if (mustCoup) {
                        if (act === 'coup' && canCoup) valid[slot] = true;
                    } else {
                        if (act === 'coup'        && canCoup) valid[slot] = true;
                        if (act === 'assassinate' && canAss)  valid[slot] = true;
                        if (act === 'steal')                   valid[slot] = true;
                    }
                }
                // null slots remain invalid (padding)
            }
        }

        // Guarantee at least one valid action (income fallback)
        if (!valid.some(Boolean)) valid[0] = true;

        return { actions, valid };
    }

    // ── Four decision methods (called by index.html game loop) ───────────────

    /**
     * chooseAction(player, gameState, gameHistory)
     * Returns { action: string, targetId: number|null }
     */
    chooseAction(player, gameState, gameHistory) {
        const sv  = this._encodeState(gameState, gameHistory, 0, null, this.playerId, null);
        const { actions, valid } = this._buildActionList(player, gameState);
        const { chosen } = this._sampleFromHead(sv, 0, valid);

        const selected = actions[chosen] || actions.find((a, i) => valid[i] && a);
        if (this.training) {
            this.trajectory.push({
                stateVec: sv,
                headIdx: 0,
                actionIdx: chosen,
                validMask: valid.slice()
            });
        }
        return selected || { action: 'income', targetId: null };
    }

    /**
     * decideChallengeAction(action, claimantId, gameState, gameHistory)
     * Returns boolean — true = challenge
     */
    decideChallengeAction(action, claimantId, gameState, gameHistory) {
        const targetId = gameState.pendingAction ? gameState.pendingAction.targetId : null;
        const sv = this._encodeState(gameState, gameHistory, 1, action, claimantId, targetId);
        const mask = [true, true];
        const { chosen } = this._sampleFromHead(sv, 1, mask);
        if (this.training) {
            this.trajectory.push({
                stateVec: sv,
                headIdx: 1,
                actionIdx: chosen,
                validMask: mask.slice()
            });
        }
        return chosen === 1;   // 0 = allow, 1 = challenge
    }

    /**
     * decideBlockAction(action, actorId, gameState, gameHistory)
     * Returns boolean — true = block
     */
    decideBlockAction(action, actorId, gameState, gameHistory) {
        const targetId = gameState.pendingAction ? gameState.pendingAction.targetId : null;
        const sv = this._encodeState(gameState, gameHistory, 2, action, actorId, targetId);
        const mask = [true, true];
        const { chosen } = this._sampleFromHead(sv, 2, mask);
        if (this.training) {
            this.trajectory.push({
                stateVec: sv,
                headIdx: 2,
                actionIdx: chosen,
                validMask: mask.slice()
            });
        }
        return chosen === 1;   // 0 = allow, 1 = block
    }

    /**
     * decideBlockClaim(action, actorId, gameState, gameHistory)
     * Returns string — character name to claim for the block
     */
    decideBlockClaim(action, actorId, gameState, gameHistory) {
        const cfg     = NEURAL_CFG;
        const allowed = NEURAL_BLOCK_CHARS[action] || [];
        const mask    = cfg.CHARACTERS.map(ch => allowed.includes(ch));
        const sv      = this._encodeState(gameState, gameHistory, 3, action, actorId, null);
        const { chosen } = this._sampleFromHead(sv, 3, mask);
        if (this.training) {
            this.trajectory.push({
                stateVec: sv,
                headIdx: 3,
                actionIdx: chosen,
                validMask: mask.slice()
            });
        }
        const picked = cfg.CHARACTERS[chosen];
        return allowed.includes(picked) ? picked : (allowed[0] || 'duke');
    }

    /**
     * decideChallengeBlock(action, blockerId, blockChar, gameState, gameHistory)
     * Returns boolean — true = challenge the block
     */
    decideChallengeBlock(action, blockerId, blockChar, gameState, gameHistory) {
        const targetId = gameState.pendingAction ? gameState.pendingAction.targetId : null;
        const sv = this._encodeState(gameState, gameHistory, 1, action, blockerId, targetId);
        const mask = [true, true];
        const { chosen } = this._sampleFromHead(sv, 1, mask);
        if (this.training) {
            this.trajectory.push({
                stateVec: sv,
                headIdx: 1,
                actionIdx: chosen,
                validMask: mask.slice()
            });
        }
        return chosen === 1;   // 0 = accept block, 1 = challenge block
    }

    // ── Training ─────────────────────────────────────────────────────────────

    /**
     * Called once per game when a winner is determined.
     * Runs one REINFORCE gradient step then saves weights.
     * @param {boolean} won
     * @param {object}  [finalState] - optional gameState at end of game for reward shaping
     */
    onGameEnd(won, finalState = null) {
        this.gamesPlayed++;
        if (won) this.wins++;

        if (!this.training || this.trajectory.length === 0) {
            this.trajectory = [];
            return;
        }

        // Total decisions this game (before capping) — proxy for turns survived
        const turnsSurvived = this.trajectory.length;

        // Option 3: cap this game's trajectory to the last MAX_TRAJ_STEPS decisions
        const traj = this.trajectory.slice(-NEURAL_CFG.MAX_TRAJ_STEPS);
        this.trajectory = [];

        // ── Shaped terminal reward ────────────────────────────────────────
        //   base:   +REWARD_WIN on win, REWARD_LOSS on loss
        //   coins:  encourage accumulating resources — coin count at game end
        //   turns:  encourage survival — number of decisions made this game
        //           (each decision is a proxy for a turn/interaction survived)
        let finalCoins = 0;
        if (finalState && finalState.players && finalState.players[this.playerId]) {
            finalCoins = finalState.players[this.playerId].coins || 0;
        }

        const baseReward  = won ? NEURAL_CFG.REWARD_WIN : NEURAL_CFG.REWARD_LOSS;
        const coinBonus   = NEURAL_CFG.REWARD_COIN_COEFF * finalCoins;
        const turnBonus   = NEURAL_CFG.REWARD_TURN_COEFF * turnsSurvived;
        const reward      = baseReward + coinBonus + turnBonus;

        const T = traj.length;
        const G = new Array(T);
        G[T - 1] = reward;
        for (let t = T - 2; t >= 0; t--) G[t] = NEURAL_CFG.GAMMA * G[t + 1];
        traj.forEach((step, t) => { step.G = G[t]; });

        // Option 1: accumulate steps; fire one backward pass every TRAIN_EVERY games
        this._buffer.push(...traj);
        if (this.gamesPlayed % NEURAL_CFG.TRAIN_EVERY !== 0) return;
        if (this._buffer.length === 0) return;

        // One backward pass over the entire buffer — avoids 10× redundant passes
        try {
            this._trainBatch(this._buffer);
            this._saveWeights();
        } catch (e) {
            console.error('NeuralAI: training error (skipping step)', e);
        }
        this._buffer = [];
    }

    // Single backward pass over a flat array of {stateVec, headIdx, actionIdx, G} steps.
    _trainBatch(steps) {
        const cfg = NEURAL_CFG;

        if (steps.length > 1) {
            const allG = steps.map(s => s.G);
            const mean = allG.reduce((a, b) => a + b, 0) / allG.length;
            const std  = Math.sqrt(allG.reduce((s, r) => s + (r - mean) ** 2, 0) / allG.length) + 1e-8;
            steps.forEach(s => { s.G = (s.G - mean) / std; });
        }

        // Group steps by decision head
        const byHead = [[], [], [], []];
        steps.forEach(s => byHead[s.headIdx].push(s));

        // Pre-create input tensors OUTSIDE optimizer.minimize so we can dispose them
        // afterwards. (tf.tidy cannot be used inside minimize — it would dispose
        // activations needed for backpropagation.)
        const headData = byHead.map(hSteps => {
            if (hSteps.length === 0) return null;
            const N = hSteps.length;
            return {
                states:  tf.tensor2d(hSteps.flatMap(s => Array.from(s.stateVec)), [N, cfg.STATE_DIM]),
                returns: tf.tensor1d(hSteps.map(s => s.G)),
                indices: hSteps.map(s => s.actionIdx),
                masks:   hSteps.map(s => s.validMask),
            };
        });

        // One backward pass across all four heads
        const lossVal = this.optimizer.minimize(() => {
            let totalLoss = tf.scalar(0);

            for (let h = 0; h < 4; h++) {
                const d = headData[h];
                if (!d) continue;

                const outputs  = this.model.predict(d.states);
                let logits     = outputs[h];                            // [N, K]
                const K        = logits.shape[1];
                const masks    = d.masks.map(mask =>
                    mask && mask.length === K ? mask : new Array(K).fill(true)
                );
                const penalty  = tf.tensor2d(
                    masks.flatMap(mask => mask.map(v => v ? 0.0 : -1e9)),
                    [masks.length, K]
                );
                logits = logits.add(penalty);

                const probs    = tf.softmax(logits);
                const logProbs = tf.log(probs.add(1e-8));

                const oneHot   = tf.cast(tf.oneHot(d.indices, K), 'float32');
                const chosenLP = tf.sum(tf.mul(logProbs, oneHot), 1);  // [N]

                const pgLoss  = tf.neg(tf.mean(tf.mul(chosenLP, d.returns)));
                const entropy = tf.neg(tf.mean(tf.sum(tf.mul(probs, logProbs), 1)));

                totalLoss = totalLoss
                    .add(pgLoss)
                    .sub(entropy.mul(tf.scalar(cfg.ENTROPY_COEFF)));
            }

            return totalLoss;
        }, /* returnCost= */ true);

        // Dispose input tensors now that the backward pass is complete
        headData.forEach(d => { if (d) { d.states.dispose(); d.returns.dispose(); } });
        if (lossVal) lossVal.dispose();
    }

    // ── Weight persistence (synchronous via dataSync) ────────────────────────

    _saveWeights() {
        try {
            const data = this._weightsData();
            localStorage.setItem(this.weightsKey, JSON.stringify(data));
        } catch (e) {
            console.warn('NeuralAI: failed to save weights', e);
        }
    }

    _loadWeights() {
        try {
            const raw = localStorage.getItem(this.weightsKey);
            if (!raw) return;
            const saved   = JSON.parse(raw);
            const tensors = saved.map(w => tf.tensor(w.data, w.shape));
            this.model.setWeights(tensors);
            tensors.forEach(t => t.dispose());
            console.log(`NeuralAI: loaded weights (games: ${this.gamesPlayed})`);
        } catch (e) {
            console.log('NeuralAI: starting with fresh weights');
        }
    }

    _weightsData() {
        return this.model.getWeights().map(w => ({
            shape: w.shape,
            data:  Array.from(w.dataSync()),
        }));
    }

    exportWeightsFile() {
        const payload = {
            weightsKey: this.weightsKey,
            hidden1: this.hidden1,
            hidden2: this.hidden2,
            savedAt: new Date().toISOString(),
            weights: this._weightsData(),
        };
        NeuralAI.downloadWeightsPayload(payload);
    }

    importWeightsPayload(payload) {
        const weights = NeuralAI.extractWeights(payload);
        const tensors = weights.map(w => tf.tensor(w.data, w.shape));
        this.model.setWeights(tensors);
        tensors.forEach(t => t.dispose());
        localStorage.setItem(this.weightsKey, JSON.stringify(weights));
    }

    static extractWeights(payload) {
        const weights = Array.isArray(payload) ? payload : payload?.weights;
        if (!Array.isArray(weights) || weights.length === 0) {
            throw new Error('Weights file does not contain a weights array.');
        }
        for (const w of weights) {
            if (!Array.isArray(w.shape) || !Array.isArray(w.data)) {
                throw new Error('Weights file has an invalid tensor entry.');
            }
        }
        return weights;
    }

    static downloadWeightsPayload(payload) {
        const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${payload.weightsKey || 'coup_neural_ai_weights'}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /** Wipe stored weights and reset stats. */
    resetWeights() {
        localStorage.removeItem(this.weightsKey);
        this.gamesPlayed = 0;
        this.wins        = 0;
        console.log('NeuralAI: weights reset');
    }

    getStats() {
        const wr = this.gamesPlayed > 0
            ? (this.wins / this.gamesPlayed * 100).toFixed(1) + '%'
            : 'n/a';
        return { gamesPlayed: this.gamesPlayed, wins: this.wins, winRate: wr };
    }
}

// ── 4. Monkey-patches (run after all inline <script> tags have executed) ────
document.addEventListener('DOMContentLoaded', () => {

    // ── 4a. Inject Neural Network AI options into index.html's AI dropdowns ──
    // Only runs on index.html (not aicontest.html, which hardcodes its options).
    // The player-count <select> fires 'change' which rebuilds the dropdowns,
    // so we listen on it and inject our options a tick later each time.
    const isIndexPage   = !!document.getElementById('setupScreen');
    const playerCountEl = document.getElementById('playerCount');

    function injectNeuralOptions() {
        if (!isIndexPage) return;
        document.querySelectorAll('#aiSelectors select').forEach(sel => {
            if (!sel.querySelector('option[value="neural_64_32"]')) {
                const opt64      = document.createElement('option');
                opt64.value      = 'neural_64_32';
                opt64.textContent = 'Neural Network AI (64/32)';
                sel.appendChild(opt64);
            }
            if (!sel.querySelector('option[value="neural_32_16"]')) {
                const opt32      = document.createElement('option');
                opt32.value      = 'neural_32_16';
                opt32.textContent = 'Neural Network AI (32/16)';
                sel.appendChild(opt32);
            }
        });
    }

    if (playerCountEl) {
        playerCountEl.addEventListener('change', () => setTimeout(injectNeuralOptions, 20));
    }
    setTimeout(injectNeuralOptions, 20);  // also run on initial load

    // ── 4b. Patch startGame to handle the 'neural' AI type ──────────────────
    // The original switch falls through to RandomAI for unknown types.
    // After the original startGame finishes, we replace those players' engines.
    const _origStartGame = window.startGame;
    if (_origStartGame) {
        window.startGame = function () {
            _origStartGame();
            // Fix any player whose dropdown selected 'neural'
            if (typeof gameState !== 'undefined') {
                gameState.players.forEach(p => {
                    if (!p.isHuman) {
                        const sel = document.getElementById(`ai-type-${p.id}`);
                        if (sel && sel.value === 'neural_64_32') {
                            p.aiEngine = new NeuralAI(p.id, 'neural_64_32', 64, 32);
                            p.aiType   = 'neural_64_32';
                            p.name     = `Neural(64/32) ${p.id}`;
                        } else if (sel && sel.value === 'neural_32_16') {
                            p.aiEngine = new NeuralAI(p.id, 'neural_32_16', 32, 16);
                            p.aiType   = 'neural_32_16';
                            p.name     = `Neural(32/16) ${p.id}`;
                        }
                    }
                });
            }
        };
    }

    // ── 4c. Patch checkGameOver to call onGameEnd on every NeuralAI ─────────
    const _origCheckGameOver = window.checkGameOver;
    if (_origCheckGameOver) {
        window.checkGameOver = function () {
            const result = _origCheckGameOver();
            if (result && typeof gameState !== 'undefined') {
                const active = gameState.players.filter(p => !p.eliminated);
                if (active.length === 1) {
                    const winnerId = active[0].id;
                    gameState.players.forEach(p => {
                        if (p.aiEngine instanceof NeuralAI) {
                            // Pass gameState so onGameEnd can read final coin counts
                            p.aiEngine.onGameEnd(p.id === winnerId, gameState);
                        }
                    });
                }
            }
            return result;
        };
    }
});
