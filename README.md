# COUP Neural AI Project

This repository contains my submitted work for the COUP AI project. The goal of my work was to build, train, evaluate, and document a neural-network-based reinforcement learning agent for the social deduction card game COUP.

The original `mycoup` folder also contained starter code, base game files, images, rules, and other materials written or provided by my professor. I intentionally separated my files into this folder so this repository only includes the work I contributed.

## What I Built

I worked on a Neural AI agent for COUP using a policy-gradient reinforcement learning approach based on REINFORCE. The agent learns from full game trajectories and updates its policy based on the final outcome of each game, with reward shaping for coins and turns survived.

The neural agent uses a multi-head policy network so it can handle the different decision types in COUP:

- choosing an action
- deciding whether to challenge
- deciding whether to block
- choosing which character to claim when blocking

The agent represents the game state with a 73-dimensional vector that includes public player information, my agent's private hand, the current pending action context, and the decision type. The action-selection output uses validity masking so the AI cannot choose illegal moves, such as targeting eliminated players or assassinating without enough coins.

## Training and Evaluation

I tested two neural network sizes:

- `32/16` hidden-layer model
- `64/32` hidden-layer model

The report evaluates the trained agents over 10,000 simulated games against a Statistical AI and a Challenger AI.

From the report:

- The `32/16` Neural AI won 5,479 out of 10,000 games, for a 54.8% win rate.
- The `64/32` Neural AI won 6,266 out of 10,000 games, for a 62.7% win rate.
- Both neural agents performed above the expected 33.3% baseline for a three-player game.

The strongest trained agent learned a low-risk strategy: it usually avoided unnecessary challenges, built coins through safer actions such as income and foreign aid, blocked defensively, and used coups as the main finishing action.

## Files Included

- `neural-ai.js` - my neural AI implementation.
- `neural-ai.txt` - notes and explanation for the neural AI.
- `experiment-guide.txt` - experiment and training guide.
- `COUP_Project_Report.pdf` - final project report describing the method, architecture, rewards, evaluation, and results.
- `coup_neural_ai_weights_v3_32_16.json` - trained weights for the 32/16 model.
- `coup_neural_ai_weights_v3_32_16after100000.json` - 32/16 model weights after additional training.
- `coup_neural_ai_weights_v3_64_32 (1).json` - trained weights for the 64/32 model.
- `coup_neural_ai_weights_v3_64_32after100000.json` - 64/32 model weights after additional training.

## Files Not Included

The following files and folders from the original `mycoup` project were not included because they were written or provided by my professor as part of the starter project or class materials:

- base COUP game implementation files, including `index.html` and `aicontest.html`
- starter game engine and shared logic files
- baseline or professor-provided AI files, such as `random-ai.js`, `minimax-ai.js`, `statistical-ai.js`, `challenger-ai.js`, and `ppo-ai.js`
- helper logic files such as belief, policy, probability, and engine code
- image assets for cards, coins, and bot avatars
- rules files and other starter documentation
- backup files, generated logs, local settings, and Git metadata

Those files are required for the full original browser game environment, but they are not included here because this repository is meant to show only my own project contribution.

## Project Summary

My contribution was focused on the reinforcement learning agent: implementing the neural policy, designing the state and action representations, saving trained weights, running experiments, evaluating performance, and documenting the results in the project report.
