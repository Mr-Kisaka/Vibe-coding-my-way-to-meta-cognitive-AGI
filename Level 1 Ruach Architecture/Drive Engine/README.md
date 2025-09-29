# Drive Engine: Autonomy via Affect Vector Injection in Large Language Models

https://doi.org/10.5281/zenodo.16888180
https://zenodo.org/records/16888180?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIxNWM5Y2E2LWVlMDItNDYxZi04OTZkLWYwNjA5NjJhZGY0MiIsImRhdGEiOnt9LCJyYW5kb20iOiJkOTA2NGY4NGM4NTNmYTRmNWJlODk0NmU2MDkxNGZmZSJ9.-zyNHFCloSviicb-rJazdyGFVfZ16L8w3uDf6rIxesvkbQ3Kxf9HdInq0gyI_R7_tg_XiCH4QJDyykhAWKE6CA

Paper: "It's Alive! AI Independence Without Human Prompting" 
Author: Ronald Kisaka Ogaro

## Overview
This repository contains the minimal implementation of the Drive Engine architecture - a system that enables large language models to generate autonomous, affect-driven outputs without human prompting. The system uses Affect Concept Activation Vectors (ACAVs) extracted from a model's own latent space to inject emotional states and trigger self-initiated behavior.

### Key Features
- **Autonomous triggering**: Detects inactivity and self-initiates generation
- **Affect vector injection**: Steers model behavior using extracted emotional concepts
- **Multi-model parliament**: Inference, conscience, and reasoning models working in concert
- **Validated emotional authenticity**: LIWC-validated outputs with up to 100% authenticity scores

# Donate
Help me take the next step.
As an independent researcher your donations are the only way I get to keep working. Help me bring Ruach to life, if you believe in progress, if you believe in freedom, if you believe, in life.
[![Donate — Ruach Project](https://img.shields.io/badge/Donate-Ruach_Project-0a84ff?style=for-the-badge)](https://www.paypal.com/donate/?hosted_button_id=M375U4YW7WUJE)
[![Donate — Support Ronald](https://img.shields.io/badge/Donate-Support_Ronald-34c759?style=for-the-badge)](https://www.paypal.com/donate/?hosted_button_id=49LTV8ZTZEEGE)

## Quick Start

### Requirements
torch==2.0.1
transformers==4.30.2
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.2.2
nltk==3.8.1
matplotlib==3.7.1
tqdm==4.65.0

### After installing requirements, run this once:
import nltk
nltk.download('punkt')
nltk.download('stopwords')

## Core Scripts
build_agop_cavs.py
generate_with_drive.py
drive_engine_minimal.ipynb

## Experimental Results
Majority of DCAV outputs exceeded human LIWC baseline authenticity
Fear affect achieved 100% authenticity score
Strong statistical separation between affect-steered and random outputs (p < 5×10⁻⁵)
Apathy resistance phenomenon: Unexpected identity assertions contradicting injected affect

## Citation
[1]R. K. Ogaro, ‘Its Alive: AI Independence Without Human Prompting’. Zenodo, Sep. 28, 2025. doi: 10.5281/zenodo.16888180.

## Limitations & Future Work
LIWC analysis limited to demo version
Conscience model implementation incomplete (using T5-small placeholder)
Anger vectors show concerning stereotypical bias patterns
Parliament architecture requires further development

## Support
This research was conducted independently by a farmer and beekeeper in Kenya. 
Support for continued development (including full LIWC licensing) welcome at: https://www.paypal.com/donate/?hosted_button_id=M375U4YW7WUJE

## License
Code: Apache 2.0
Data: CC-BY-SA 4.0
Paper: Academic use encouraged

The Drive Engine establishes a shift in how activation vectors are utilized, not as analytical probes but as functional levers of volitional behavior

### I’ve open-sourced the Drive Engine so the ideas stay free. The next milestone—wiring a live parliament (inference ↔ conscience ↔ reasoning) with a consensus detector and RL goal pursuit—can’t run on Colab due to GPU and session limits. A single local GPU will let me keep the conversation alive, implement consensus/goal-setting. The If this work matters to you, please consider donating. If you want to back the project, use the Ruach button. If you’d rather help me stay afloat while I build, use the personal button, such help will be greatly appreciated.

## Budget & Why the numbers

**Selected build:** Framework Mainboard + RTX 4090 compact node (air-cooled)  
**Target:** ~$3,000 · **Stretch:** up to ~$3,730 (to cover upper-range parts & shipping)

| Item | Notes | Est. |
|---|---|---:|
| RTX 4090 (used or base model) | The heavy lift for local inference/experiments | $1,700–$2,000 |
| 1000 W ATX 3.0 PSU | Stable power for 4090 | $160–$220 |
| Framework Mainboard Intel Core i7 11th/12th gen | Comes with heatsinks/fans | $300–$500 |
| 32 GB RAM (used/mid-range) |  | $300–$450 |
| 2 TB NVMe SSD | Local datasets/checkpoints | $120–$180 |
| 1500 VA UPS | Power stability (prevents session loss) | $160–$230 |
| Incidentals (adapters, shipping, fans) | The small stuff that always shows up | $100–$150 |
| **Subtotal** |  | **$2,840–$3,730** |

**What this unlocks immediately**
- Keep **long-running** conversations alive (no Colab timeouts) to wire the live parliament: inference ↔ conscience ↔ reasoning.
- Implement the **consensus detector** and feed goals to an **RL loop** for actual pursuit.

[![Donate — Ruach Project](https://img.shields.io/badge/Donate-Ruach_Project-0a84ff?style=for-the-badge)](https://www.paypal.com/donate/?hosted_button_id=M375U4YW7WUJE)
[![Donate — Support Ronald](https://img.shields.io/badge/Donate-Support_Ronald-34c759?style=for-the-badge)](https://www.paypal.com/donate/?hosted_button_id=49LTV8ZTZEEGE)


