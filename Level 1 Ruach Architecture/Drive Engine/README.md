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

