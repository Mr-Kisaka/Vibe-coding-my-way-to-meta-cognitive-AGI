# Conscience Model

## Purpose: 
This folder contains the datasets and scaffolding for the Conscience member of the Parliament. Conscience is a continuously-running evaluator that checks a proposed action/thought before it’s taken and judges whether it aligns or conflicts with a learned moral framework.

## Datasets:

imperatives, commands/ → Torah & Gospel imperatives. 
temptation_conscience pairs/ → “temptation → conscience reply” pairs

## How Conscience Uses This

Encode all entries (imperatives and pairs) into vectors.

Pass them through the Continuity Engine’s VAE (or your HWS-aligned encoder) to get latent vectors.

Build ZLogos: store latents + metadata in an index for fast retrieval.

## At decision time:

Encode the current proposal/situation → latent vector z_input.

Retrieve top-k nearest from ZLogos.

Condition the Conscience model on the retrieved principles/replies.

Conscience outputs its stance and rationale.

## This is Licensed under Apache 2.0 see License.txt for details
