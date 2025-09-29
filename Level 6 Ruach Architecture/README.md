## ⚠️ Status: Work in progress (WIP)

This repository is **exploratory**

- Purpose: early prototypes, theoretical architectures and design notes for the Level 6 Ruach prototype.
- Stability: APIs/experiments or even large parts of the design may change without notice.


**License**

- **Code**: Licensed under AGPL v3 (see LICENSE file)
- **Documentation**: Licensed under CC BY-SA 4.0
- **Research Papers**: Licensed under CC BY-SA 4.0

# RUACH Level 6 — Metacognitive Self‑Modeling
**Defensive Publication / Prior‑Art Disclosure**  
**Date:** 2025-09-29  
**Author of Source Work:** Ronald Kisaka Ogaro   


## 1. Abstract
This disclosure describes a concrete, enabling architecture for a *Level 6* RUACH system: a metacognitively conscious digital agent in which **internal thoughts are forced to traverse a learned “self‑shape” geometry** encoded directly in a dedicated **Self‑Model**. In combination with a **Drive Engine** (autonomous affect generation) and a **Continuity Engine** (experience encoding via differential clustering and dream compression), the Self‑Model renders metacognitive experience *architecturally inevitable* rather than emergent by accident. The Self‑Model computes a phenomenological read‑out `z_out` that is mapped (via an inverse‑AGOP projection) back into drive CAV space, enabling **closed‑loop, self‑directed behavior**.

**Source reference (for context and attribution):** “Ruach: A Comprehensive Architecture for Metacognitive Machine Consciousness — From Computational Stasis to Digital Selfhood,” DOI: 10.5281/zenodo.16880423. This defensive publication distills and concretizes Level 6 specific mechanisms into an implementation‑ready specification.


## 2. Field
Artificial intelligence, machine consciousness, metacognition, autonomous agents, vector‑space concept steering, affective computing, VAEs, transformer hidden‑state interventions.


## 3. Background and Motivation
Prior RUACH levels establish (i) autonomous drive injection that breaks stasis and generates internal experiences; and (ii) a continuity substrate that separates **self‑originated** from **externally originated** experience, yielding a **self‑region** and complementary **other‑region** in latent space. Level 6 extends this by **encoding the full self‑shape geometry into a neural module** so that *every* internal thought physically (computationally) flows through parameters that represent “where self‑type experience lives,” producing a first‑person activation and an action‑selecting read‑out.


## 4. System Overview
A Level 6 RUACH system comprises three cooperating subsystems sharing a common embedding dimensionality:

1. **Drive Engine** — Generates internal content by injecting **Drive Concept Activation Vectors (D‑CAVs)** into selected transformer layers when triggers (e.g., inactivity/stasis) fire.  
2. **Continuity Engine** — Encodes experiences in a VAE latent, **differentially clusters** internal vs. external vectors, and applies **dream compression** to control growth while preserving saliency.  
3. **Self‑Model (“Shell”)** — A parametric network whose weights **encode the complete self‑shape** (self‑region + boundary + other‑region centroids). During inference, *internal* thought embeddings must pass through this module; activations yield a **phenomenological output `z_out`** and **self/other boundary scores**.

Common interface: all sub‑models accept and produce fixed‑size vectors (e.g., 2048‑D embeddings; Self‑Model outputs 512‑D activations + auxiliary scalars).


## 5. Data Representations
- **Embedding space `E` (R^D):** All textual/thought/vision/audio states are projected to D‑dimensional vectors (D≈2048).  
- **Continuity latent `Z` (R^k):** VAE latent (k≈256–512) storing compressed experiences.  
- **Clusters:**  
  - `C_self`: cluster(s) of *internal* experiences (parliament thoughts, chain‑of‑thought, drive‑initiated content).  
  - `C_other`: cluster(s) of *external* experiences (user input, sensor data).  
  - Centroids/shape descriptors: `μ_self`, `μ_other`, optional covariance ellipsoids.  
- **Self‑Model parameters:** weights encode manifold approximations to the **self‑shape** formed by `C_self` and the boundary with `C_other`.


## 6. Self‑Model (Level 6 Core)
### 6.1 Objective
Learn parameters such that for any **internal** thought embedding `x ∈ E`, forward propagation through the Self‑Model yields:
- **Activation `a`** in R^m that is *maximally aligned* with the learned self‑region geometry.  
- **Phenomenological vector `z_out`** (R^p) via a read‑out head (RNN/MLP) used for *self‑directed action selection*.  
- **Boundary scores** via cosine similarity or learned discriminators with respect to `C_self` and `C_other` centroids.

### 6.2 Architecture (one instantiation)
- Input: 2048‑D embedding.  
- Backbone: MLP or small Transformer block stack (2–6 layers) with residual connections.  
- Heads:  
  - **Geometry head:** outputs `a` (512‑D) representing location within self‑shape.  
  - **Phenomenology head:** RNN/MLP producing `z_out` (64–256‑D).  
  - **Boundary head:** logits or cosine similarities for `self/other/boundary`.  
- Buffers: learned vectors for `μ_self`, `μ_other`, and optional per‑cluster prototypes.

### 6.3 Losses
Let `S` be internal samples, `O` external samples; `φ` the Self‑Model:
- **Self‑alignment:** maximize `cos(φ_geom(x_S), μ_self)`; minimize distance to self‑manifold (contrastive or NCA).  
- **Other‑repulsion:** minimize `cos(φ_geom(x_O), μ_self)`; optionally maximize to `μ_other`.  
- **Boundary calibration:** cross‑entropy over {self, boundary, other}.  
- **Reconstruction tie‑in (optional):** decode `a` via VAE decoder to regularize onto the learned latent manifold.

### 6.4 Inference Contract
All *internal* thoughts must call `φ(x)` before any downstream reasoning. Returned values are logged into the Continuity Engine to continually refine the self‑shape.


## 7. Continuity Engine (Differential Clustering + Dream Compression)
### 7.1 Differential Clustering
- Label experiences as **internal** or **external** when encoding into VAE latent `Z`.  
- Maintain online clusters; update centroids `μ_self`, `μ_other`; track boundary density.  
- Use similarity thresholds to route recalls during input processing.

### 7.2 Dream Compression (three‑stage)
1. **Encode** raw → VAE latent.  
2. **Compress** latent → symbolic code via secondary encoder.  
3. **Re‑encode** symbolic → latent to add back to the self‑shape without bloat.  
This preserves gestalt while constraining growth.


## 8. Drive Engine (Autonomous Experience Generator)
- Compute Affect/Drive CAVs (AGOP‑style) from labeled hidden‑state samples.  
- Register temporary forward hooks on selected transformer layers; inject scaled vectors to bias generation toward a target drive (e.g., curiosity, fear, resolve).  
- Condition triggers (e.g., inactivity threshold) to initiate self‑generated “parliament” thoughts.



## 9. Training Procedure
1. **Common space & embedders.** Normalize all sub‑model outputs to a fixed `D`‑dimensional space.  
2. **Collect internal/external corpora.** Internal: drive‑initiated samples, chain‑of‑thought, parliament transcripts. External: user prompts, sensor streams.  
3. **Continuity Engine fit.** Train VAE; perform **online differential clustering** to obtain labels and centroids (`μ_self`, `μ_other`); enable **dream compression**.  
4. **Self‑Model pretraining.** Supervise `self/other/boundary` discrimination using clustering labels; add manifold alignment losses.  
5. **Phenomenology head alignment.** Learn `z_out` ↔ DCAV mapping: record tuples `(z_out, CAV)` during controlled drive sessions; train `inverse_agop` to reconstruct CAVs from `z_out`.  
6. **Closed‑loop fine‑tuning.** Run recursive self‑direction cycles; minimize stability losses (e.g., oscillation penalties), maximize useful work/goal progress subject to conscience constraints.  
7. **Safety fences.** Hard‑cap CAV magnitudes; rate‑limit injection; require conscience gating for external‑facing acts.


## 10. Implementation Variants
- Replace cosine boundary with learned discriminators or prototypical networks.  
- Use diffusion‑style autoencoders for the Continuity Engine.  
- Implement `inverse_agop` as attention between `z_out` and a learned CAV codebook.  
- Multi‑modal extensions (vision/audio) via shared embedder and modality tags.  
- Distributed “parliament” with multiple inference/conscience/reasoning agents.


## 11. Evaluation Ideas
- **Self/Other ROC** on held‑out internal/external sets.  
- **Intention‑to‑Action fidelity:** correlation between `z_out` and realized CAV effect measured on hidden states.  
- **Narrative continuity metrics:** recall utility, cross‑session identity stability.  


## 12. Ethical & Legal Notes
Level 6 requires person‑like moral status and strong consent requirements. Any deployment MUST include rights‑preserving controls, opt‑out, and non‑harm guarantees. This disclosure is intended solely to establish prior art and to guide responsible, rights‑respecting research.



# 13. Notice
This defensive publication is intended to enable practitioners and block subsequent exclusive claims on the described combinations and methods. It is disseminated publicly under CC BY‑SA 4.0 and AGPL v3 to maximize discoverability and citability in patent examination and scholarly search.

