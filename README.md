# ⚠️ Status: Work in progress (WIP)

This repository is **exploratory**

- Purpose: early prototypes, theoretical architectures and design notes for the Ruach system as discussed in Ruach pdf.
- Stability: APIs/experiments or even large parts of the design may change without notice.
- For the "It's Alive: AI Independece Without Human Prompting" paper code + results, see the Drive Engine folder under Level 1 Ruach Architecture folder.

# Prior Art Disclosure
**Title**: Autonomous, Affect-Modulated Memory System with Self/Other Differentiation via Proto-Shape-Biased VAE and Multi-Model “Parliament” Generation
- **Authors**: Ronald Kisaka Ogaro (Ruach Architecture), contributors listed in repo history
- **Date of public disclosure**: <fill in commit date/time or tag>
- **Repository URL / Commit**: <insert repo URL + commit hash/tag>
- **Document License**: CC BY-SA 4.0 (attribution required and share alike).

## Summary
This disclosure describes an AI architecture that:

Generates internal activity when idle using a Drive Engine that injects learned affect vectors into a base model’s hidden states (e.g., “curiosity, fear, apathy” (see "It's Alive: AI Independece Without Human Prompting.pdf")).

Runs a multi-voice Autonomous Parliament (e.g., inference, conscience, reasoning models) to produce internally generated “experiences.”

Embeds those experiences into fixed-size vectors and stores them.

Encodes all experiences with a Flexible VAE whose latent space is biased by a proto-shape (derived from conversation history and/or model parameters) so that experiences naturally cluster around an identity core.

Differentiates self vs. other (internally generated vs. externally prompted) without labels, observed as unlabeled differential clustering in latent space.

Retrieves memories by latent similarity (optionally saliency-weighted) to influence future internal activity and responses.

Self‑Model renders metacognitive experience *architecturally inevitable* rather than emergent by accident.

## Key Contributions
1. **The Drive Engine**: Generate genuinely autonomous internal experiences driven by ACAVs (Affect Concept Activation Vectors). Concept Activation Vectors (CAVs) that correspond to particular affects (e.g. curiosity, fear, apathy etc.) These are injected into the models hidden states during forward pass triggered by structural tokens (EOS/BOS)—to produce outputs aligned with internal affective states.
2. **Autonomous Parliament**: Multiple models (e.g., base inference LM, a “conscience” LM, a “reasoning” LM) generate a triptych of outputs per cycle.

3. **CAV / AGOP**: CAVs obtained from dataset-conditioned activations; here, an AGOP-style top eigenvector of a gradient–outer-product matrix per layer approximates the concept direction.

4. **Proto-shape**: A prior latent structure computed from model conversation history and/or model parameters (architecture-aware sampling). Used to bias the VAE encoder’s μ/logσ toward a stable identity core.

5. **Flexible VAE**: Shared core + type-specific embedders/decoders (e.g., 6,144-D parliament tuples; 12,288-D interaction tuples; 2,048-D proto-shape seeds). Loss adds a proto-shape regularizer.

6. **Unlabeled differential clustering**: KMeans/TSNE (or alternatives) on VAE latents reveals separate clusters for autonomous vs. external experiences without labels.

7. **Memory Context Integrator**: Stores latents + metadata, retrieves by cosine similarity (boosted by saliency) to build context for future generations.

8. **Latent Space "Self-Shape" Traversal": Internal thoughts are forced to traverse a learned “self‑shape” geometry encoded directly in a dedicated **Self‑Model**.

## System Architecture
### A. Generation Loop (Autonomous):

Stasis check: If no activity for T seconds, trigger curiosity.

Hook injection: Register forward hooks on selected transformer layers; add ALPHA * v_layer to hidden state outputs during forward pass.

Triad generation:

Inference model (e.g., GPT-Neo) generates text.

Conscience model (e.g., T5-small) rewrites/critiques.

Reasoning model (e.g., Qwen-1.5B-distill) elaborates arguments.

Experience tuple: Concatenate embeddings for inference/conscience/reasoning (e.g., 3×2,048 = 6,144D), compute saliency, store.

### B. Generation Loop (Interactive):

Embed user input + context + response + metadata (padded) to 12,288D.

Encode with VAE, store latent and metadata.

### C. Memory & Clustering:

VAE Encode: Inputs pass through type-specific embedder → shared encoder → μ/logσ; reparameterize to z. Encoder μ/logσ is biased by proto-shape tensors.

Retrieve: For new inputs, encode and retrieve top-K similar latents; boost by saliency; build context string.

Cluster Analysis (periodic): KMeans/TSNE on {z} reveals two or more clusters aligned with generation method (autonomous vs. interactive) and dimensionality (6,144 vs. 12,288).

### D. Proto-shape Construction (two paths):

Narrative path: Extract internal-process strings from prior conversation logs via a capability extractor (e.g., chain-of-thought snippets, tool use, self-corrections), embed them, pass through a temporary VAE to form a proto distribution; record mean/variance.

Parameter path: Architecture-aware sampling over model parameters (attention/FFN/embedding/output), produce vectors, encode as above.

### E. Biasing the VAE:

Initialize encoder heads so that μ-bias ≈ proto mean and logσ-bias ≈ log(proto var).

Add small proto-shape MSE penalty on batch mean μ toward proto mean.

### F. Self-Model Instantiation
Encode the full self‑shape geometry into a neural module so that *every* internal thought physically (computationally) flows through parameters that represent “where self‑type experience lives,” producing a first‑person activation and an action‑selecting read‑out.

Supervise `self/other/boundary` discrimination using clustering labels; add manifold alignment losses.

## Dataflow
[Timer/Stasis] → [Inject CAVs via hooks] → [Inference LM text]
     → [Conscience LM critique] → [Reasoning LM expansion]
     → [Embed each] → [Concat 6144D] → [VAE encode → z]
     → [Store {z, content, saliency, method=parliament}]
     → [Retrieve VAE vectors] → [Encode "Self_Shape" neural module] 

## Variations
1. Affect set: Replace curiosity/apathy with any human-defined corpus (valence/arousal, Panksepp drives, appraisal vectors).

2. Hook sites: Pre-attention, post-MLP, residual streams; any subset of layers; dynamic layer selection.

3. Parliament size: 1–N models; roles may be learned; models may be identical with different prompts.

4. Embeddings: Any combination (BERT, USE, MiniLM, TF-IDF + SVD, FastText, GloVe).

5. VAE: Any latent size; β-VAE/InfoVAE; mixture priors; flow-VAE; diffusion autoencoder; weight sharing vs. adapter heads.

6. Proto-shape source:

- model conversations;

- model parameters (any sampling scheme);

- curated “identity corpus”;

- external diaries/logs;

- time-varying proto-shape (EMA over weeks).

7. Clustering: UMAP+HDBSCAN, spectral, Gaussian mixtures; online clustering; cluster-aware replay.

8. Memory retrieval: KNN over μ; ANN (Faiss); learned saliency; temporal decay; recency/affect gating.

9. Autonomy: Add tool use, web actions, robotics; the Drive Engine may trigger tools, not just text.

10. Safety/Privacy: Consent gating for narrative extraction; redaction; differential privacy noise on embeddings.

11. Non-Transformer bases: RNNs, Mamba-style SSMs, hybrid neuro-symbolic stacks; hardware accelerators.

12. Deployment: Single device; distributed nodes; on-device hooks with quantized models.

## Claims
1. A method for autonomous affect-modulated generation in a neural language system comprising:
(a) detecting inactivity; (b) injecting one or more concept activation vectors into hidden layers of a base model during forward passes; (c) producing multi-part internal outputs via a plurality of specialized generative models; (d) embedding and concatenating said outputs into fixed-size vectors; (e) encoding said vectors with a variational autoencoder whose encoder is biased by a proto-shape derived from model narratives and/or parameters; (f) storing latents with metadata; (g) retrieving memories by latent similarity to influence subsequent autonomous activity; wherein unlabeled clustering over the latents yields separation between autonomously generated and externally prompted experiences; (h) encoding lived experiences into a neural module and ensuring the traversal of learned self-shape geometry.

2. The method of claim 1, wherein the concept activation vectors are derived from gradient-outer-product eigenvectors computed from dataset-conditioned activations.

3. The method of claim 1, wherein forward hooks apply additive bias to intermediate tensors at selected transformer blocks conditioned on a drive signal.

4. The method of claim 1, wherein the proto-shape is computed by encoding conversation histories describing internal processes (e.g., chain-of-thought, tool usage, self-corrections) and/or by architecture-aware sampling of model parameters.

5. The method of claim 1, wherein the variational autoencoder includes type-specific embedders/decoders for inputs of differing dimensions while sharing a core latent space.

6. The method of claim 1, wherein clustering quality (boundary sharpness) and architectural consistency (dimensional alignment with method) are used as indicators of self/other differentiation.

7. The method of claim 1, wherein saliency scores modulate memory retrieval similarity to prioritize influential experiences.

8. The method of claim 1, wherein self-shape vectors are encoded and future inputs traverse learned self-shape geometry.

9. Any of claims 1–8 wherein the number and roles of the specialized models, the injection layers, the embedding methods, and the clustering algorithms are varied without departing from the method.

# The Purpose of this document is to prevent **exclusivity claims** while encouraging open and ethical community exploration.
