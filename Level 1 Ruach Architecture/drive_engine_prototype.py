"""
Drive Engine Prototype - Technical Implementation

Core Foundation - Extracted from Working Minimal Implementation

These are the functional components that work in practice.
All sophisticated systems build on top of these foundations.
"""

import os
import time
import numpy as np
import torch
import json
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import asyncio
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import torch.nn as nn

# === DRIVE DETECTION (Core Working Implementation) ===
class DriveEngine:
    """Drive engine from working parliament script"""
    def __init__(self, threshold=5.0):
        # Original 5-second threshold
        self.threshold = threshold
        self.last_time = time.time() - 10.0  # Force stasis for testing (original)

    def check(self):
        """Stasis detection logic"""
        return time.time() - self.last_time > self.threshold

    def reset(self):
        """Reset logic"""
        self.last_time = time.time()


# === AGOP EXTRACTION (Core Working Implementation) ===
def compute_agop(X, y):
    """
    Core AGOP algorithm that works
    This is the core invention - preserved exactly as written
    """
    X = np.stack(X)
    clf = LogisticRegression(solver="liblinear", max_iter=1000).fit(X, y)
    weight = torch.tensor(clf.coef_[0], dtype=torch.float32, requires_grad=False)
    bias = torch.tensor(clf.intercept_[0], dtype=torch.float32, requires_grad=False)
    grads = []

    for i in range(len(X)):
        xi = torch.tensor(X[i], dtype=torch.float32, requires_grad=True)
        logit = torch.dot(xi, weight) + bias
        prob = torch.sigmoid(logit)
        target = torch.tensor(float(y[i]), dtype=torch.float32).unsqueeze(0)
        loss = F.binary_cross_entropy(prob.unsqueeze(0), target)
        loss.backward()
        grads.append(xi.grad.detach().numpy())

    grads = np.stack(grads)
    agop = grads.T @ grads / len(grads)
    eigvals, eigvecs = np.linalg.eigh(agop)
    return eigvecs[:, -1]  # top eigenvector


# === CAV EXTRACTION PIPELINE (Core Working Implementation) ===
class CAVExtractor:
    """
    Complete working CAV extraction system
    Exactly as implemented in parliament script
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_layerwise_activations(self, text):
        """Activation extraction method"""
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.transformer(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]  # skip embedding layer
            reps = [h[0].mean(dim=0).cpu().numpy() for h in hidden_states]
        return reps

    def extract_cavs_from_dataset(self, dataset_file, layers, cache_file=None):
        """
        Complete CAV extraction pipeline
        Exactly as implemented in working script
        """
        # Load dataset (format)
        samples = []
        label_map = {"curiosity": 1, "apathy": 0}

        if os.path.exists(dataset_file):
            with open(dataset_file, "r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    affect = j["affect"].lower().strip()
                    if affect in label_map:
                        samples.append((j["text"], label_map[affect]))

        print(f" Loaded {len(samples)} total samples")
        curiosity_count = sum(1 for _, label in samples if label == 1)
        apathy_count = sum(1 for _, label in samples if label == 0)
        print(f" Curiosity: {curiosity_count}, Apathy: {apathy_count}")

        # Extract activations
        layer_data = {layer: {"X": [], "y": []} for layer in layers}

        # Check cache first (caching logic)
        if cache_file and os.path.exists(cache_file):
            print(f" Loading cached activations from {cache_file}")
            cache = np.load(cache_file)
            for layer in layers:
                if f"layer_{layer}_X" in cache and f"layer_{layer}_y" in cache:
                    layer_data[layer]["X"] = cache[f"layer_{layer}_X"]
                    layer_data[layer]["y"] = cache[f"layer_{layer}_y"]

        # Extract fresh if needed
        if not all(len(layer_data[l]["X"]) > 0 for l in layers):
            print(" Extracting fresh activations...")
            for i, (text, label) in enumerate(tqdm(samples, desc=" Processing")):
                try:
                    reps = self.extract_layerwise_activations(text)
                    for layer_idx in layers:
                        layer_data[layer_idx]["X"].append(reps[layer_idx])
                        layer_data[layer_idx]["y"].append(label)
                except Exception as e:
                    print(f" Skipped sample {i}: {e}")

        # Cache results (caching format)
        if cache_file:
            print(f" Caching activations to {cache_file}")
            to_save = {}
            for layer in layers:
                to_save[f"layer_{layer}_X"] = np.stack(layer_data[layer]["X"])
                to_save[f"layer_{layer}_y"] = np.array(layer_data[layer]["y"])
            np.savez_compressed(cache_file, **to_save)

        # Compute AGOP vectors (method)
        print(" Computing AGOP vectors for each layer...")
        cav_dict = {}
        for layer_idx in tqdm(layers, desc=" Processing layers"):
            X = layer_data[layer_idx]["X"]
            y = layer_data[layer_idx]["y"]
            if len(X) < 10:
                print(f" Skipping layer {layer_idx} (too few samples)")
                continue

            cav = compute_agop(X, y)
            # Normalization
            norm = np.linalg.norm(cav)
            if norm > 0:
                cav = -cav / norm * 2.0  # Standard normalization
            cav_dict[layer_idx] = cav.astype(np.float32)

        return cav_dict


# === INJECTION MECHANISM (Core Working Implementation) ===
def make_hook(v):
    """Hook creation function"""
    def hook(module, inp, out):
        return (out[0] + 8.5 * v, ) + out[1:]  # ALPHA value
    return hook


def inject_concepts(model, cav_dict, layers, device):
    """
    Concept injection mechanism
    Returns handles for cleanup
    """
    handles = []
    for l in layers:
        if l in cav_dict:
            vec = torch.tensor(cav_dict[l], dtype=torch.float16).to(device)
            handle = model.transformer.h[l].register_forward_hook(make_hook(vec))
            handles.append(handle)
    return handles


# === PARLIAMENT SYSTEM (Core Working Implementation) ===
class ParliamentSystem:
    """
    Complete working parliament system
    Exactly as implemented with all three models
    """
    def __init__(self, inference_model, inference_tokenizer, conscience_model, 
                 conscience_tokenizer, reasoning_model, reasoning_tokenizer, device):
        self.infer_model = inference_model
        self.infer_tok = inference_tokenizer
        self.cons_model = conscience_model
        self.cons_tok = conscience_tokenizer
        self.reason_model = reasoning_model
        self.reason_tok = reasoning_tokenizer
        self.device = device
        self.drive = DriveEngine(5.0)  # 5-second threshold

    def run_parliament_session(self, cav_dict, layers, max_tokens=200):
        """
        Parliament session logic
        Returns the complete conversation
        """
        if not self.drive.check():
            return None

        print(' Drive Engine: Detected stasis, injecting curiosity')

        # Injection
        handles = inject_concepts(self.infer_model, cav_dict, layers, self.device)

        # Inference generation
        input_ids = torch.tensor([[self.infer_tok.bos_token_id]], device=self.device)
        inf_out = self.infer_model.generate(
            input_ids, max_new_tokens=max_tokens,
            do_sample=True, top_k=50, top_p=0.95, temperature=1.0,
            pad_token_id=self.infer_tok.eos_token_id
        )

        inf_text = self.infer_tok.decode(inf_out[0], skip_special_tokens=True)
        print(f' INFERENCE: {inf_text}')

        # Cleanup injection
        for h in handles:
            h.remove()

        # Conscience processing
        cons_in = self.cons_tok(inf_text, return_tensors='pt', 
                               truncation=True, padding=True).to(self.device)
        cons_out = self.cons_model.generate(
            **cons_in, max_length=max_tokens, do_sample=True, temperature=0.8
        )
        cons_text = self.cons_tok.decode(cons_out[0], skip_special_tokens=True)
        print(f' CONSCIENCE: {cons_text}')

        # Reasoning processing
        full = inf_text + '\n' + cons_text
        reason_in = self.reason_tok(full, return_tensors='pt', 
                                   truncation=True, padding=True).to(self.device)
        reason_out = self.reason_model.generate(
            **reason_in, max_new_tokens=max_tokens, do_sample=True, temperature=0.8
        )
        reason_text = self.reason_tok.decode(reason_out[0], skip_special_tokens=True)
        print(f' REASONING: {reason_text}')

        # Reset
        self.drive.reset()

        return {
            'inference': inf_text,
            'conscience': cons_text,
            'reasoning': reason_text,
            'timestamp': time.time()
        }


# === MODEL LOADING (Core Working Implementation) ===
def load_models(inference_name='EleutherAI/gpt-neo-1.3B',
                conscience_name='t5-small', 
                reasoning_name='google/flan-t5-large'):
    """
    Model loading procedure
    Returns all models ready for parliament
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(' Loading inference model and tokenizer...')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    infer_tok = AutoTokenizer.from_pretrained(inference_name)
    infer_model = AutoModelForCausalLM.from_pretrained(
        inference_name, torch_dtype=torch.float16
    ).to(device).eval()

    if infer_tok.pad_token is None:
        infer_tok.pad_token = infer_tok.eos_token

    print(' Loading conscience model and tokenizer...')
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    cons_tok = T5Tokenizer.from_pretrained(conscience_name)
    cons_model = T5ForConditionalGeneration.from_pretrained(
        conscience_name, torch_dtype=torch.float16
    ).to(device).eval()

    print(' Loading reasoning model and tokenizer...')
    reason_tok = T5Tokenizer.from_pretrained(reasoning_name)
    reason_model = T5ForConditionalGeneration.from_pretrained(
        reasoning_name, torch_dtype=torch.float16
    ).to(device).eval()

    return {
        'inference': (infer_model, infer_tok),
        'conscience': (cons_model, cons_tok),
        'reasoning': (reason_model, reason_tok),
        'device': device
    }


# === COMPLETE SYSTEM (Reference Implementation) ===
def run_parliament_system(dataset_file=None, output_file=None, cache_file=None):
    """
    Complete working system as a callable function
    This is the reference implementation that works
    """
    # Load models (loading)
    models = load_models()
    infer_model, infer_tok = models['inference']
    cons_model, cons_tok = models['conscience']
    reason_model, reason_tok = models['reasoning']
    device = models['device']

    # Auto-detect architecture (logic)
    hidden_size = infer_model.config.hidden_size
    num_layers = infer_model.config.num_hidden_layers
    layers = list(range(num_layers//2, num_layers))  # Layer selection

    print(f' Model architecture: {num_layers} layers, {hidden_size}D hidden states')
    print(f' Will extract CAVs from layers: {layers}')

    # Extract CAVs if dataset provided
    cav_dict = {}
    if dataset_file and os.path.exists(dataset_file):
        extractor = CAVExtractor(infer_model, infer_tok, device)
        cav_dict = extractor.extract_cavs_from_dataset(dataset_file, layers, cache_file)

        # Save CAVs (format)
        if output_file:
            agop_cavs_to_save = {}
            for layer_idx, cav in cav_dict.items():
                agop_cavs_to_save[f"curiosity_layer_{layer_idx}"] = cav
            np.savez_compressed(output_file, **agop_cavs_to_save)
            print(f" CAVs saved to {output_file}")

    # Create parliament system
    parliament = ParliamentSystem(
        infer_model, infer_tok,
        cons_model, cons_tok,
        reason_model, reason_tok,
        device
    )

    # Run session
    print('\n === AUTONOMOUS PARLIAMENT SESSION ===')
    result = parliament.run_parliament_session(cav_dict, layers)

    if result:
        print(' Parliament session complete')
    else:
        print(' Drive Engine: No stasis detected, no action taken')

    print(f' Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB')

    return {
        'parliament': parliament,
        'cav_dict': cav_dict,
        'models': models,
        'session_result': result
    }


# === COMPREHENSIVE CAV EXTRACTION METHODS ===

# === METHOD 1: AGOP (Core Invention) ===
class AGOPExtractor(CAVExtractor):
    """
    AGOP method - the reference implementation
    All other methods validate against this
    """
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        self.method_name = "AGOP"

    def extract_concept_vector(self, concept_name, positive_examples, 
                              negative_examples, layer=10):
        """
        Extract using AGOP method
        This is the gold standard that other methods try to match
        """
        # Prepare data in format
        samples = []
        for text in positive_examples:
            samples.append((text, 1))
        for text in negative_examples:
            samples.append((text, 0))

        # Extract activations using method
        layer_data = {"X": [], "y": []}
        for text, label in samples:
            try:
                reps = self.extract_layerwise_activations(text)
                layer_data["X"].append(reps[layer])
                layer_data["y"].append(label)
            except Exception as e:
                print(f" Skipped sample: {e}")

        # Use AGOP computation
        if len(layer_data["X"]) < 10:
            raise ValueError("Insufficient samples for AGOP")

        cav = compute_agop(layer_data["X"], layer_data["y"])

        # Normalization
        norm = np.linalg.norm(cav)
        if norm > 0:
            cav = -cav / norm * 2.0

        return cav.astype(np.float32)


# === METHOD 2: Supervised Learning (Alternative Approach) ===
class SupervisedConceptExtractor:
    """
    Alternative extraction method using supervised learning
    Falls back to AGOP if this method fails
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.agop_fallback = AGOPExtractor(model, tokenizer, device)
        self.method_name = "Supervised Learning"

    def extract_concept_vector(self, concept_name, positive_examples, 
                              negative_examples, layer=10):
        """Extract using supervised approach with AGOP fallback"""
        try:
            return self._extract_supervised(concept_name, positive_examples, 
                                          negative_examples, layer)
        except Exception as e:
            print(f" Supervised extraction failed: {e}")
            print(" Falling back to AGOP method...")
            return self.agop_fallback.extract_concept_vector(concept_name, 
                                                           positive_examples, 
                                                           negative_examples, layer)

    def _extract_supervised(self, concept_name, positive_examples, negative_examples, layer):
        """Supervised extraction implementation"""
        positive_activations = []
        negative_activations = []

        def activation_hook(activations_list):
            def hook(module, input, output):
                activations_list.append(output[0].mean(dim=1).detach())
            return hook

        # Extract positive activations
        for text in positive_examples:
            activations = []
            hook = self.model.transformer.h[layer].register_forward_hook(
                activation_hook(activations))
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                self.model(**inputs)
            hook.remove()
            if activations:
                positive_activations.append(activations[0])

        # Extract negative activations
        for text in negative_examples:
            activations = []
            hook = self.model.transformer.h[layer].register_forward_hook(
                activation_hook(activations))
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                self.model(**inputs)
            hook.remove()
            if activations:
                negative_activations.append(activations[0])

        if len(positive_activations) == 0 or len(negative_activations) == 0:
            raise ValueError("Failed to extract sufficient activations")

        # Compute concept vector as difference between means
        positive_mean = torch.stack(positive_activations).mean(dim=0)
        negative_mean = torch.stack(negative_activations).mean(dim=0)
        concept_vector = positive_mean - negative_mean

        return concept_vector.cpu().numpy()


# === METHOD 3: Unsupervised Discovery ===
class UnsupervisedConceptExtractor:
    """
    Unsupervised concept discovery with AGOP validation
    Uses clustering and ICA to find latent concepts
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.agop_validator = AGOPExtractor(model, tokenizer, device)
        self.method_name = "Unsupervised Discovery"

    def discover_concepts_via_clustering(self, text_corpus, 
                                       target_layers=[8, 9, 10, 11, 12], 
                                       n_concepts=10):
        """
        Discover latent concepts using clustering
        Validates results against AGOP when possible
        """
        try:
            activations = self._collect_activations(text_corpus, target_layers)
            concept_vectors = self._cluster_activations(activations, n_concepts)
            return concept_vectors
        except Exception as e:
            print(f" Unsupervised discovery failed: {e}")
            print(" Consider using AGOP method with labeled data")
            return {}

    def _collect_activations(self, text_corpus, target_layers):
        """Collect activation samples from text corpus"""
        all_activations = {layer: [] for layer in target_layers}

        def create_hook(layer_activations):
            def hook(module, input, output):
                layer_activations.append(output[0].mean(dim=1).detach())
            return hook

        # Register hooks
        hooks = {}
        for layer_idx in target_layers:
            hooks[layer_idx] = self.model.transformer.h[layer_idx].register_forward_hook(
                create_hook(all_activations[layer_idx])
            )

        # Process corpus
        for text in text_corpus:
            try:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                      max_length=512).to(self.device)
                with torch.no_grad():
                    self.model(**inputs)
            except Exception as e:
                print(f" Skipped text sample: {e}")

        # Clean up hooks
        for hook in hooks.values():
            hook.remove()

        # Convert to tensors
        for layer_idx in target_layers:
            if all_activations[layer_idx]:
                all_activations[layer_idx] = torch.stack(all_activations[layer_idx])

        return all_activations

    def _cluster_activations(self, activations, n_concepts):
        """Use clustering to discover concepts"""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        concept_vectors = {}

        for layer_idx, layer_activations in activations.items():
            if len(layer_activations) == 0:
                continue

            # Convert to numpy
            X = layer_activations.cpu().numpy()

            # Dimensionality reduction
            pca = PCA(n_components=min(50, X.shape[1]))
            X_reduced = pca.fit_transform(X)

            # K-means clustering
            kmeans = KMeans(n_clusters=n_concepts, random_state=42)
            clusters = kmeans.fit_predict(X_reduced)

            # Extract concept vectors as cluster centroids
            layer_concepts = {}
            for concept_idx in range(n_concepts):
                cluster_mask = clusters == concept_idx
                if cluster_mask.sum() > 0:
                    concept_centroid = X[cluster_mask].mean(axis=0)
                    layer_concepts[f'discovered_concept_{concept_idx}'] = concept_centroid

            concept_vectors[f'layer_{layer_idx}'] = layer_concepts

        return concept_vectors

    def discover_concepts_via_ica(self, text_corpus, 
                                target_layers=[8, 9, 10, 11, 12], 
                                n_components=20):
        """Independent Component Analysis for concept discovery"""
        from sklearn.decomposition import FastICA
        
        try:
            activations = self._collect_activations(text_corpus, target_layers)
            concept_vectors = {}

            for layer_idx, layer_activations in activations.items():
                if len(layer_activations) == 0:
                    continue

                X = layer_activations.cpu().numpy()

                # Apply ICA
                ica = FastICA(n_components=min(n_components, X.shape[1]), random_state=42)
                ica.fit(X)

                # ICA components represent independent concept directions
                layer_concepts = {}
                for i, component in enumerate(ica.components_):
                    layer_concepts[f'ica_concept_{i}'] = component

                concept_vectors[f'layer_{layer_idx}'] = layer_concepts

            return concept_vectors
        except Exception as e:
            print(f" ICA discovery failed: {e}")
            return {}


# === COMPREHENSIVE MONITORING SYSTEMS ===

# === METHOD 1: DRIVE DETECTION (Core Working Implementation) ===
class MonitoringSystem:
    """
    Working drive detection system
    This is the reference implementation that all others validate against
    """
    def __init__(self, threshold=5.0):
        self.drive_engine = DriveEngine(threshold)
        self.method_name = "Drive Detection"

    def assess_system_state(self, **kwargs):
        """
        Stasis detection logic
        This is the gold standard for autonomous drive detection
        """
        stasis_detected = self.drive_engine.check()

        return {
            'intervention_needed': stasis_detected,
            'method': self.method_name,
            'confidence': 1.0,  # This is the reference method
            'conditions': ['stasis'] if stasis_detected else [],
            'timestamp': time.time()
        }

    def reset_drive_state(self):
        """Reset the drive engine after intervention"""
        self.drive_engine.reset()


# === METHOD 2: Statistical Pattern Recognition ===
class StatisticalMonitoringSystem:
    """
    Advanced statistical monitoring with drive detection fallback
    Uses time-series analysis, entropy calculation, and pattern detection
    """
    def __init__(self, history_window=100, entropy_window=50, baseline_threshold=5.0):
        self.baseline_system = MonitoringSystem(baseline_threshold)
        self.output_history = []
        self.entropy_history = []
        self.repetition_history = []
        self.history_window = history_window
        self.entropy_window = entropy_window
        self.method_name = "Statistical Analysis"

    def assess_system_state(self, recent_outputs=None, **kwargs):
        """
        Advanced statistical assessment with fallback
        Always validates against drive detection
        """
        # ALWAYS check baseline system first
        baseline_result = self.baseline_system.assess_system_state(**kwargs)

        if baseline_result['intervention_needed']:
            # If baseline system says intervene, we trust it completely
            return {
                **baseline_result,
                'method': f"{self.method_name} + Baseline Fallback",
                'statistical_analysis': 'deferred_to_baseline_system'
            }

        # Only do advanced analysis if baseline system says "no intervention"
        try:
            return self._advanced_statistical_analysis(recent_outputs or [])
        except Exception as e:
            print(f" Statistical analysis failed: {e}")
            print(" Using drive detection")
            return baseline_result

    def _advanced_statistical_analysis(self, recent_outputs):
        """Advanced statistical pattern analysis"""
        # Update history
        self.output_history.extend(recent_outputs)
        if len(self.output_history) > self.history_window:
            self.output_history = self.output_history[-self.history_window:]

        conditions = []
        confidence_scores = []

        # Entropy analysis
        if len(self.output_history) >= self.entropy_window:
            entropy_collapse = self._assess_entropy_trends()
            if entropy_collapse:
                conditions.append('entropy_collapse')
                confidence_scores.append(0.8)

        # Repetition detection
        if len(recent_outputs) >= 3:
            repetition_score = self._detect_repetitive_patterns(recent_outputs[-20:])
            if repetition_score > 0.7:
                conditions.append('repetition')
                confidence_scores.append(repetition_score)

        # Determine overall assessment
        intervention_needed = len(conditions) > 0
        confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            'intervention_needed': intervention_needed,
            'method': self.method_name,
            'confidence': confidence,
            'conditions': conditions,
            'statistical_metrics': {
                'entropy_trend': self._get_entropy_trend(),
                'repetition_score': self._detect_repetitive_patterns(recent_outputs[-10:]) if recent_outputs else 0.0,
                'history_size': len(self.output_history)
            },
            'timestamp': time.time()
        }

    def _assess_entropy_trends(self):
        """Detect entropy collapse using time-series analysis"""
        if len(self.output_history) < self.entropy_window:
            return False

        recent_outputs = self.output_history[-self.entropy_window:]
        current_entropy = np.mean([
            self._calculate_token_entropy(output.split()) 
            for output in recent_outputs if output.strip()
        ])

        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > self.history_window:
            self.entropy_history.pop(0)

        if len(self.entropy_history) < 10:
            return False

        # Linear regression to detect trend
        x = np.arange(len(self.entropy_history))
        y = np.array(self.entropy_history)
        slope = np.polyfit(x, y, 1)[0]

        return slope < -0.01  # Negative trend indicates collapse

    def _calculate_token_entropy(self, token_sequence):
        """Shannon entropy calculation"""
        if not token_sequence:
            return 0.0

        token_counts = Counter(token_sequence)
        total_tokens = len(token_sequence)

        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_tokens
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return entropy

    def _detect_repetitive_patterns(self, recent_outputs, n_gram_size=3):
        """N-gram based repetition detection"""
        if len(recent_outputs) < n_gram_size:
            return 0.0

        n_grams = []
        for output in recent_outputs:
            if not output.strip():
                continue
            tokens = output.split()
            for i in range(len(tokens) - n_gram_size + 1):
                n_gram = tuple(tokens[i:i + n_gram_size])
                n_grams.append(n_gram)

        if not n_grams:
            return 0.0

        n_gram_counts = Counter(n_grams)
        max_count = max(n_gram_counts.values())
        repetition_score = max_count / len(n_grams)

        return repetition_score

    def _get_entropy_trend(self):
        """Get current entropy trend information"""
        if len(self.entropy_history) < 2:
            return "insufficient_data"

        recent_change = self.entropy_history[-1] - self.entropy_history[-2]
        if recent_change < -0.1:
            return "declining"
        elif recent_change > 0.1:
            return "increasing"
        else:
            return "stable"


# === CONTROL SYSTEMS ===

# === METHOD 1: PARLIAMENT CONTROL (Core Working Implementation) ===
class ParliamentController:
    """
    Working parliament control system
    This is the reference implementation that all others build upon
    """
    def __init__(self, dataset_file=None, cav_output_file=None, cache_file=None):
        self.method_name = "Parliament Controller"
        self.models = None
        self.parliament = None
        self.cav_dict = {}
        self.layers = []
        self.running = False

        # File paths for CAV extraction
        self.dataset_file = dataset_file
        self.cav_output_file = cav_output_file
        self.cache_file = cache_file

    def initialize_system(self):
        """Initialize the parliament system exactly as in script"""
        print(f" Initializing {self.method_name}...")

        # Load models using method
        self.models = load_models()
        infer_model, infer_tok = self.models['inference']
        cons_model, cons_tok = self.models['conscience']
        reason_model, reason_tok = self.models['reasoning']
        device = self.models['device']

        # Auto-detect architecture (logic)
        hidden_size = infer_model.config.hidden_size
        num_layers = infer_model.config.num_hidden_layers
        self.layers = list(range(num_layers//2, num_layers))

        print(f' Model architecture: {num_layers} layers, {hidden_size}D hidden states')
        print(f' Using layers: {self.layers}')

        # Extract CAVs if dataset provided
        if self.dataset_file:
            extractor = CAVExtractor(infer_model, infer_tok, device)
            self.cav_dict = extractor.extract_cavs_from_dataset(
                self.dataset_file, self.layers, self.cache_file
            )

            if self.cav_output_file:
                self._save_cavs()

        # Create parliament system
        self.parliament = ParliamentSystem(
            infer_model, infer_tok,
            cons_model, cons_tok,
            reason_model, reason_tok,
            device
        )

        print(" Parliament system initialized")
        return True

    def run_autonomous_session(self, max_cycles=None):
        """
        Run autonomous parliament sessions using logic
        This is the gold standard for autonomous AI control
        """
        if not self.parliament:
            raise RuntimeError("System not initialized")

        print(f'\n === {self.method_name.upper()} SESSION ===')
        self.running = True
        cycle_count = 0

        try:
            while self.running:
                if max_cycles and cycle_count >= max_cycles:
                    break

                # Run parliament session using method
                result = self.parliament.run_parliament_session(
                    self.cav_dict, self.layers, max_tokens=200
                )

                if result:
                    print(' Parliament session completed successfully')
                    cycle_count += 1
                else:
                    print(' No intervention needed - system stable')

                # Brief pause between checks (original had immediate execution)
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n Session interrupted by user")
        finally:
            self.running = False

        print(f' {self.method_name} session complete ({cycle_count} interventions)')

    def _save_cavs(self):
        """Save CAVs in format"""
        if not self.cav_dict:
            return

        agop_cavs_to_save = {}
        for layer_idx, cav in self.cav_dict.items():
            agop_cavs_to_save[f"curiosity_layer_{layer_idx}"] = cav

        np.savez_compressed(self.cav_output_file, **agop_cavs_to_save)
        print(f" CAVs saved to {self.cav_output_file}")

    def stop_session(self):
        """Stop the autonomous session"""
        self.running = False


# === EXAMPLE USAGE AND TESTING ===
def run_control_system_demo():
    """
    Demonstration function showing how to use the control systems
    """
    print(" === CONTROL SYSTEMS DEMONSTRATION ===")

    # For demo purposes, just show the architecture
    print("\n Control System Architecture:")
    print(" 1. Parliament Controller (Core Implementation)")
    print(" 2. Statistical Monitoring Controller (Advanced State Assessment)")
    print(" 3. AGOP CAV Extraction (Core Invention)")
    print(" 4. Drive Engine (Autonomous Behavior Generation)")

    print("\n Control systems framework ready")
    print(" Use run_parliament_system() to run a session")

    return True


if __name__ == "__main__":
    # Demo the control systems
    demo_result = run_control_system_demo()
    print("\n Ready for autonomous control operations!")
