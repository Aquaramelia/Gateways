"""
HYBRID PARAMETER PREDICTION SYSTEM
===================================

WHAT IT DOES:
- Takes a word as input
- Outputs all 14 parameters

HOW IT WORKS:
- ML predicts 9 semantic parameters (arousal, complexity, etc.)
- Rules derive 5 physical parameters (width, height, etc.) from the ML predictions

EXPECTED PERFORMANCE:
- Semantic parameters: MAE ~0.033 (excellent)
- Physical parameters: MAE ~0.10-0.15 (good)
- Overall: All 14 parameters with reasonable accuracy

USAGE:
    generator = HybridParameterGenerator()
    params = generator.generate_all_parameters('cathedral')
    
    # Returns dict with all 14 parameters:
    # {
    #   'arousal_norm': 0.524,
    #   'valence_norm': 0.612,
    #   ...
    #   'width': 3.82,
    #   'height': 6.14,
    #   ...
    # }
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
from sentence_transformers import SentenceTransformer
from model_references import SimplifiedModel

class HybridParameterGenerator:
    """
    Generates all 14 parameters using:
    - ML ensemble (5 models) for 9 core semantic parameters
    - Derivation rules for 5 physical parameters
    """
    
    def __init__(self, model_dir='Model/ensemble'):
        """
        Load the trained ensemble models
        
        Args:
            model_dir: Directory containing the trained models
        """
        print("Loading hybrid parameter generator...")
        
        # Load configuration
        config = joblib.load(f'{model_dir}/config.pkl')
        self.num_models = config['num_models']
        self.core_params = config['predictable_params']
        
        # These are the 9 parameters the ML predicts
        assert len(self.core_params) == 9, "Should have 9 core parameters"
        
        # Load the 5 ensemble models
        self.models = []
        self.scalers_X = []
        self.scalers_y = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(self.num_models):
            # Load model
            model = SimplifiedModel(input_size=384, output_size=len(self.core_params))
            model.load_state_dict(torch.load(
                f'{model_dir}/model_{i}.pth',
                map_location=device
            ))
            model.eval()
            model.to(device)
            self.models.append(model)
            
            # Load scalers
            self.scalers_X.append(joblib.load(f'{model_dir}/scaler_X_{i}.pkl'))
            self.scalers_y.append(joblib.load(f'{model_dir}/scaler_y_{i}.pkl'))
        
        # Load sentence transformer for embeddings
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.device = device
        
        print(f"✓ Loaded {self.num_models} models")
        print(f"✓ Core parameters (ML): {', '.join(self.core_params)}")
        print(f"✓ Physical parameters (derived): width, height, curvature, structural_fidelity, detail_frequency")
        print("Ready to generate parameters!\n")
    
    def predict_core_params(self, word):
        """
        Use ML ensemble to predict 9 core semantic parameters
        
        Args:
            word: Input word (string)
            
        Returns:
            dict: {parameter_name: value} for 9 core parameters
        """
        # Get word embedding (384 dimensions)
        embedding = self.embedder.encode([word])[0]
        
        # Get predictions from all 5 models
        predictions = []
        for i in range(self.num_models):
            # Scale input
            X_scaled = self.scalers_X[i].transform(embedding.reshape(1, -1))
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Predict
            self.models[i].eval()
            with torch.no_grad():
                pred_scaled = self.models[i](X_tensor).cpu().numpy()
                pred = self.scalers_y[i].inverse_transform(pred_scaled)[0]
            
            predictions.append(pred)
        
        # Average predictions from all models (ensemble)
        avg_pred = np.mean(predictions, axis=0)
        
        # Convert to dict
        core_params_dict = {
            param: float(avg_pred[i]) 
            for i, param in enumerate(self.core_params)
        }
        
        return core_params_dict
    
    def derive_physical_params(self, word, core_params):
        """
        Derive 5 physical parameters from:
        - Word properties (length, phonetics)
        - ML-predicted core parameters
        
        Args:
            word: Input word (string)
            core_params: dict of 9 core parameters from ML
            
        Returns:
            dict: {parameter_name: value} for 5 physical parameters
        """
        # Extract word features
        word_len = len(word)
        num_vowels = sum(1 for c in word.lower() if c in 'aeiouäöü')
        vowel_ratio = num_vowels / word_len if word_len > 0 else 0
        
        # Extract relevant core parameters
        complexity = core_params['complexity']
        intensity = core_params['intensity']
        symmetry = core_params['symmetry']
        ornament = core_params['ornament_density']
        depth_param = core_params['depth']
        arousal = core_params['arousal_norm']
        
        physical = {}
        
        # ====================================================================
        # WIDTH: Correlates with complexity and ornament density
        # ====================================================================
        # More complex/ornate structures need more horizontal space
        # Base: 2.5 | Range: 1.5-5.0
        base_width = 2.5
        complexity_factor = complexity * 1.2  # 0.0 to 1.2
        ornament_factor = ornament * 0.8     # 0.0 to 0.8
        
        physical['width'] = np.clip(
            base_width + complexity_factor + ornament_factor,
            1.5, 5.0
        )
        
        # ====================================================================
        # HEIGHT: Correlates with intensity and depth
        # ====================================================================
        # High intensity → taller (more dramatic, imposing)
        # High depth → taller (more substantial presence)
        # Base: 3.0 | Range: 2.0-6.5
        base_height = 3.0
        intensity_factor = intensity * 2.0    # 0.0 to 2.0
        depth_factor = depth_param * 1.5      # 0.0 to 1.5
        
        physical['height'] = np.clip(
            base_height + intensity_factor + depth_factor,
            2.0, 6.5
        )
        
        # ====================================================================
        # CURVATURE: Correlates with vowel ratio and symmetry
        # ====================================================================
        # More vowels → more curves (flowing, smooth sound)
        # High symmetry often appears with curves (organic forms)
        # Base: 0.6 | Range: 0.2-0.95
        base_curvature = 0.6
        vowel_factor = vowel_ratio * 0.3      # 0.0 to 0.3
        # Symmetry centered at 0.5: both very low and very high can be curved
        symmetry_factor = (symmetry - 0.5) * 0.2  # -0.1 to +0.1
        
        physical['curvature'] = np.clip(
            base_curvature + vowel_factor + symmetry_factor,
            0.2, 0.95
        )
        
        # ====================================================================
        # STRUCTURAL_FIDELITY: Inverse of complexity, scales with symmetry
        # ====================================================================
        # Simple + symmetric → high fidelity (clean, geometric, precise)
        # Complex + asymmetric → low fidelity (organic, irregular)
        # Base: 0.7 | Range: 0.2-0.95
        base_fidelity = 0.7
        complexity_penalty = complexity * -0.4    # -0.4 to 0.0
        symmetry_bonus = symmetry * 0.3           # 0.0 to 0.3
        
        physical['structural_fidelity'] = np.clip(
            base_fidelity + complexity_penalty + symmetry_bonus,
            0.2, 0.95
        )
        
        # ====================================================================
        # DETAIL_FREQUENCY: Quantized from complexity and ornament
        # ====================================================================
        # This is discrete (1-6), based on overall detail level
        # Combine complexity and ornament density
        detail_score = (complexity + ornament) / 2.0
        
        # Map continuous score to discrete frequency tiers
        if detail_score < 0.3:
            detail_freq = 2      # Minimal detail
        elif detail_score < 0.5:
            detail_freq = 3      # Low-moderate detail
        elif detail_score < 0.7:
            detail_freq = 4      # Moderate-high detail
        elif detail_score < 0.85:
            detail_freq = 5      # High detail
        else:
            detail_freq = 6      # Maximum detail
        
        physical['detail_frequency'] = detail_freq

        # ====================================================================
        # THICKNESS: Correlates with depth and arousal
        # ====================================================================
        # High depth → thicker (more substantial, solid)
        # High arousal → thicker (more physical presence, impact)
        # Base: 0.3 | Range: 0.1-0.6
        base_thickness = 0.3
        depth_factor = depth_param * 0.2      # 0.0 to 0.2
        arousal_factor = arousal * 0.1        # 0.0 to 0.1

        physical['thickness'] = np.clip(
            base_thickness + depth_factor + arousal_factor,
            0.1, 0.6
        )
        
        return physical
    
    def generate_all_parameters(self, word):
        """
        Generate complete 14-parameter set for a word
        
        Args:
            word: Input word (string)
            
        Returns:
            dict: All 14 parameters
                {
                    'arousal_norm': float,
                    'valence_norm': float,
                    'dominance_norm': float,
                    'symmetry': float,
                    'complexity': float,
                    'ornament_density': float,
                    'intensity': float,
                    'depth': float,
                    'thickness': float,
                    'width': float,
                    'height': float,
                    'curvature': float,
                    'structural_fidelity': float,
                    'detail_frequency': int
                }
        """
        # Step 1: ML prediction for 9 core semantic parameters
        core = self.predict_core_params(word)
        
        # Step 2: Derive 5 physical parameters from core + word features
        physical = self.derive_physical_params(word, core)
        
        # Combine and return
        all_params = {**core, **physical}
        
        return all_params
    
    def generate_batch(self, words):
        """
        Generate parameters for multiple words
        
        Args:
            words: List of words
            
        Returns:
            dict: {word: {parameter: value}}
        """
        results = {}
        for word in words:
            results[word] = self.generate_all_parameters(word)
        return results



# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("HYBRID PARAMETER PREDICTION SYSTEM")
    
    # Initialize generator
    generator = HybridParameterGenerator()
    
    # Test on example words
    test_words = [
        'cathedral',   # Should be: tall, complex, symmetric, intense
        'whisper',     # Should be: small, simple, soft, delicate
        'lightning',   # Should be: tall, intense, asymmetric, dramatic
        'apple',       # Should be: small, round, simple
        'mountain',    # Should be: very tall, dramatic, intense
        'butterfly',   # Should be: small, delicate, ornate
    ]
    
    for word in test_words:
        params = generator.generate_all_parameters(word)
        
        print(f"{word.upper()}:")
        print(f" Core Semantic (ML):")
        print(f"    arousal={params['arousal_norm']:.3f} \n"
              f"    complexity={params['complexity']:.3f} \n "
              f"    symmetry={params['symmetry']:.3f} ")
        print(f"    intensity={params['intensity']:.3f} \n"
              f"    ornament={params['ornament_density']:.3f} \n"
              f"    depth={params['depth']:.3f}")
        
        print(f" Physical (Derived):")
        print(f"    width={params['width']:.2f} \n"
              f"    height={params['height']:.2f} \n"
              f"    curvature={params['curvature']:.3f}")
        print(f"    fidelity={params['structural_fidelity']:.3f} \n"
              f"    detail_freq={params['detail_frequency']} \n",
              f"    thickness={params['thickness']:.3f}")
        print()
    
    print("\n BATCH PROCESSING EXAMPLE \n")
    
    # Process multiple words at once
    batch_words = ['storm', 'garden', 'crystal']
    batch_results = generator.generate_batch(batch_words)
    
    for word, params in batch_results.items():
        print(f"{word}: height={params['height']:.2f}, "
              f"complexity={params['complexity']:.3f}, "
              f"intensity={params['intensity']:.3f}")
    