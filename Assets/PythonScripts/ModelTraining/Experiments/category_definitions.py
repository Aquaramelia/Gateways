# Reference categories for classification

CATEGORY_DEFINITIONS = {
    'weather_dramatic': {
        'params': {
            'width': (3.0, 4.5), 'height': (4.0, 5.5),
            'curvature': (0.4, 0.6), 'symmetry': (0.3, 0.5),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.3, 0.5),
            'ornament_density': (0.4, 0.6), 'intensity': (0.85, 0.95),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.8, 0.95), 'valence_norm': (0.3, 0.5),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['lightning', 'thunder']
    },
    
    'weather_gentle': {
        'params': {
            'width': (2.5, 3.5), 'height': (2.5, 3.5),
            'curvature': (0.85, 0.95), 'symmetry': (0.6, 0.8),
            'complexity': (0.2, 0.4), 'structural_fidelity': (0.2, 0.3),
            'ornament_density': (0.2, 0.4), 'intensity': (0.2, 0.4),
            'detail_frequency': (2, 3), 'depth': (0.3, 0.5),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.2, 0.4), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['mist', 'fog', 'cloud', 'smoke']
    },
    
    'weather_moderate': {
        'params': {
            'width': (2.5, 3.5), 'height': (3.0, 4.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.6, 0.8),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.4, 0.6),
            'ornament_density': (0.4, 0.6), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.4, 0.6),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['rain', 'snow']
    },
    
    'celestial_bright': {
        'params': {
            'width': (2.5, 3.5), 'height': (2.5, 3.5),
            'curvature': (0.85, 0.95), 'symmetry': (0.8, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.6, 0.8),
            'ornament_density': (0.6, 0.8), 'intensity': (0.7, 0.9),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['sunrise', 'sunset', 'star']
    },
    
    'celestial_calm': {
        'params': {
            'width': (2.0, 3.0), 'height': (2.0, 3.0),
            'curvature': (0.85, 0.95), 'symmetry': (0.85, 0.95),
            'complexity': (0.3, 0.5), 'structural_fidelity': (0.7, 0.9),
            'ornament_density': (0.4, 0.6), 'intensity': (0.3, 0.5),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.2, 0.4), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.4, 0.6)
        },
        'words': ['moon', 'evening', 'midnight']
    },
    
    'flower_delicate': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.0, 2.5),
            'curvature': (0.85, 0.95), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.6, 0.8), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.3, 0.5), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['daisy', 'violet', 'lily', 'orchid']
    },
    
    'flower_bold': {
        'params': {
            'width': (2.5, 3.0), 'height': (2.5, 3.0),
            'curvature': (0.8, 0.95), 'symmetry': (0.85, 0.95),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.7, 0.85), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['sunflower', 'rose', 'tulip']
    },
    
    'tree_tall': {
        'params': {
            'width': (3.0, 4.0), 'height': (5.0, 6.0),
            'curvature': (0.6, 0.8), 'symmetry': (0.7, 0.85),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.7, 0.85),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['pine', 'oak', 'palm', 'birch']
    },
    
    'tree_graceful': {
        'params': {
            'width': (3.5, 4.5), 'height': (4.5, 5.5),
            'curvature': (0.75, 0.9), 'symmetry': (0.65, 0.8),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.65, 0.8),
            'ornament_density': (0.6, 0.8), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.6, 0.8),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.3, 0.5), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['willow', 'cherry', 'maple', 'bamboo']
    },
    
    'landscape_gentle': {
        'params': {
            'width': (3.5, 4.5), 'height': (2.5, 3.5),
            'curvature': (0.75, 0.9), 'symmetry': (0.8, 0.9),
            'complexity': (0.3, 0.5), 'structural_fidelity': (0.5, 0.7),
            'ornament_density': (0.4, 0.6), 'intensity': (0.4, 0.6),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.2, 0.4), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['garden', 'meadow', 'valley']
    },
    
    'landscape_dramatic': {
        'params': {
            'width': (4.0, 5.0), 'height': (5.0, 6.5),
            'curvature': (0.5, 0.7), 'symmetry': (0.7, 0.85),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.8, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.7, 0.9),
            'detail_frequency': (4, 5), 'depth': (0.7, 0.9),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.8, 0.95)
        },
        'words': ['mountain', 'cliff', 'desert']
    },
    
    'landscape_water': {
        'params': {
            'width': (3.5, 4.5), 'height': (3.0, 4.0),
            'curvature': (0.7, 0.85), 'symmetry': (0.6, 0.8),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.4, 0.6),
            'ornament_density': (0.4, 0.6), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['ocean', 'river', 'waterfall']
    },
    
    'animal_small_delicate': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.0, 2.5),
            'curvature': (0.8, 0.95), 'symmetry': (0.85, 0.95),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.7, 0.85),
            'ornament_density': (0.7, 0.9), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.4, 0.6),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.2, 0.4)
        },
        'words': ['butterfly', 'bee', 'ladybug', 'firefly']
    },
    
    'animal_small_agile': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.0, 2.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.8, 0.9),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['sparrow', 'dragonfly', 'spider']
    },
    
    'animal_medium_gentle': {
        'params': {
            'width': (2.5, 3.5), 'height': (2.5, 3.5),
            'curvature': (0.75, 0.9), 'symmetry': (0.8, 0.9),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.3, 0.5), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['rabbit', 'squirrel', 'cat', 'dove']
    },
    
    'animal_medium_wild': {
        'params': {
            'width': (3.0, 3.5), 'height': (3.0, 3.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.8, 0.9),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.4, 0.6),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['fox', 'owl', 'raven', 'hawk']
    },
    
    'animal_large_graceful': {
        'params': {
            'width': (3.5, 4.5), 'height': (4.0, 5.0),
            'curvature': (0.7, 0.85), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (3, 4), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['deer', 'horse', 'dolphin', 'giraffe']
    },
    
    'animal_large_powerful': {
        'params': {
            'width': (4.0, 4.5), 'height': (3.5, 4.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.4, 0.6), 'intensity': (0.7, 0.9),
            'detail_frequency': (3, 4), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.7, 0.9), 'valence_norm': (0.4, 0.6),
            'dominance_norm': (0.8, 0.95)
        },
        'words': ['bear', 'lion', 'tiger', 'elephant', 'whale']
    },
    
    'animal_distinctive': {
        'params': {
            'width': (3.0, 3.5), 'height': (3.0, 3.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.8, 0.9),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.75, 0.9),
            'ornament_density': (0.7, 0.85), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['zebra', 'wolf']
    },
    
    'architecture_sacred_grand': {
        'params': {
            'width': (3.5, 4.5), 'height': (6.0, 6.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.9, 0.95),
            'complexity': (0.7, 0.9), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.7, 0.9), 'intensity': (0.6, 0.8),
            'detail_frequency': (5, 6), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['cathedral', 'mosque', 'temple']
    },
    
    'architecture_sacred_modest': {
        'params': {
            'width': (3.0, 4.0), 'height': (5.0, 5.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.8, 0.95),
            'ornament_density': (0.6, 0.8), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['church']
    },
    
    'architecture_grand': {
        'params': {
            'width': (4.0, 5.0), 'height': (5.5, 6.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.7, 0.9), 'structural_fidelity': (0.8, 0.95),
            'ornament_density': (0.7, 0.9), 'intensity': (0.6, 0.8),
            'detail_frequency': (5, 6), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['palace', 'castle', 'mansion', 'tower']
    },
    
    'architecture_monument': {
        'params': {
            'width': (3.0, 4.0), 'height': (5.5, 6.5),
            'curvature': (0.5, 0.7), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['monument', 'statue', 'lighthouse']
    },
    
    'architecture_humble': {
        'params': {
            'width': (3.0, 3.5), 'height': (2.5, 3.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.8, 0.9),
            'complexity': (0.3, 0.5), 'structural_fidelity': (0.8, 0.9),
            'ornament_density': (0.2, 0.4), 'intensity': (0.4, 0.6),
            'detail_frequency': (2, 3), 'depth': (0.5, 0.7),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.3, 0.5), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.4, 0.6)
        },
        'words': ['cottage', 'cabin', 'barn']
    },
    
    'architecture_functional': {
        'params': {
            'width': (2.5, 3.5), 'height': (3.0, 4.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.8, 0.9),
            'ornament_density': (0.3, 0.5), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['bridge', 'gate', 'door', 'window', 'windmill']
    },
    
    'weapon_blade': {
        'params': {
            'width': (2.0, 2.5), 'height': (4.5, 5.5),
            'curvature': (0.3, 0.5), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.4, 0.6), 'intensity': (0.7, 0.9),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.7, 0.9), 'valence_norm': (0.3, 0.5),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['sword', 'arrow', 'spear']
    },
    
    'weapon_ranged': {
        'params': {
            'width': (2.5, 3.5), 'height': (3.5, 4.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.85, 0.95),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.4, 0.6),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.4, 0.6),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['bow']
    },
    
    'armor_protective': {
        'params': {
            'width': (3.0, 4.0), 'height': (3.5, 4.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.4, 0.6),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['armor', 'helmet', 'shield']
    },
    
    'precious_gem': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.0, 2.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.7, 0.9), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.8, 0.95), 'intensity': (0.8, 0.95),
            'detail_frequency': (5, 6), 'depth': (0.5, 0.7),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['diamond', 'ruby', 'emerald', 'sapphire', 'pearl']
    },
    
    'precious_metal': {
        'params': {
            'width': (2.5, 3.5), 'height': (2.5, 3.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.85, 0.95),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.8, 0.95),
            'ornament_density': (0.8, 0.95), 'intensity': (0.7, 0.9),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.7, 0.9)
        },
        'words': ['gold', 'silver', 'crown']
    },
    
    'instrument_string': {
        'params': {
            'width': (2.5, 3.5), 'height': (3.5, 4.5),
            'curvature': (0.75, 0.9), 'symmetry': (0.8, 0.9),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.6, 0.8), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['violin', 'guitar', 'harp']
    },
    
    'instrument_wind': {
        'params': {
            'width': (2.0, 2.5), 'height': (3.5, 4.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (4, 5), 'depth': (0.4, 0.6),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['flute', 'trumpet']
    },
    
    'instrument_percussion': {
        'params': {
            'width': (2.5, 3.0), 'height': (2.5, 3.0),
            'curvature': (0.8, 0.95), 'symmetry': (0.9, 0.95),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['drum', 'bell']
    },
    
    'instrument_keyboard': {
        'params': {
            'width': (3.5, 4.0), 'height': (3.0, 3.5),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.7, 0.9), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.6, 0.8), 'intensity': (0.6, 0.8),
            'detail_frequency': (5, 6), 'depth': (0.6, 0.8),
            'thickness': (0.35, 0.45),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['piano']
    },
    
    'writing_delicate': {
        'params': {
            'width': (1.5, 2.0), 'height': (3.5, 4.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.8, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.3, 0.5),
            'thickness': (0.2, 0.3),
            'arousal_norm': (0.3, 0.5), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.4, 0.6)
        },
        'words': ['pen', 'pencil', 'feather']
    },
    
    'writing_substantial': {
        'params': {
            'width': (2.5, 3.5), 'height': (3.0, 4.0),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.8, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['book', 'scroll', 'letter', 'ink']
    },
    
    'light_small': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.5, 3.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.8, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.7, 0.85),
            'ornament_density': (0.5, 0.7), 'intensity': (0.7, 0.9),
            'detail_frequency': (4, 5), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['candle', 'lantern', 'torch']
    },
    
    'light_flame': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.5, 3.5),
            'curvature': (0.8, 0.95), 'symmetry': (0.5, 0.7),
            'complexity': (0.6, 0.8), 'structural_fidelity': (0.3, 0.5),
            'ornament_density': (0.5, 0.7), 'intensity': (0.85, 0.95),
            'detail_frequency': (4, 5), 'depth': (0.4, 0.6),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.8, 0.95), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['flame', 'spark']
    },
    
    'food_fruit': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.0, 2.5),
            'curvature': (0.85, 0.95), 'symmetry': (0.85, 0.95),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.7, 0.85),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.7, 0.9),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['apple', 'cherry', 'grape', 'orange', 'lemon', 'peach', 'strawberry']
    },
    
    'food_sweet': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.0, 2.5),
            'curvature': (0.8, 0.95), 'symmetry': (0.75, 0.9),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.6, 0.8),
            'ornament_density': (0.5, 0.7), 'intensity': (0.6, 0.8),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.5, 0.7), 'valence_norm': (0.8, 0.95),
            'dominance_norm': (0.4, 0.6)
        },
        'words': ['chocolate', 'honey']
    },
    
    'food_staple': {
        'params': {
            'width': (2.5, 3.0), 'height': (2.0, 2.5),
            'curvature': (0.7, 0.85), 'symmetry': (0.8, 0.9),
            'complexity': (0.2, 0.4), 'structural_fidelity': (0.7, 0.85),
            'ornament_density': (0.2, 0.4), 'intensity': (0.3, 0.5),
            'detail_frequency': (2, 3), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.2, 0.4), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.3, 0.5)
        },
        'words': ['bread', 'milk']
    },
    
    'food_beverage': {
        'params': {
            'width': (2.0, 2.5), 'height': (2.5, 3.0),
            'curvature': (0.75, 0.9), 'symmetry': (0.85, 0.95),
            'complexity': (0.3, 0.5), 'structural_fidelity': (0.7, 0.85),
            'ornament_density': (0.4, 0.6), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.4, 0.6)
        },
        'words': ['coffee', 'tea', 'wine']
    },
    
    'household_functional': {
        'params': {
            'width': (2.5, 3.0), 'height': (2.5, 3.0),
            'curvature': (0.6, 0.8), 'symmetry': (0.85, 0.95),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.85, 0.95),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (4, 5), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.5, 0.7),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['clock', 'mirror', 'lock', 'key']
    },
    
    'time_transition': {
        'params': {
            'width': (3.0, 4.0), 'height': (3.0, 4.0),
            'curvature': (0.75, 0.9), 'symmetry': (0.75, 0.9),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.5, 0.7),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['dawn', 'dusk', 'morning', 'spring', 'summer', 'autumn', 'winter']
    },
    
    'sensory_sound': {
        'params': {
            'width': (2.5, 3.5), 'height': (2.5, 3.5),
            'curvature': (0.8, 0.95), 'symmetry': (0.7, 0.85),
            'complexity': (0.4, 0.6), 'structural_fidelity': (0.4, 0.6),
            'ornament_density': (0.5, 0.7), 'intensity': (0.4, 0.6),
            'detail_frequency': (3, 4), 'depth': (0.4, 0.6),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.3, 0.5), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.4, 0.6)
        },
        'words': ['echo', 'whisper', 'silence', 'music', 'song', 'melody', 'rhythm', 'harmony']
    },
    
    'water_feature': {
        'params': {
            'width': (3.0, 4.0), 'height': (3.0, 4.0),
            'curvature': (0.7, 0.85), 'symmetry': (0.75, 0.9),
            'complexity': (0.5, 0.7), 'structural_fidelity': (0.6, 0.8),
            'ornament_density': (0.5, 0.7), 'intensity': (0.5, 0.7),
            'detail_frequency': (3, 4), 'depth': (0.5, 0.7),
            'thickness': (0.3, 0.4),
            'arousal_norm': (0.4, 0.6), 'valence_norm': (0.6, 0.8),
            'dominance_norm': (0.5, 0.7)
        },
        'words': ['fountain', 'harbor', 'island']
    },
    
    'phenomena_colorful': {
        'params': {
            'width': (4.0, 5.0), 'height': (3.0, 4.0),
            'curvature': (0.8, 0.95), 'symmetry': (0.8, 0.95),
            'complexity': (0.7, 0.9), 'structural_fidelity': (0.6, 0.8),
            'ornament_density': (0.8, 0.95), 'intensity': (0.8, 0.95),
            'detail_frequency': (5, 6), 'depth': (0.5, 0.7),
            'thickness': (0.25, 0.35),
            'arousal_norm': (0.6, 0.8), 'valence_norm': (0.8, 0.95),
            'dominance_norm': (0.6, 0.8)
        },
        'words': ['rainbow']
    },
}