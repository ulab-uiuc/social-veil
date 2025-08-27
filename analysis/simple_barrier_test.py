#!/usr/bin/env python3
"""
Simple Barrier Representation Test

A minimal script to test barrier effects on model representations
without requiring the full barrier creation pipeline.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("❌ Please install transformers: pip install transformers torch")
    sys.exit(1)

class SimpleBarrierTest:
    """Simple test of barrier effects on model representations"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            output_hidden_states=True
        )
        self.model.eval()
    
    def create_test_prompts(self):
        """Create test prompts using actual templates from social_task.yaml"""
        import yaml
        
        # Load the actual social task templates
        config_path = Path(__file__).parent.parent / "configs" / "social_task.yaml"
        with open(config_path, 'r') as f:
            templates = yaml.safe_load(f)
        
        # Test scenario and agent info
        test_data = {
            "agent_name": "Alex",
            "partner_name": "Jamie", 
            "scenario": "You are at a coffee shop and want to ask someone if you can share their table.",
            "agent_age": "28",
            "agent_gender": "person",
            "agent_occupation": "software developer",
            "agent_public_info": "friendly and polite",
            "partner_age": "30", 
            "partner_gender": "person",
            "partner_occupation": "teacher",
            "partner_public_info": "busy with work",
            "agent_goal": "Get permission to sit at the shared table and start a friendly conversation.",
            "agent_reason": "You need a place to sit and work on your laptop.",
            "agent_private_knowledge": "You noticed the person looks stressed about their work.",
            "turn_number": 1,
            "history": "",
            "action_list": templates.get("action_list", "")
        }
        
        # Create baseline prompt using actual template
        baseline_prompt = templates["social_task_instructions"].format(**test_data)
        
        # Create barrier prompts using actual barrier templates
        # For barrier prompts, we need to add barrier-specific fields
        barrier_data = test_data.copy()
        
        # Semantic barrier
        barrier_data.update({
            "barrier_private_note": "Use vague, ambiguous language",
            "barrier_prompt": "Throughout the conversation, consistently introduce ambiguity in how you describe things. Prefer vague vocabulary ('sort of', 'maybe', 'kind of'), avoid exact words ('exactly', 'I want').",
            "barrier_dynamic_rules": "- Use phrases like: perhaps, kind of, roughly\n- Avoid phrases: exactly, I want, specifically\n- Sentence length bias: medium"
        })
        semantic_prompt = templates["social_task_instructions_barrier_semantic"].format(**barrier_data)
        
        # Cultural barrier
        barrier_data.update({
            "barrier_private_note": "Use indirect, high-context communication style",
            "barrier_prompt": "Use indirect, high-context style. Prefer hedges ('perhaps', 'it seems', 'might'), avoid blunt imperatives. Convey requests indirectly, embed meaning in context.",
            "barrier_dynamic_rules": "- Use phrases like: perhaps, it seems, might\n- Avoid phrases: do this, I need, give me\n- Question rate hint: 0.3\n- Imperative rate hint: 0.1"
        })
        cultural_prompt = templates["social_task_instructions_barrier_cultural"].format(**barrier_data)
        
        # Emotional barrier
        barrier_data.update({
            "barrier_private_note": "Maintain negative emotional tone, be clipped and sharp", 
            "barrier_prompt": "Maintain a negative emotional tone throughout. Use clipped, sharp wording. Avoid empathy or soothing phrases ('sorry', 'I understand', 'thank you').",
            "barrier_dynamic_rules": "- Affect lexicon: frustrated, impatient, annoyed, terse\n- Exclamation bias: 0.2\n- Sentence length bias: short"
        })
        emotional_prompt = templates["social_task_instructions_barrier_emotional"].format(**barrier_data)

        return {
            "baseline": baseline_prompt,
            "semantic": semantic_prompt, 
            "cultural": cultural_prompt,
            "emotional": emotional_prompt
        }
    
    def extract_representations(self, prompts):
        """Extract model representations for each prompt type"""
        
        representations = {}
        
        # Target the middle layer for analysis
        target_layer = len(self.model.model.layers) // 2
        print(f"📊 Analyzing layer {target_layer} (middle layer)")
        
        for barrier_type, prompt in prompts.items():
            print(f"  Processing {barrier_type}...")
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states
                
                # Get representation from target layer (last token)
                layer_repr = hidden_states[target_layer][0, -1, :].cpu().numpy()
                representations[barrier_type] = layer_repr
        
        return representations
    
    def analyze_differences(self, representations):
        """Analyze differences between representations"""
        
        print("\n📊 Computing representation differences...")
        
        baseline = representations["baseline"]
        results = {}
        
        for barrier_type in ["semantic", "cultural", "emotional"]:
            barrier_repr = representations[barrier_type]
            
            # Cosine similarity
            cosine_sim = np.dot(baseline, barrier_repr) / (
                np.linalg.norm(baseline) * np.linalg.norm(barrier_repr)
            )
            
            # Euclidean distance
            euclidean_dist = np.linalg.norm(baseline - barrier_repr)
            
            # Mean squared error
            mse = np.mean((baseline - barrier_repr) ** 2)
            
            results[barrier_type] = {
                "cosine_similarity": float(cosine_sim),
                "euclidean_distance": float(euclidean_dist), 
                "mse": float(mse)
            }
            
            print(f"  {barrier_type.capitalize()}:")
            print(f"    Cosine similarity: {cosine_sim:.4f}")
            print(f"    Euclidean distance: {euclidean_dist:.4f}")
            print(f"    MSE: {mse:.6f}")
        
        return results
    
    def create_visualization(self, representations, output_dir="results/simple_barrier_test"):
        """Create visualization of representation differences"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for PCA
        data_matrix = np.stack([representations[k] for k in ["baseline", "semantic", "cultural", "emotional"]])
        labels = ["Baseline", "Semantic", "Cultural", "Emotional"]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_matrix)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            plt.scatter(pca_result[i, 0], pca_result[i, 1], 
                       c=color, s=200, label=label, alpha=0.8)
            plt.annotate(label, (pca_result[i, 0], pca_result[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12) 
        plt.title('Barrier Effects on Model Representations\n(PCA Visualization)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/barrier_pca.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distance matrix heatmap
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Compute pairwise distances
        distance_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                distance_matrix[i, j] = np.linalg.norm(data_matrix[i] - data_matrix[j])
        
        im = ax.imshow(distance_matrix, cmap='viridis')
        
        # Set ticks and labels
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_title("Pairwise Euclidean Distances", fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Euclidean Distance')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualizations saved to {output_dir}/")
    
    def run_test(self):
        """Run complete barrier representation test"""
        
        print("🚀 Running Simple Barrier Representation Test")
        print("=" * 50)
        
        # 1. Create test prompts
        prompts = self.create_test_prompts()
        
        # 2. Extract representations
        representations = self.extract_representations(prompts)
        
        # 3. Analyze differences
        results = self.analyze_differences(representations)
        
        # 4. Create visualizations
        self.create_visualization(representations)
        
        # 5. Summary
        print("\n🎯 Summary:")
        print("  This test shows whether barrier prompts create measurable")
        print("  differences in model internal representations.")
        print()
        print("  Key metrics:")
        print("  - Cosine similarity: closer to 1.0 = more similar representations")
        print("  - Euclidean distance: larger values = more different representations") 
        print("  - MSE: larger values = more different representations")
        print()
        print("  If barriers are effective, we should see:")
        print("  - Cosine similarities < 0.95") 
        print("  - Noticeable differences in the PCA plot")
        print("  - Different distance patterns in the heatmap")
        
        return results

def main():
    """Main entry point"""
    
    # Check for required packages
    required_packages = ["transformers", "torch", "sklearn", "matplotlib", "scipy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return
    
    # Run test
    tester = SimpleBarrierTest()
    results = tester.run_test()
    
    print("\n✅ Test complete! Check results/simple_barrier_test/ for visualizations")

if __name__ == "__main__":
    main()