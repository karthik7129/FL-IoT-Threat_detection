#!/usr/bin/env python3
"""
Real-time IoT Threat Detection Pipeline
Loads trained model and makes predictions on extracted features
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Any
import json
from datetime import datetime

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NeuralNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IoTThreatPredictor:
    """
    Real-time IoT threat detection using trained neural network model
    """
    
    def __init__(self, model_path: str, num_features: int = 17, num_classes: int = 10):
        self.model_path = model_path
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Attack type mapping
        self.attack_names = {
            0: "Benign",
            1: "Gafgyt Combo",
            2: "Gafgyt Junk", 
            3: "Gafgyt TCP",
            4: "Gafgyt Scan",  # if present in your data
            5: "Mirai ACK",
            6: "Mirai Scan",
            7: "Mirai SYN",
            8: "Mirai UDP",
            9: "Mirai UDPPlain"
        }
        
        # Feature names in correct order
        self.feature_names = [
            'HH_jit_L1_variance', 'HH_jit_L0.01_variance', 'HH_jit_L3_variance',
            'HH_jit_L0.1_variance', 'HH_jit_L0.1_mean', 'H_L0.01_variance',
            'HH_jit_L5_variance', 'MI_dir_L0.01_variance', 'MI_dir_L0.1_mean',
            'MI_dir_L0.01_weight', 'HH_L0.01_std', 'MI_dir_L0.1_weight',
            'H_L0.01_mean', 'H_L0.01_weight', 'HH_jit_L5_mean',
            'MI_dir_L1_weight', 'H_L0.1_weight'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the trained neural network model"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Initialize model
            self.model = NeuralNetwork(self.num_features, self.num_classes)
            
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the entire dict is the state dict
                    self.model.load_state_dict(checkpoint)
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_features(self, features_df: pd.DataFrame) -> torch.Tensor:
        """Preprocess features for model prediction"""
        try:
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    logger.warning(f"Missing feature: {feature}, setting to 0")
                    features_df[feature] = 0.0
            
            feature_data = features_df[self.feature_names].copy()
            feature_data = feature_data.fillna(0.0)
            feature_data = feature_data.replace([np.inf, -np.inf], 0.0)
            
            for col in feature_data.columns:
                col_min = feature_data[col].min()
                col_max = feature_data[col].max()
                if col_max > col_min:
                    feature_data[col] = (feature_data[col] - col_min) / (col_max - col_min)
                else:
                    feature_data[col] = 0.0
            
            tensor = torch.FloatTensor(feature_data.values).to(self.device)
            logger.info(f"Preprocessed features shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return torch.FloatTensor([]).to(self.device)
    
    def predict(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Make predictions on feature data"""
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        if features_df.empty:
            logger.warning("No features provided for prediction")
            return []
        
        try:
            # Preprocess features
            feature_tensor = self.preprocess_features(features_df)
            
            if feature_tensor.numel() == 0:
                logger.error("Failed to preprocess features")
                return []
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(feature_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
            
            # Convert to results
            results = []
            for i in range(len(features_df)):
                pred_class = predicted_classes[i].cpu().item()
                prob_scores = probabilities[i].cpu().numpy()
                
                result = {
                    'flow_id': features_df.iloc[i].get('flow_id', f'flow_{i}'),
                    'predicted_class': pred_class,
                    'predicted_attack': self.attack_names.get(pred_class, 'Unknown'),
                    'confidence': float(prob_scores[pred_class]),
                    'probability_scores': {
                        self.attack_names.get(j, f'Class_{j}'): float(prob_scores[j])
                        for j in range(len(prob_scores))
                    },
                    'timestamp': datetime.now().isoformat(),
                    'is_malicious': pred_class != 0,  # 0 is benign
                    'threat_level': self._get_threat_level(pred_class, float(prob_scores[pred_class]))
                }
                
                results.append(result)
            
            logger.info(f"Made predictions for {len(results)} flows")
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return []
    
    def _get_threat_level(self, predicted_class: int, confidence: float) -> str:
        """Determine threat level based on prediction and confidence"""
        if predicted_class == 0:  # Benign
            return "None"
        
        if confidence >= 0.9:
            return "High"
        elif confidence >= 0.7:
            return "Medium"
        elif confidence >= 0.5:
            return "Low"
        else:
            return "Uncertain"
    
    def predict_single_flow(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for a single flow"""
        # Convert single flow features to DataFrame
        df = pd.DataFrame([features])
        results = self.predict(df)
        return results[0] if results else {}
    
    def save_predictions(self, predictions: List[Dict[str, Any]], output_file: str):
        """Save predictions to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Predictions saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "status": "Model loaded",
            "model_path": self.model_path,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "attack_types": self.attack_names,
            "feature_names": self.feature_names
        }

def main():
    """Main function for testing the predictor"""
    # Initialize predictor
    model_path = "../SavedGlobalModel/final_model.pth"
    predictor = IoTThreatPredictor(model_path)
    
    # Print model info
    model_info = predictor.get_model_info()
    print("Model Info:")
    print(json.dumps(model_info, indent=2))
    
    # Test with dummy features
    dummy_features = {feature: np.random.random() for feature in predictor.feature_names}
    dummy_features['flow_id'] = 'test_flow_1'
    
    print("\nTesting single flow prediction:")
    result = predictor.predict_single_flow(dummy_features)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
