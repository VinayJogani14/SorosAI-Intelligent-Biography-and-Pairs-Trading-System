"""
CodeExplainer: Uses CodeT5 model to generate natural language explanations of Python code
"""

import torch
from transformers import RobertaForCausalLM, RobertaTokenizer
from typing import Optional, Dict, Any
import logging
from config import MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExplainer:
    """
    A class that uses CodeT5 to explain Python code in natural language
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the CodeExplainer with a CodeT5 model
        
        Args:
            model_name: Name of the pre-trained CodeT5 model to use
        """
        self.model_name = model_name or MODEL_CONFIG["code_t5_model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing CodeExplainer with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self._load_model()
            logger.info("CodeExplainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CodeExplainer: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the CodeT5 model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = RobertaForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def explain_code(self, code: str, max_length: Optional[int] = None) -> str:
        """
        Generate a natural language explanation for the given Python code
        
        Args:
            code: Python code string to explain
            max_length: Maximum length of the generated explanation
            
        Returns:
            Natural language explanation of the code
        """
        if not code.strip():
            return "No code provided to explain."
        
        try:
            # Prepare input
            input_text = f"Explain this Python code: {code}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=MODEL_CONFIG["code_t5_max_length"],
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate explanation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length or MODEL_CONFIG["max_explanation_length"],
                    temperature=MODEL_CONFIG["temperature"],
                    top_p=MODEL_CONFIG["top_p"],
                    top_k=MODEL_CONFIG["top_k"],
                    num_beams=MODEL_CONFIG["num_beams"],
                    early_stopping=MODEL_CONFIG["early_stopping"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output (remove the input part to get only the generated explanation)
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after the input prompt)
            explanation = full_output[len(input_text):].strip()
            
            logger.info(f"Generated explanation for code of length {len(code)}")
            return explanation if explanation else "Generated explanation is empty"
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def explain_function(self, function_code: str) -> str:
        """
        Generate explanation specifically for a function
        
        Args:
            function_code: Python function code
            
        Returns:
            Function explanation
        """
        prompt = f"Explain what this Python function does: {function_code}"
        return self._generate_with_prompt(prompt)
    
    def explain_class(self, class_code: str) -> str:
        """
        Generate explanation specifically for a class
        
        Args:
            class_code: Python class code
            
        Returns:
            Class explanation
        """
        prompt = f"Explain what this Python class does: {class_code}"
        return self._generate_with_prompt(prompt)
    
    def explain_algorithm(self, algorithm_code: str) -> str:
        """
        Generate explanation specifically for an algorithm
        
        Args:
            algorithm_code: Python algorithm code
            
        Returns:
            Algorithm explanation
        """
        prompt = f"Explain how this algorithm works: {algorithm_code}"
        return self._generate_with_prompt(prompt)
    
    def _generate_with_prompt(self, prompt: str, max_length: Optional[int] = None) -> str:
        """
        Generate explanation using a custom prompt
        
        Args:
            prompt: Custom prompt for generation
            max_length: Maximum length of generated text
            
        Returns:
            Generated explanation
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MODEL_CONFIG["code_t5_max_length"],
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate explanation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length or MODEL_CONFIG["max_explanation_length"],
                    temperature=MODEL_CONFIG["temperature"],
                    top_p=MODEL_CONFIG["top_p"],
                    top_k=MODEL_CONFIG["top_k"],
                    num_beams=MODEL_CONFIG["num_beams"],
                    early_stopping=MODEL_CONFIG["early_stopping"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output (remove the input part to get only the generated explanation)
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after the input prompt)
            explanation = full_output[len(prompt):].strip()
            return explanation if explanation else "Generated explanation is empty"
            
        except Exception as e:
            logger.error(f"Error generating explanation with custom prompt: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_type": "CodeT5",
            "max_input_length": MODEL_CONFIG["code_t5_max_length"],
            "max_output_length": MODEL_CONFIG["max_explanation_length"],
            "generation_params": {
                "temperature": MODEL_CONFIG["temperature"],
                "top_p": MODEL_CONFIG["top_p"],
                "top_k": MODEL_CONFIG["top_k"],
                "num_beams": MODEL_CONFIG["num_beams"]
            }
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
