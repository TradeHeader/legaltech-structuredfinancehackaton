from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch

class LlamaModel:
    def __init__(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=2048).to(self.device)

        # Define generation configuration
        generation_config = GenerationConfig(
            max_length=512,  # The maximum length of the generated sequence
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )

        # Generate response
        outputs = self.model.generate(
            inputs['input_ids'],
            generation_config=generation_config,
            **inputs
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
