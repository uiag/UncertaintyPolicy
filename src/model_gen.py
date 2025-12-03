#Contains the LLM model and relevant functions
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed, GPT2LMHeadModel, GPT2TokenizerFast
import math

#Class to generate answers using LLM
class GenModel:
    #Initialize tokenizer and model
    def __init__(self, model_name, seed, device="cpu"):
        set_seed(seed)
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        #Currently only two models implemented, raise Exception otherwise
        if "flan-t5" in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        elif "gpt2" in model_name:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        else:
            raise Exception(f"Not implemented model name: {model_name}")

    #Function to generate a single answer and also return seq_logprob and token_probs to calculate uncertainty signals
    def generate(self, prompt, text_only):
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device) #Tokenize prompt
        outputs = self.model.generate(**inp, do_sample=False, return_dict_in_generate=True, output_scores=True) #Generate output
        generated_ids = outputs.sequences
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True) #Retrieve textual output

        #Return textual output if no seq_logprob and token_probs required
        if text_only:
            return text

        #Calculate seq_logprob and token_probs
        token_probs = []
        seq_logprob = 0.0
        for i, scores in enumerate(outputs.scores):
            probs = torch.softmax(scores, dim=-1)
            chosen_ids = generated_ids[:,i]
            tid = chosen_ids[0].item()
            p = probs[0, tid].item()
            seq_logprob += math.log(max(p, 1e-12))
            token_probs.append(probs[0].cpu().numpy())
        
        return text, seq_logprob, token_probs

    #Function to sample multiple answers to calculate diversity signals
    def sample_n(self, prompt, n=10):
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device) #Tokenize prompt
        outputs = self.model.generate(**inp, do_sample=True, return_dict_in_generate=True, output_scores=False, num_return_sequences=n) #Generate output
        samples = [self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True) for i in range(n)] #Retrieve textual samples
        return samples
