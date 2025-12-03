#This file contains the run configuration
MODEL_NAME = "google/flan-t5-base" #Name of the LLM to use (only GPT2 and Flan-T5 work)
DEVICE = "cuda"  #Use GPU
DATA_N = 10000 #Samples to use in total
SAMPLE_N = 25 #How many samples to use for sample based features
SEED = 1234 #Seed
NOISE = 0.15 #Noise level to use during clarification simulation
C_CLARIFY = 0.25 #Cost of clarification (0.25 is equivalent to a necessary increase in correct prediction of 12.5% such that a clarification is better than answering)
C_WRONG = 1.0 #Penalty for a wrong answer (Correct answer is 1 in utility, thus only choose answer over abstention if correct answer probability > 0.5)