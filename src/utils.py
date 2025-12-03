#File containing utility functions
from datasets import load_dataset
import random
import pandas as pd

#Load dataset
def load_fever(seed, max_examples=None, split="train"):
    ds = load_dataset("fever", 'v1.0', split=split)
    ds = ds.shuffle(seed=seed) #Shuffel dataset

    #Only choose first n samples
    if max_examples:
        ds = ds.select(range(max_examples))

    df = ds.to_pandas()
    df = df.rename(columns={'label':'gold_label','claim':'claim_text'}) #rename label field
    print(df)
    df_len = len(df)
    print(f"Length Dataset: {df_len}")

    #Create train/validation/test set with sizes 75%/15%/15%
    cut = int(0.7*df_len)
    cut2 = int(0.85*df_len)
    df_train = df.iloc[:cut]
    df_val = df.iloc[cut:cut2]
    df_test = df.iloc[cut2:]
    return df_train, df_val, df_test

#Function to convert a claim into a prompt
def claim_to_prompt(claim, clarification=None):
    #Add clarification if given
    if clarification:
        prompt = f"You are a fact verification model. Your task is to classify each CLAIM with exactly one of the following labels:\n\nSupported\nRefuted\nNot enough info\n\nIf a CLARIFICATION is provided, it gives additional information for the classification. Answer with exactly one of the labels and do not explain your answer.\n\n### Task\n\nCLAIM: {claim}\nCLARIFICATION: {clarification}\nLABEL:"
    else:
        prompt = f"You are a fact verification model. Your task is to classify each CLAIM with exactly one of the following labels:\n\nSupported\nRefuted\nNot enough info\n\nIf a CLARIFICATION is provided, it gives additional information for the classification. Answer with exactly one of the labels and do not explain your answer.\n\n### Task\n\nCLAIM: {claim}\nCLARIFICATION:\nLABEL:"
    return prompt

#Function to simulate clarifications with given noise
def simulate_clarification(gold_label, noise_prob=0.15):
    #Simulated user reply: with prob (1-noise) return gold_label, else random wrong label
    labels = ["Supported","Refuted","Not enough info"]
    if random.random() > noise_prob:
        return gold_label
    else:
        other = random.choice([l for l in labels if l != gold_label])
        return other

#Function to canonicalize the outputs of the model
def canonicalize_pred(pred_text):
    if not pred_text:
        return None
    s = pred_text.lower()
    if s == "supported, refuted, not enough info":
        return None
    if "support" in s or "yes" in s or "true" in s:
        return "Supported"
    if "refut" in s or "false" in s or "incorrect" in s:
        return "Refuted"
    if "not" in s or "enough" in s or "insufficient" in s:
        return "Not enough info"
    return None

#Function to determin whether a given answer is correct
def is_correct(pred_text, gold_label):
    try:
        return canonicalize_pred(pred_text) == gold_label
    except:
        return False

#Function to bin data by uncertainty
def bin_by_uncertainty(df, clar=True, nbins=20):
    df = df.copy()
    df["bin"] = pd.qcut(df["Uncertainty"], q=nbins, duplicates='drop')
    grouped = df.groupby("bin")

    #Get mean of bins
    centers = grouped["Uncertainty"].mean()
    counts = grouped.size()
    correct_rate = grouped["p_corr_empirical"].mean()
    fix_rate = grouped["p_fix_empirical"].mean()
    if clar:
        p_correct_mean = grouped["p_corr"].mean()
        p_fix_mean = grouped["p_fix"].mean()
        eu_answer = grouped["EU_answer"].mean()
        eu_clarify = grouped["EU_clarify"].mean()
        eu_abstain = grouped["EU_abstain"].mean()
    else:
        p_correct_mean = None
        p_fix_mean = None
        eu_answer = None
        eu_clarify = None
        eu_abstain = None

    eu_mean = grouped["EU"].mean()
    clarify_rate = (grouped["action"]
                    .apply(lambda x: (x == "CLARIFY").mean()))
    answer_rate = (grouped["action"]
                   .apply(lambda x: (x == "ANSWER").mean()))
    abstain_rate = (grouped["action"]
                    .apply(lambda x: (x == "ABSTAIN").mean()))

    #output statistics for bins
    out = pd.DataFrame({
        "center": centers,
        "count": counts,
        "correct_rate": correct_rate,
        "fix_rate": fix_rate,
        "p_correct_mean": p_correct_mean,
        "p_fix_mean": p_fix_mean,
        "eu_mean": eu_mean,
        "eu_answer": eu_answer,
        "eu_clarify": eu_clarify,
        "eu_abstain": eu_abstain,
        "clarify_rate": clarify_rate,
        "answer_rate": answer_rate,
        "abstain_rate": abstain_rate,
    })
    out = out.sort_values("center")
    return out