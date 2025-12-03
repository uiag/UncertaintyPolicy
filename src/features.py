#File to calculate the uncertainty features
import numpy as np
import re
from collections import Counter
from utils import canonicalize_pred
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math

#Function to normalize a given text
def normalize_text(s):
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s

#Function to find the majority string in given samples
def majority_string(samples):
    norm = [normalize_text(s) for s in samples]
    c = Counter(norm)
    most, count = c.most_common(1)[0]
    return most, count

#Calcualte diversity score of samples
def diversity_score(samples):
    if not samples:
        return None
    _, maj_count = majority_string(samples)
    return 1.0 - (maj_count / len(samples))

#Calculate the topk gap mean using token_probs
def topk_gap_mean(token_probs):
    if not token_probs:
        return None
    gaps = []
    for probs in token_probs:
        p = np.array(probs)
        ids = np.argsort(p)[-2:]
        top = p[ids[-1]]
        second = p[ids[-2]]
        gaps.append(float(top - second))
    return -1*float(np.mean(gaps)) #Scale with -1 as higher value in feature should represent higher uncertainty

#Calcualte token_entropy from token_probs
def token_entropy(token_probs):
    if not token_probs:
        return None
    ent = []
    for probs in token_probs:
        p = np.clip(np.array(probs), 1e-12, 1.0)
        ent.append(-np.sum(p * np.log(p)))
    return float(np.mean(ent)), float(np.max(ent))

#Calculate seq_logprob_norm
def seq_logprob_norm(seq_logprob, length):
    if seq_logprob is None or length == 0:
        return None
    return -1*float(seq_logprob / max(1, length)) #Scale with -1 as higher value in feature should represent higher uncertainty

#Calculate fraction of low probability tokens
def frac_low_prob_tokens(token_probs, thresh=0.5):
    if not token_probs:
        return None
    low = 0
    total = 0
    for p in token_probs:
        p = np.asarray(p, dtype=float)
        if p.size == 0:
            continue
        total += 1
        if float(p.max()) < thresh:
            low += 1
    if total == 0:
        return None
    return float(low / len(token_probs))

#Calculate the entropy of the samples
def sample_entropy(samples):
    if not samples:
        return None
    norm = [normalize_text(s) for s in samples]
    counts = Counter(norm)
    probs = np.array(list(counts.values()), dtype=float) / float(len(norm))
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))

#Count the number of unique samples
def num_unique_samples(samples):
    if not samples:
        return 0
    return len(set([normalize_text(s) for s in samples]))

#Calculate the diversity in classes
def class_diversity(samples):
    if not samples:
        return None
    classes = [canonicalize_pred(s) for s in samples]
    counts = Counter(classes)
    top_count = counts.most_common(1)[0][1]
    return 1 - float(top_count / len(samples))

#Count the number of unique classes
def num_unique_classes(samples):
    if not samples:
        return 0
    return len(set([canonicalize_pred(s) for s in samples]))

#Calculate the disagreement ratio between the samples and the greedy answer
def disagreement_ratio(samples, greedy):
    if not samples:
        return None
    norm_g = normalize_text(greedy)
    norm = [normalize_text(s) for s in samples]
    disagree = sum([1 for s in norm if s != norm_g])
    return float(disagree / max(1, len(norm)))

#Function to compute all uncertainty features for given sample
def compute_uncertainty_features(row):
    out = {}
    greedy = row.get('greedy_ans', "")
    samples = row.get('samples', [])
    token_probs = row.get('token_probs', [])
    seq_logprob = row.get('seq_logprob', None)

    # token-level
    out['avg_token_entropy'], out['max_token_entropy'] = token_entropy(token_probs)
    out['topk_gap_mean'] = topk_gap_mean(token_probs)
    out['frac_low_prob_tokens'] = frac_low_prob_tokens(token_probs)

    # sequence-level
    out['seq_logprob_norm'] = seq_logprob_norm(seq_logprob, len(str(greedy).split()))

    # sampling / self-consistency
    out['sample_diversity'] = diversity_score(samples)
    out['sample_entropy'] = sample_entropy(samples)
    out['num_unique_samples'] = num_unique_samples(samples)
    out['disagreement_ratio'] = disagreement_ratio(samples, greedy)

    # class-level
    out['class_diversity'] = class_diversity(samples)
    out['num_unique_classes'] = num_unique_classes(samples)

    return out

#Function to rescale features to [0,1]
def normalize_features(df_train, df_val, df_test, feature_cols):
    #Fit scaler only on training data
    scaler = MinMaxScaler()
    scaler.fit(df_train[feature_cols])

    def transform(df):
        #Normalize and overwrite the original feature columns
        df_norm = df.copy()
        df_norm[feature_cols] = scaler.transform(df[feature_cols])

        #Add the mean of the normalized feature columns as Uncertainty feature (used in Figures)
        df_norm["Uncertainty"] = df_norm[feature_cols].mean(axis=1)

        return df_norm

    return transform(df_train), transform(df_val), transform(df_test)
