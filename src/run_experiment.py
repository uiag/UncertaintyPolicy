#Main file to run the experiment
import argparse, os, random
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import *
from utils import load_fever, claim_to_prompt, simulate_clarification, is_correct, canonicalize_pred, bin_by_uncertainty
from model_gen import GenModel
from features import compute_uncertainty_features, normalize_features
from predictors import PredictorSuite
from policy import ExpectedUtilityPolicy
from evaluation import compute_summary, errors_prevented_per_clarify, mcNemar_test
from viz import plot_eu_vs_u, plot_action_distribution, plot_uncertainty_calibration, plot_cost_tradeoff, plot_expected_utility_policy, plot_policy_comparison

#Function to generate answers and uncertainty statistics for questions
def generate_for_df(df_part, gen):
    rows = []
    it = df_part.iterrows()
    for i, ex in tqdm(it, total=len(df_part)):
        claim = ex['claim_text']
        gold = canonicalize_pred(ex['gold_label'])
        prompt = claim_to_prompt(claim)

        greedy_ans, seq_logprob, token_probs = gen.generate(prompt, text_only=False)
        samples = gen.sample_n(prompt, n=args.sample_n)
        row = {
            'claim_text': claim,
            'gold_label': gold,
            'greedy_ans': greedy_ans,
            'seq_logprob': seq_logprob,
            'token_probs': token_probs,
            'samples': samples,
            'baseline_correct': is_correct(greedy_ans, gold),
            'p_corr_empirical': float(sum([is_correct(sample, gold) for sample in samples])/len(samples))
        }
        rows.append(row)
    return pd.DataFrame(rows)

#Function to simulate a clarification
def simulate_fix(df_part, gen, noise_prob=0.0):
        fixes = []
        clarified_ans = []
        clarified_samples = []
        p_fix_empirical = []
        it = df_part.iterrows()
        for i, r in tqdm(it, total=len(df_part)):
            user_reply = simulate_clarification(r['gold_label'], noise_prob=noise_prob)
            new_prompt = claim_to_prompt(r['claim_text'], user_reply)
            ans1 = gen.generate(new_prompt, text_only=True)
            clarified_ans.append(ans1)
            fixes.append(is_correct(ans1, r['gold_label']))
            samples = gen.sample_n(new_prompt, n=args.sample_n)
            clarified_samples.append(samples)
            p_fix_empirical.append(float(sum([is_correct(sample, r['gold_label']) for sample in samples])/len(samples)))
        df_part['clarified_ans'] = clarified_ans
        df_part['would_be_fixed_by_clarify'] = fixes
        df_part['clarified_samples'] = clarified_samples
        df_part['p_fix_empirical'] = p_fix_empirical
        return df_part

#Main function to run experiment
def run(args):
    #Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    #load dataset
    train_df, val_df, test_df = load_fever(seed=args.seed, max_examples=args.max_examples)
    #initiate LLM
    gen = GenModel(args.model_name, seed=args.seed, device=args.device)
    #generate baseline outputs for train/val/test
    print("Generating for train...")
    gen_train = generate_for_df(train_df, gen)
    print("Generating for val...")
    gen_val = generate_for_df(val_df, gen)
    print("Generating for test...")
    gen_test = generate_for_df(test_df, gen)

    #extract features
    for df_part in [gen_train, gen_val, gen_test]:
        feats = []
        for _, r in df_part.iterrows():
            f = compute_uncertainty_features(r)
            feats.append(f)
        feat_df = pd.DataFrame(feats)
        for col in feat_df.columns:
            df_part[col] = feat_df[col].values

    #simulate clarification on train/val/test and add noise to training data
    print("Generating fix for train...")
    gen_train = simulate_fix(gen_train, gen, noise_prob=args.clarify_noise)
    print("Generating fix for val...")
    gen_val = simulate_fix(gen_val, gen, noise_prob=args.clarify_noise)
    print("Generating fix for test...")
    gen_test = simulate_fix(gen_test, gen)

    #prepare features & labels
    feature_cols = ["avg_token_entropy", "max_token_entropy", "topk_gap_mean", "frac_low_prob_tokens", "seq_logprob_norm", 
        "sample_diversity", "sample_entropy", "num_unique_samples", "disagreement_ratio", "class_diversity", "num_unique_classes"]
    gen_train, gen_val, gen_test = normalize_features(gen_train, gen_val, gen_test, feature_cols) #Rescale features to [0,1]
    feature_cols.append("Uncertainty")
    print("Feature cols:", feature_cols)

    #Fit predictors
    pred = PredictorSuite(feature_cols, args.seed)
    print("Training linear...")
    pred.fit_linear(gen_train, args.output_dir)
    print("Training lightgbm...")
    pred.fit_lightgbm(gen_train, gen_val, args.output_dir)
    print("Training mlp (this may take time)...")
    pred.fit_mlp(gen_train, gen_val, args.output_dir)

    #Predict probabilities on test set
    p_linear_corr, p_linear_fix = pred.predict_linear(gen_test)
    p_lgb_corr, p_lgb_fix = pred.predict_lightgbm(gen_test)
    p_mlp_corr, p_mlp_fix = pred.predict_mlp(gen_test)

    #we will select one predictor for policy evaluations at a time
    results = []
    for name, (p_corr, p_fix) in [('Linear', (p_linear_corr, p_linear_fix)),('LightGBM',(p_lgb_corr,p_lgb_fix)),('MLP',(p_mlp_corr,p_mlp_fix))]:
        df_eval = gen_test.copy()
        df_eval['p_corr'] = p_corr
        df_eval['p_fix'] = p_fix
        #clip to [0,1]
        df_eval['p_corr'] = df_eval['p_corr'].clip(0,1)
        df_eval['p_fix'] = df_eval['p_fix'].clip(0,1)

        #Run EU policy
        policy = ExpectedUtilityPolicy(c_clarify=args.c_clarify, c_wrong=args.c_wrong)
        actions = []
        final_answers = []
        final_corrects = []
        eus = []
        eu_action = []
        for _, r in df_eval.iterrows():
            #Get action
            action, eu = policy.choose(float(r['p_corr']), float(r['p_fix']))
            if action == 'ANSWER':
                final = r['greedy_ans']
            elif action == 'ABSTAIN':
                final = None
            elif action == 'CLARIFY':
                final = r['clarified_ans']
            final_corr = is_correct(final, r['gold_label']) if final is not None else False
            actions.append(action)
            final_answers.append(final)
            final_corrects.append(final_corr)
            eus.append(eu)
            eu_action.append(eu[action])
        df_eval['action'] = actions
        df_eval['final_ans'] = final_answers
        df_eval['final_correct'] = final_corrects
        df_eval['EU_answer'] = [e['ANSWER'] for e in eus]
        df_eval['EU_clarify'] = [e['CLARIFY'] for e in eus]
        df_eval['EU_abstain'] = [e['ABSTAIN'] for e in eus]
        df_eval['EU'] = eu_action

        #Baselines:
        #Always answer
        df_always = gen_test.copy()
        df_always['action'] = 'ANSWER'
        df_always['final_ans'] = df_always['greedy_ans']
        df_always['final_correct'] = df_always.apply(lambda r: is_correct(r['greedy_ans'], r['gold_label']), axis=1)
        df_always['EU'] = df_eval['EU_answer']
        
        #Always clarify
        df_clar = gen_test.copy()
        df_clar['action'] = 'CLARIFY'
        df_clar['final_ans'] = df_clar['clarified_ans']
        df_clar['final_correct'] = df_clar.apply(lambda r: is_correct(r['final_ans'], r['gold_label']), axis=1)
        df_clar['EU'] = df_eval['EU_clarify']

        #Randomized clarify with same clarify rate as EU policy
        clarify_prob = (df_eval['action']=='CLARIFY').mean()
        df_rand = gen_test.copy()
        finals_rand = []
        acts_rand = []
        eu = []
        for i, r in df_rand.iterrows():
            if random.random() < clarify_prob:
                finals_rand.append(r['clarified_ans'])
                acts_rand.append('CLARIFY')
                eu.append(df_eval['EU_clarify'].iloc[i])
            else:
                finals_rand.append(r['greedy_ans'])
                acts_rand.append('ANSWER')
                eu.append(df_eval['EU_answer'].iloc[i])
        df_rand['action'] = acts_rand
        df_rand['final_ans'] = finals_rand
        df_rand['final_correct'] = df_rand.apply(lambda r: is_correct(r['final_ans'], r['gold_label']), axis=1)
        df_rand['EU'] = eu

        #Evaluate results
        sum_eu = compute_summary(df_eval, action_col='action', correct_col='final_correct')
        sum_always = compute_summary(df_always, action_col='action', correct_col='final_correct')
        sum_clar = compute_summary(df_clar, action_col='action', correct_col='final_correct')
        sum_rand = compute_summary(df_rand, action_col='action', correct_col='final_correct')

        #Get statistical significance of policy better than randomized clarifications
        filtered_index = df_eval.index[df_eval['action'] != "ABSTAIN"]
        obs_answered, pval_answered = mcNemar_test(df_eval, df_rand, filtered_index)
        obs, pval = mcNemar_test(df_eval, df_rand, df_eval.index)

        #compute errors prevented per clarification
        epc = errors_prevented_per_clarify(df_eval, baseline_correct_col='baseline_correct')
        epc_rand = errors_prevented_per_clarify(df_rand, baseline_correct_col='baseline_correct')

        #save results
        res = {'predictor':name, 'eu_summary':sum_eu, 'always_summary':sum_always, 'clar_summary':sum_clar, 'rand_summary':sum_rand,
               'policy_better_than_randomized':(obs,pval), 'policy_better_than_randomized_answered': (obs_answered, pval_answered),'epc':epc, 'epc_rand':epc_rand}
        results.append((name, df_eval, df_always, df_clar, df_rand, res))

        #plots
        binned = bin_by_uncertainty(df_eval)
        binned_rand = bin_by_uncertainty(df_rand, clar=False)
        binned_clar = bin_by_uncertainty(df_clar, clar=False)
        plot_eu_vs_u(df_eval, U_col='Uncertainty', EU_col='EU_answer', savepath=os.path.join(args.output_dir,f'eu_vs_u_{name}.png'))
        plot_action_distribution(binned, savepath=os.path.join(args.output_dir,f'action_distr_{name}.png'))
        plot_uncertainty_calibration(binned, savepath=os.path.join(args.output_dir,f'uncertainty_cal_{name}.png'))
        plot_expected_utility_policy(binned, savepath=os.path.join(args.output_dir,f'eu_policy_{name}.png'))
        plot_policy_comparison(binned, binned_rand, binned_clar, savepath=os.path.join(args.output_dir,f'policy_comparison_{name}.png'))
        plot_cost_tradeoff(df_eval, c_wrong=args.c_wrong, savepath=os.path.join(args.output_dir,f'cost_tradeoff_{name}.png'))

    #write outputs
    for name, df_eval, df_always, df_clar, df_rand, res in results:
        df_eval.to_csv(os.path.join(args.output_dir, f"eval_{name}.csv"), index=False)
        pd.DataFrame([res]).to_csv(os.path.join(args.output_dir, f"summary_{name}.csv"), index=False)
    print("Done.")

#Argument parser for running. Uses inputs from config.py by default
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--max_examples", type=int, default=DATA_N)
    parser.add_argument("--sample_n", type=int, default=SAMPLE_N)
    parser.add_argument("--c_clarify", type=float, default=C_CLARIFY)
    parser.add_argument("--c_wrong", type=float, default=C_WRONG)
    parser.add_argument("--clarify_noise", type=float, default=NOISE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default="outputs/run")
    args = parser.parse_args()
    run(args)
