#File to create figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')

#Function to plot the expected utility vs. the uncertainty for each sample in the test set
def plot_eu_vs_u(df, U_col='U', EU_col='EU_answer', savepath=None):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=U_col, y=EU_col, data=df, alpha=0.6)
    plt.xlabel('Uncertainty')
    plt.ylabel('EU(answer)')
    plt.title('EU(answer) vs Uncertainty')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)

#Function to plot the distribution of actions vs. uncertainty
def plot_action_distribution(binned, savepath=None):
    plt.figure(figsize=(10, 6))
    plt.plot(binned["center"], binned["clarify_rate"], 
             label="Clarify", linewidth=2)
    plt.plot(binned["center"], binned["answer_rate"], 
             label="Answer", linewidth=2)
    plt.plot(binned["center"], binned["abstain_rate"], 
             label="Abstain", linewidth=2)
    plt.xlabel("Uncertainty")
    plt.ylabel("Action Rate")
    plt.title("Action Distribution by Uncertainty")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)

#Function to plot the calibration of the model
def plot_uncertainty_calibration(binned, savepath=None):
    plt.figure(figsize=(10, 6))
    plt.plot(binned["center"], binned["correct_rate"], 
             label="Empirical correctness", linewidth=2)
    plt.plot(binned["center"], binned["p_correct_mean"], 
             label="Predicted correctness", linewidth=2)

    plt.plot(binned["center"], binned["fix_rate"], 
             label="Empirical correctness clarification", linewidth=2)
    plt.plot(binned["center"], binned["p_fix_mean"], 
             label="Predicted correctness clarification", linewidth=2)
    plt.xlabel("Uncertainty")
    plt.ylabel("Correct Answer Rate")
    plt.title("Model Calibration")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)

#Function to plot the expected utility for different policies
def plot_expected_utility_policy(binned, savepath=None):
    plt.figure(figsize=(10, 6))
    plt.plot(binned["center"], binned["eu_answer"], 
             label="EU(answer)", linewidth=2)
    plt.plot(binned["center"], binned["eu_clarify"], 
             label="EU(clarify)", linewidth=2)
    plt.plot(binned["center"], binned["eu_abstain"], 
             label="EU(abstain)", linewidth=2)
    plt.xlabel("Uncertainty")
    plt.ylabel("Expected Utility")
    plt.title("Expected Utility of Policy")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)

#Function to compare learned policy, random clarifications and always clarify
def plot_policy_comparison(pol, rnd, alc, savepath=None):
    plt.figure(figsize=(10, 6))
    plt.plot(pol["center"], pol["eu_mean"], label="Learned policy", linewidth=2)
    plt.plot(rnd["center"], rnd["eu_mean"], label="Random clarify", linewidth=2)
    plt.plot(alc["center"], alc["eu_mean"], label="Always clarify", linewidth=2)
    plt.xlabel("Uncertainty")
    plt.ylabel("Expected Utility")
    plt.title("Comparison of Policies")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)

#Function to plot the tradeoff between clarification costs and Expected Utility and clarification rate
def plot_cost_tradeoff(df, c_wrong=1.0, savepath=None):
    #Simulate different clarification costs
    c_clarify_values = np.linspace(0.0, 0.5, 20)
    results = []
    for c in c_clarify_values:
        EUa = df['p_corr'] - c_wrong*(1-df['p_corr'])
        EUc = df['p_fix'] - c_wrong*(1-df['p_fix']) - c
        EUab = 0
        actions = np.argmax(np.stack([EUa,EUc,np.full_like(EUa,EUab)],axis=1), axis=1)
        act_labels = np.array(['ANSWER','CLARIFY','ABSTAIN'])[actions]
        avg_util = np.mean(np.choose(actions,[EUa,EUc,EUab]))
        results.append((c, avg_util, (act_labels=='CLARIFY').mean()))
    r = pd.DataFrame(results, columns=['cost','utility','clarify_rate'])

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(r['cost'], r['utility'], label='Expected Utility', linewidth=2)
    ax1.set_xlabel('Clarification Cost')
    ax1.set_ylabel('Expected Utility')
    ax1.grid(True, ls='--', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(r['cost'], r['clarify_rate'], color='orange', linewidth=2, label='Clarification Rate')
    ax2.set_ylabel('Clarification Rate')
    ax1.set_title('Utility–Cost Tradeoff')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

