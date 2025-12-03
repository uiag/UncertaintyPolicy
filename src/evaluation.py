#File to conduct the evaluation of the results
import numpy as np
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar
from utils import canonicalize_pred

#Compute summary for given data
def compute_summary(df, action_col='action', correct_col='final_correct'):
    total = len(df)
    answered = df[df[action_col]!='ABSTAIN'].shape[0] #Amount of answered questions
    clarify = (df[action_col]=='CLARIFY').sum() #Amount of clarified questions
    acc_answered = df[df[action_col]!='ABSTAIN'][correct_col].mean() if answered>0 else 0.0 #Accuracy on answered questions
    overall_acc = df[correct_col].mean() #Accuracy over all questions (counting ABSTAIN as wrong)

    #Calculate F1 score for all questions and answered questions
    answered_df = df[df[action_col]!='ABSTAIN']
    f_score_answered = calculate_f1(answered_df, correct_col)
    f_score = calculate_f1(df, correct_col)

    return {'total':total,'answer_rate':answered/total,'clarify_rate':clarify/total,'acc_answered':acc_answered,'overall_acc':overall_acc, 'F1_score_answered': f_score_answered, 'F1_score': f_score}

#Calculate F1 score for given data
def calculate_f1(answered_df, correct_col):
    errors = answered_df[answered_df[correct_col] == False].shape[0] #Amount of wrong answers
    tp_supported = answered_df[answered_df[correct_col] & (answered_df["final_ans"].apply(canonicalize_pred) == "Supported")].shape[0] #Amount of True Positives for Supported
    tp_refuted = answered_df[answered_df[correct_col] & (answered_df["final_ans"].apply(canonicalize_pred) == "Refuted")].shape[0] #Amount of True Positives for Refuted
    tp_not_enough_info = answered_df[answered_df[correct_col] & (answered_df["final_ans"].apply(canonicalize_pred) == "Not enough info")].shape[0] #Amount of True Positives for Not enough info

    #Calculate individual F1 scores and average over them
    try:
        f_score_supported = (2*tp_supported)/(2*tp_supported + errors)
        f_score_refuted = (2*tp_refuted)/(2*tp_refuted + errors)
        f_score_not_enough_info = (2*tp_not_enough_info)/(2*tp_not_enough_info + errors)
        f_score = (f_score_supported+f_score_refuted+f_score_not_enough_info)/3
    except:
        f_score = None
    return f_score

#Function to quantify statistical significance using McNemar test
def mcNemar_test(df_eval, df_rand, filtered_index):

    A = np.array(df_eval.loc[filtered_index, "final_correct"].astype(int).values) #Correct answers using policy
    B = np.array(df_rand.loc[filtered_index, "final_correct"].astype(int).values) #Correct answers using randomized clarifications

    #Build contingency table
    a = np.sum((A == True) & (B == True))
    b = np.sum((A == True) & (B == False))
    c = np.sum((A == False) & (B == True))
    d = np.sum((A == False) & (B == False))

    table = [[a, b],
            [c, d]]

    #Run test (Special case of Cochran’s Q test)
    result = mcnemar(table, exact=False, correction=False)
    return b>c, result.pvalue #Return (Policy better than randomized clarifications, p-value)

#Calculate amount of errors prevented per clarification
def errors_prevented_per_clarify(df, baseline_correct_col='baseline_correct'):
    clarified = df[df['action']=='CLARIFY'] #Clarified questions
    if len(clarified)==0:
        return 0.0
    prevented = ((~clarified[baseline_correct_col]) & (clarified['final_correct'])).sum() #Preventions are baseline wrong and final correct
    return prevented / len(clarified)
