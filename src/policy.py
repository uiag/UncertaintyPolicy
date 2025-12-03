#File for the expected utility policy

#Class for Policy
class ExpectedUtilityPolicy:
    #Initialize costs
    def __init__(self, c_clarify=0.25, c_wrong=1.0):
        self.c_clarify = c_clarify
        self.c_wrong = c_wrong

    #Choose action given probability of correct answer and probability of correct answer after clarification
    def choose(self, p_corr, p_fix):
        #EU calculations
        EU_answer = p_corr * 1.0 + (1-p_corr)*(-self.c_wrong)
        EU_clarify = p_fix * 1.0 + (1-p_fix)*(-self.c_wrong) - self.c_clarify
        EU_abstain = 0.0
        
        eu = {'ANSWER': EU_answer, 'CLARIFY': EU_clarify, 'ABSTAIN': EU_abstain}
        action = max(eu.items(), key=lambda kv: kv[1])[0]
        return action, eu
