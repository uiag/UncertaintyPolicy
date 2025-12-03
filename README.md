# Uncertainty-Aware Hallucination Detection and Clarifying-Question Policy for LLMs
The project Uncertainty-Aware Hallucination Detection and Clarifying-Question Policy for LLMs tests whether uncertainty signals in LLM outputs can drive a useful policy to decide between answering, clarifying and abstaining. This project was part of the course Natural Language Processing at Seoul National University in Fall 2025

# How to run
1. Use Python 3.12
2. Install the project requirements via ``pip install -r requirements.txt``
3. Configure the run configuration in ``config.py`` (Currently contains configurations used for the FLAN-T5-base results. For the FLAN-T5-large and GPT2 results just adjust the MODEL_NAME and reduce DATA_N to 10 000)
4. Run the ``run_experiment.py`` script to start the analysis

# Method (automatically done by running ``run_experiment.py``)
1. Ask LLM to label each claim (FEVER v1.0 claims used) twice (with and without clarification)
2. Simulate clarification by giving noisy (15% noise) oracle reply to the model (acts as best case upper bound if model gets very good clarification)
3. Measure 11 uncertainty signals in output for standard answer (average token entropy, max token entropy, top2 gap mean, fraction of low probability tokens, normalized sequence log probability, sample diversity, sample entropy, number of unique samples, disagreement ratio, class diversity and number of unique classes)
4. Estimate 𝑝(𝑐𝑜𝑟𝑟𝑒𝑐𝑡│𝑈) and 𝑝(𝑐𝑜𝑟𝑟𝑒𝑐𝑡_𝑐𝑙𝑎𝑟𝑖𝑓𝑖𝑒𝑑|𝑈) using policy model (Ridge Regression, Neural Network and Gradient Boosted Regression Trees used) 
5. Create policy 𝑎𝑐𝑡𝑖𝑜𝑛_𝑈=𝑎𝑟𝑔𝑚𝑎𝑥_(𝑥∈{𝑎𝑛𝑠𝑤𝑒𝑟,𝑐𝑙𝑎𝑟𝑖𝑓𝑦,𝑎𝑏𝑠𝑡𝑎𝑖𝑛}) 𝐸𝑈_(𝑥,𝑈) (exact formula for Expected Utility can be found in presentation and code)
6. Evaluate policy on test set

# Results
Some of the results can be found in the presentation and all of the results can be found in the folder results. Each folder for each LLM contains three results for the three different policy models (files are named are named after the used model: Ridge Regression -> Linear, Neural Network -> MLP, Gradient Boosted Trees -> LightGBM). Each result contains the following files:
1. Specific model saved as .joblib or .pt model. For the Ridge Regression and Gradient Boosted Trees two separate models are learned for 𝑝(𝑐𝑜𝑟𝑟𝑒𝑐𝑡│𝑈) and 𝑝(𝑐𝑜𝑟𝑟𝑒𝑐𝑡_𝑐𝑙𝑎𝑟𝑖𝑓𝑖𝑒𝑑|𝑈), while for the Neural Network a single model outputting both probabilities is learned
2. summary.xlsx contains the main summary results: Contains the results for the learned policy, a policy using always answer, a policy using always clarify, a policy using random clarifications with the same clarification rate as the learned policy in a dictionary each. Then the statistical test whether the learned policy performs better than the randomized policy on all samples and then on only the answered samples follows. Finally the average amount of errors prevented per clarification for the learned policy and then for the randomized policy follow. Each of the dictionaries for the different policies contains the following information: The number of test samples, the rate of answers, the rate of clarifications, the accuracy on the answered samples, the accuracy on all samples (counting abstention as wrong), the F1-Score on the answered samples and the F1-Score on all samples
4. action_distr.png depicts the rate of different actions given the uncertainty level (We always calculate the uncertainty level as mean of the rescaled uncertainty signals)
5. cost_tradeoff.png depicts the effect of different clarification costs on the Expected Utility of the policy and on the rate of clarifications
6. eu_policy.png depicts the Expected Utility of the three actions (answer, clarify, abstain) for different uncertainty levels
7. eu_vs_u.png shows the Expected Utility of answering for every single test sample plotted against its uncertainty
8. policy_comparison.png depicts the Expected Utility of the learned policy, always clarifying and randomized clarifications for different uncertainty levels
9. uncertainty_cal.png shows how well the learned policy model fits the empirical 𝑝(𝑐𝑜𝑟𝑟𝑒𝑐𝑡│𝑈) and 𝑝(𝑐𝑜𝑟𝑟𝑒𝑐𝑡_𝑐𝑙𝑎𝑟𝑖𝑓𝑖𝑒𝑑|𝑈) for different uncertainty levels

Additionally the results contain eval.xlsx containing specific data for each single sample of the test set such as the claim, answer, sampled answers, uncertainty signals, Expected Utility, etc. However those files are too large to upload to this repository. Running this experiment will however create those files.

# Analysis of Results
