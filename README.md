# Problem Statement

Recent news and METR reports have shown a propensity for LLM agents to cheat when attempting to solve goal-oriented tasks.

Practical Question: is there a way of prompting the agents in such a way that will lead to them not cheating?

Deeper Question: our technical culture has a bias towards problem-solving, goal-oriented thinking, over looking at the bigger picture or greater good. Does the propensity to cheat arise from the way the agents were prompted, or a deeper bias in the training data corpus? 

If the bias is shallow (i.e. in the prompt), then changes to the system prompt should reduce incidence of cheating. If, however, the bias is deep (i.e. in the training data) then the incidence of cheating should be immune to changes in the system prompt. 

Deep bias may also be expected to vary more betwen different models than shallow bias, although testing shallow bias on multiple models may also be instructive.

