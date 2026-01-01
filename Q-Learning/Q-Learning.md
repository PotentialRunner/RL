# Q-Learning

## 1. Prerequisites

### 1.1 Value-based methods and Policy-based methods

**value-based** methods compute the expected discounted rewards corresponding to a given state via a value function (value-function).

<img title="" src="./images/value-based%20method.png" alt="Formula for the value-based method" data-align="inline">

By finding the optimal value function $Q^*$ or $V^*$, we are in fact indirectly finding an optimal policy $\pi^*$. The meaning of the formula here is: given a state–action pair $(s,a)$, each time we choose the action $a$ that, according to the value function, leads to the state with the largest value.

**policy-based** methods directly train an optimal policy $\pi^*$, so that given a state $s$ as input, the policy outputs the action $a$ we should take.

> **We can notice a problem: whether it is a value-based method or a policy-based method, both require a policy $\pi$. One finds the optimal policy directly, while the other finds the optimal value function first and then indirectly obtains the optimal policy $\pi$.**

### 1.2 Exploration/Exploitation

**exploration** means the **agent** will try some new actions that it has never executed in its past experience, in order to explore a new environment. This behavior can help the **agent** discover larger rewards, but it also means taking on a certain amount of risk.

**exploitation** means the **agent** executes familiar actions based on existing experience—actions that are believed to quickly bring **reward**. However, doing so may miss the possibility of exploring new environments, and the efficiency may be lower than the former.

> In practice, during training we need to make a **trade-off** between the two. We should not be stubbornly biased to one side; instead, we must weigh them flexibly to obtain the optimal result.

## 2. Two **value-based** functions

### 2.1 The state-value function (State-value function)

First, let's look at the formula of the state-value function:

![State-value function formula](./images/state%20value%20function.png)

Here, **$V_\pi(s)$** represents the value when starting from state $s$ and continuing to follow policy $\pi$ thereafter. This value is an expectation.

### 2.2 The action-value function (Action-value function)

The formula of the action-value function is a bit different:

![Action-value function formula](./images/action%20value%20function.png)

Unlike the state-value function, the action-value function is not the value of a state $s$, but the value of a $(s,a)$ state–action pair. That is, in a state $s$, if we take an action $a$, what is the corresponding value? Likewise, this value is an expectation.

> After comparing the two, we can see that **state value function** computes the value corresponding to a state, while **action value function** computes the value corresponding to a state–action pair. This is the difference, and the similarity is that both of their values are expectations.
> 
> Also note that if we implement these two methods in the straightforward way—i.e., filling in every state with the value it should have—this would mean we need to start from every state and ... which is extremely time-consuming and unrealistic. At this point, there is a very useful mathematical tool to help us handle this problem — the **bellman equation**.

## 3. The Bellman Equation (Bellman equation)
