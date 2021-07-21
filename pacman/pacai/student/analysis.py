"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I changed the noise val to zero
    so the agent will always go in the desired direction
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    Penalize the agent heavily for being alive
    leads to it risking the short path to the close exit
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -2

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Lower discount to incentivize taking a safer path,
    keep the negative living reward from a to prefer the close exit
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -2

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    noice=0 removes risk of hitting penalty,
    living reward of -1 prioritizes close path
    """

    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    The default params do this
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Large living reward rewards more than exiting
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 10

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    The agent does not have enough iterations to properly explore
    the whole board and discover the optimal center path policy
    """

    # answerEpsilon = 0.2
    # answerLearningRate = 0.5

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
