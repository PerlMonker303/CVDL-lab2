import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    Source of inspiration: https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
    """
    # x = [values from model]
    # subtract the maximum value from each value, call that xsub
    xsub = x - np.max(x)
    # apply the temperature given in the parameters
    xsub = np.divide(xsub, t)
    # now compute the exponential of all elements in that array
    exps = np.exp(xsub)
    # compute the sum of exponentials (there is no point in doing that at each step)
    expsum = np.sum(exps)
    # now divide every exponential by that sum and return the resulting vector
    return exps / expsum
