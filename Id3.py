import math

def entropy(S):
    result = 0
    for p in S:
        result += -(p*math.log2(p))
    return result

def gain(S,A):
    sum = 0
    for value in A:
        sum += abs(value.count)/abs(S)* entropy(value.count)
    return entropy(S) - sum

def ID3(examples, target_attribute, attributes):
    pass