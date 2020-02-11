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
    #untuk semua examples, check positif semua atau gimana, kalo iya langsung return labelnya
    #kalau atributnya kosong, return label dengan target_attribut mayoritas

    #kalau engga : hitung entropy->hitung entropy tiap attribute, hitung gain
    #dari gain, tentuin leaf node nya, kalau example kosong, return target_attribut mayoritas, kalau masih ada, rekursif panggil ID3 lagi
    pass