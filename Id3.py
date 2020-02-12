import math
import pandas as pd

'''
I.S	: Seluruh dataset (S), label yang akan dihitung (LABEL), list data yang dihitung(L)
F.S	: floating point ENTROPY dari data yang dihitung
'''
def entropy(S, label, L = None):

	distinctValue = list(set(S[label]))
	result = 0.0
	newS = S

	if (L != None):
		newS = pd.DataFrame(S.columns)
		newS = S.iloc[L, :]

	for i in distinctValue:
		p = countValue(newS, label, i)/len(newS)
		result += p * math.log2(p)

	return -1 * result

'''
I.S	: Seluruh dataset (S), hitungan dengan acuan pada atribut tertentu (ATTRIBUTE), dengan nilai tertentu (VALUE)
F.S	: banyak instance dengan ATTRIBUTE dengan nilai VALUE
'''

def countValue(S, attribute, value):
	count = 0
	totalInstance = len(S)
	for i in range(0, totalInstance, 1):
		if (S[attribute].values[i] == value):
			count += 1
	return count

def gain(S, L, A):

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

dataTrain = pd.read_csv('play_tennis.csv')
print(dataTrain)
print(entropy(dataTrain, 'play'))
