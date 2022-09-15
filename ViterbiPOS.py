import sys
import nltk
from collections import defaultdict

#OOV processing function. 
def oovProcess(word, prevpos):
    #This will imitate the viterbi for a single word and is what we will return
    oovarray = [0] * len(poslist)
    for key in oovpos.items():
        oovarray[posToNumber[key[0]]] = key[1]

    #These are the actual oov calculations

    #These are very strong indicators of pos
    if (len(word) > 1 and not word.isalpha()):
        oovarray[posToNumber['CD']] = 1
        return oovarray
    if (len(word) == 1 and word.isalpha()):
        oovarray[posToNumber['SYM']] = 1
        return oovarray
    if (word[0:3] == 'anti'):
        oovarray[posToNumber['JJ']] = .9
        return oovarray

    #Suffixes attached to specific pos
    if ((word[-4:] == ('less' or 'like' or 'able')) and (prevpos == ('DT' or 'JJ'))):
        oovarray[posToNumber['JJ']] = .95
    elif ((word[-3:] == 'ous') and (prevpos == 'DT' or 'JJ')):
        oovarray[posToNumber['JJ']] = .95
    elif (word[-2:] == 'er' and (prevpos == 'DT' or 'JJ')):
        oovarray[posToNumber['JJR']] = .95
    elif (word[-2:] == 'ed' and not prevpos == 'DT'):
        oovarray[posToNumber['VBD']] = .9
    elif (word[-3:] == ('ity' or 'ist')):
        oovarray[posToNumber['NN']] = .7
    elif (word[-4:] == ('ship' or 'tion' or 'sion' or 'ence')):
        oovarray[posToNumber['NN']] = .75
    elif (word[-3:] == 'ing' and not prevpos == 'DT'):
        oovarray[posToNumber['VBG']] = .6
    elif (word[-1:] == 's' and prevpos == 'DT'):
        oovarray[posToNumber['NNS']] = .7
    elif (word[-2:] == 'es'):
        oovarray[posToNumber['VBZ']] = .45
    elif (word[-1] == 's'):
        oovarray[posToNumber['VBZ']] = .3

    #prefixes attached to specific POS.
    if (word[0:1] == ('il' or 'ir' or 'im' or 'in' or 'en')):
        oovarray[posToNumber['JJ']] += .4
    elif (word[0:2] == 'non'):
        oovarray[posToNumber['JJ']] += .1
        oovarray[oovarray['NN']] += .1
        oovarray[posToNumber['NNS']] += .1

        
    return oovarray
    

# Read file input
with open(sys.argv[1], 'r') as f:
    contents = f.read()

with open(sys.argv[2], 'r') as fi:
    testdata = fi.read()

file = open('taggedwords.pos', 'w')


##Set variables for training
likelyhood = defaultdict(lambda: defaultdict(int))
pos = defaultdict(lambda: defaultdict(int))
words = defaultdict(int)
oov = defaultdict(bool)
oovpos = defaultdict(int)
total = 0
processedtext = [('begin_line', 'B')]
contents = contents.lower()

##Process input so it can be put into .bigram
for line in contents.split('\n'):
    if (line == ''):
        processedtext.append(('end_line', 'E'))
        processedtext.append(('begin_line', 'B'))
    else:
        processedtext.append((line.split('\t')[0], line.split('\t')[1]))


##THIS IS THE TRAINING PHASE OF THE PROGRAM 

##Make pos dict
for ((w1, t1), (w2, t2)) in nltk.bigrams(processedtext):
    pos[t1.upper()][t2.upper()] += 1

##Make likelyhood dict and word list
for ((w1, t1), (w2, t2)) in nltk.bigrams(processedtext):
    likelyhood[t1.upper()][w1] += 1
    words[w1] += 1

##Sum words, move words with only 1 count to oov list
for item in words.items():
    total += item[1]
    if item[1] == 1:
        oov[item[0]] = True


#Make oov POS table
for ((w1, t1), (w2, t2)) in nltk.bigrams(processedtext):
    if oov[w1] == True:
        oovpos[t1.upper()] += 1

##Using this to help OOV classification, will already check for numbers and don't want them to be counted here
oovpos['CD'] = 0

##Sum oovpos
total = 0
for item in oovpos.items():
    total += item[1]
#Change values to probabilities
for item in oovpos.items():
    oovpos[item[0]] = item[1]/total

##Reset total, then sum words, then change counts to probabilities
for table in likelyhood.items():
    total = 0
    for item in table[1].items():
        total += item[1]

    for item in table[1].items():
        likelyhood[table[0]][item[0]] = item[1]/total

##Reset total, then sum POS, then change counts to probabilities
for table in pos.items():
    total = 0
    for item in table[1].items():
        total += item[1]
    for item in table[1].items():
        pos[table[0]][item[0]] = item[1]/total


def processoov(word):
    return 0 



##THIS IS THE TESTING PHASE OF THE PROGRAM

##Init variables for testing
wordlist = ['begin_line']
total = 0

#numbered list of parts of speech
poslist = []
for key in pos.items():
    poslist.append(key[0])


#Parts of speech to their corresponding number in poslist
posToNumber = {}
for posp in poslist:
    posToNumber[posp] = total
    total += 1

viterbi = [[0]*len(poslist) for i in range(120)]
oovarr = [0]*len(poslist) 
prevpos = ''
hold = 0.0
topscore = 0.0
maxpos = 0
finalpairs = []

#Make array of words and add begin and end tokens
for line in testdata.split('\n'):
    if line == '':
        if wordlist[-1] == 'begin_line':
            continue
        wordlist.append('end_line')
        wordlist.append('begin_line')
    else:
        wordlist.append(line)

#Start viterbi algorithm
total = 0
for word in wordlist:

    finalpairs.append(word)
    #Set 0, B to 1
    if total == 0: 
        viterbi[0][0] = 1
        finalpairs[-1] = finalpairs[-1] + '\tB'

    elif words[word.lower()] < 2: 
        #Lock in the previous POS here.
        for i in range(len(poslist)):
            hold = viterbi[total - 1][i]
            if (hold > topscore):
                topscore = hold
                maxpos = i
        finalpairs[-2] = finalpairs[-2] + '\t' + poslist[maxpos]
        oovarr = oovProcess(word, finalpairs[-2].split('\t')[1])
        for i in range(47):
            viterbi[total][i] = oovarr[i]
    

    #End sentence calculations
    elif word == 'end_line':
        #Lock in previous POS
        for i in range(len(poslist)):
            hold = viterbi[total - 1][i]
            if (hold > topscore):
                topscore = hold
                maxpos = i
        finalpairs[total - 1] = finalpairs[total - 1] + '\t' + poslist[maxpos]

        #Don't print b or e
        for pair in finalpairs:
            if (pair[:10] == 'begin_line' or pair == 'end_line'):
                continue
            file.write(pair + '\n')
        finalpairs.clear()
        file.write('\n')
        total = -1
        #Write the finalpairs to file then clear. This will prevent the array from becoming massive


    #Calculate B to state 1 prob
    elif total == 1:
        for i in range(len(poslist)):
            viterbi[1][i] = pos['B'][poslist[i]] * likelyhood[poslist[i]][word.lower()]

    #Bigram calculations
    else:
        for i in range(len(poslist)):
            #Go through to find the best previous part of speech
            for j in range(len(poslist)):
                hold = viterbi[total - 1][j] * pos[poslist[j]][poslist[i]] * likelyhood[poslist[i]][word.lower()]
                if (hold > topscore):
                    topscore = hold
                    maxpos = j

        #Set current viterbi's off previous best POS
        for i in range(len(poslist)):
            viterbi[total][i] = pos[poslist[maxpos]][poslist[i]] * likelyhood[poslist[i]][word.lower()]

        #put the word and pos in what we will end up printing out
        finalpairs[total - 1] = finalpairs[total - 1] + "\t" + poslist[maxpos]
    total += 1
    maxpos = 0
    topscore = 0
    hold = 0

file.close()
