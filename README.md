# POS-tagger
Viterbi HMM part of speech tagger using NLTK

Run using python3 ViterbiPOS.py trainingfile testfile

Only handles input in the form of the WSJ corpus, example training and test file in the repository

Methods for handling out of vocabulary words based on simple linguistic morphology.

End and beginning of sentence kept track of to improve the bigram models accuracy

~93% accuracy on average

python3 score.py keyfile answerfile to test score
