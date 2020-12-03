import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    # separa uma sentença em uma array de palavras ["olá", "tudo", "bem", "?"]
    # um token pode ser uma palavra, pontuação, número

    return nltk.word_tokenize(sentence)


def stem(word):
    # stemização: forma raiz de uma palavra
    # examplo:
    # palvaras = ["organizar", "organize", "organizando"]
    # palavras = [stem(w) for w in words]
    # -> ["organ", "organ", "organ"]

    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    
    # retorna a array de bag of words:
    # 1 para cada palavra que exista na sentença, 0 caso contrário
    # exemplo: 
    # frase =    ["olá", "como", "vai", "você"]
    # palavras = ["oi",  "olá",  "eu",  "como", "tchau", "vai", "legal"]
    # bag   =    [  0 ,    1 ,    0 ,    1 ,      0 ,      1 ,      0]

    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag