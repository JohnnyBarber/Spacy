import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

doc = 'Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model of sample data, known as “training data”, in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning and focuses on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.'
# model location: D:\PythonPractice\3.4.0\dist\en_core_web_sm-3.4.0
# use $pip install D:\PythonPractice\3.4.0\dist\en_core_web_sm-3.4.0


def test(doc):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(doc)
    keywords = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PRON', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            keywords.append(token.text)

    freq_word = Counter(keywords)

    max_freq = Counter(keywords).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = freq_word[word]/max_freq
    freq_word.most_common(5)

    sentence_strength = {}
    for sentence in doc.sents:
        for word in sentence:
            if word.text in freq_word.keys():
                if sentence in sentence_strength.keys():
                    sentence_strength[sentence] += freq_word[word.text]
                else:
                    sentence_strength[sentence] = freq_word[word.text]

    summary = nlargest(3, sentence_strength, key=sentence_strength.get)
    return summary[0]


if __name__ == '__main__':
    test = test(doc)
    print(test)
