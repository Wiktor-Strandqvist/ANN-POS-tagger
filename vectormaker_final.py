# -*- coding: utf-8 -*-
import gensim
import codecs
import xml.etree.ElementTree as ET
from lxml import etree


def progress_printer(sentences, words, number):
        if (sentences % number) == 0:
                print "Processed", sentences, "sentences."
                print "Containing", words, "words."

def make_model(num_featuresv, min_word_countv, contextv, downsamplingv=1e-3):
        num_features = num_featuresv
        min_word_count = min_word_countv
        num_workers = 4
        context = contextv
        downsampling = downsamplingv
        modelname = str(num_features)+'features_'+str(min_word_count)+'minwords_'+str(context)+'context'

        sentences = []
        word_counter = 0
        sentence_counter = 0
        sentence = []
        checker = 20000

        print "processing trainingset..."
        data = codecs.open('suc-train.txt', 'r', 'utf-8')
        for line in data:
                word = line.split('\t')[0]
                if word == '\n':
                    sentences.append(sentence)
                    sentence_counter += 1
                    sentence = []
                    progress_printer(sentence_counter, word_counter, checker)
                else:
                    sentence.append(word)
                    word_counter += 1

        print "processing testset..."
        data = codecs.open('suc-test.txt', 'r', 'utf-8')
        for line in data:
                word = line.split('\t')[0]
                if word == '\n':
                    sentences.append(sentence)
                    sentence_counter += 1
                    sentence = []
                    progress_printer(sentence_counter, word_counter, checker)
                else:
                    sentence.append(word)
                    word_counter += 1


        print 'total words:', word_counter
        print 'total sentences:', sentence_counter
        print 'making model...'
        model = gensim.models.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
        model.save(modelname)
        print 'success!'

def create_file():
        sent = 0
        vocab = codecs.open('vocab.txt', 'w', 'utf-8')

        sentence = ""
        print "processing test..."
        data = codecs.open('suc-test.txt', 'r', 'utf-8')
        for line in data:
                word = line.split('\t')[0]
                if word == '\n':
                    vocab.write(sentence + "\n")
                    sent += 1
                    sentence = ""
                else:
                    sentence += (word + " ")
        data.close()

        print "processing train..."
        data = codecs.open('suc-train.txt', 'r', 'utf-8')
        for line in data:
                word = line.split('\t')[0]
                if word == '\n':
                    vocab.write(sentence + "\n")
                    sent += 1
                    sentence = ""
                else:
                    sentence += (word + " ")
        data.close()

        print "processing wikipedia..."
        context = etree.iterparse('wikipedia-sv.xml', tag='sentence' )
        fast_iter(context,process_element, vocab)

def fast_iter(context, func, filev, *args, **kwargs):
    vocab = filev
    sent = 0
    """
    http://lxml.de/parsing.html#modifying-the-tree
    Based on Liza Daly's fast_iter
    http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    See also http://effbot.org/zone/element-iterparse.htm
    """
    for event, elem in context:
        func(elem, *args, **kwargs)
        sentence = ""
        for word in elem:
            try:
                sentence += (word.text + " ")
            except:
                pass

        vocab.write(sentence + "\n")
        sent += 1
        if (sent % 10000) == 0:
            print sent
        # It's safe to call clear() here because no descendants will be
        # accessed
        elem.clear()
        # Also eliminate now-empty references from the root node to elem
        for ancestor in elem.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]
    del context


def model_maker(num_featuresv, min_word_countv, contextv, downsamplingv=1e-3):
        num_features = num_featuresv
        min_word_count = min_word_countv
        num_workers = 4
        context = contextv
        downsampling = downsamplingv
        modelname = str(num_features)+'features_'+str(min_word_count)+'minwords_'+str(context)+'context'
        sentences = []
        word_counter = 0
        sentence_counter = 0
        length = 0
        print "processing data..."
        counter = 0
        data = codecs.open('vocab.txt', 'r', 'utf-8')
        for line in data:
                sentence = line.split()
                sentences.append(sentence)
                length += len(sentence)
                counter += 1
                if counter % 3000000 == 0:
                        break
        print 'making model...'
        model = gensim.models.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
        print 'saving model...'
        model.save(modelname)
        print 'success!'
