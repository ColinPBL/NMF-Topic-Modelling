# Standard libraries
import os
import time
import gc
import re
import string
import zipfile
from urllib import request
from collections import Counter

# Numpy and Scipy
import numpy as np
from scipy import stats

# Natural language processing and text processing
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from cleantext import clean
import pyate
from pyate import combo_basic, cvalues
from pyate.term_extraction_pipeline import TermExtractionPipeline

# Miscellaneous
import unidecode
from bs4 import BeautifulSoup
from sklearn.cluster import DBSCAN
from numba import jit
import logging

logging.basicConfig(level=logging.WARNING)


def parallel_treatment(document, nlp_transformer):
    return treat_this_document(document, nlp_transformer)


"""
This is the main process used to lemmatise our documents. It checks the input queue periodically
for a document, then it uses a transformer-based model to process it.
"""
def worker(input_queue, output_queue, language, init_args):
    # Load the appropriate model based on the language
    if language == 'fr':
        nlp_transformer = spacy.load('fr_dep_news_trf')
    elif language == 'en':
        nlp_transformer = spacy.load('en_core_web_trf')
    elif language == 'es':
        nlp_transformer = spacy.load('es_dep_news_trf')
    elif language == 'de':
        nlp_transformer = spacy.load('de_dep_news_trf')
    elif language == 'ca':
        nlp_transformer = spacy.load('ca_core_news_trf')
    elif language == 'zh':
        nlp_transformer = spacy.load('zh_core_web_trf')
    elif language == 'da':
        nlp_transformer = spacy.load('da_core_news_trf')
    elif language == 'ja':
        nlp_transformer = spacy.load('ja_core_news_trf')
    elif language == 'sl':
        nlp_transformer = spacy.load('sl_core_news_trf')
    elif language == 'uk':
        nlp_transformer = spacy.load('uk_core_news_trf')

    # Run until the main process indicates all documents have been treated
    while True:
        task = input_queue.get()
        if task == "STOP":
            break
        try:
            idx, doc = task  # Récupérer l'index et le document

            # Traiter le document
            result = parallel_treatment(document=doc, nlp_transformer=nlp_transformer)

            # Résultat sous forme de tuple (idx, doc_for_ngrams, tab_pos, these_sentences_norms, these_sentences_lemmas)
            output_queue.put((idx,) + result)
        except Exception as e:
            logging.info(f'worker crashed with {e}')


"""
Treat text to remove unwanted blank spaces and to ensure correct lemmatisation
"""
def transform_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

   # text = re.sub(r'[-–—‑‒−]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # écriture inclusive
    text = text.replace('(e)', '')
    text = text.replace('(E)', '')
    text = text.replace('.e.', '')
    text = text.replace('.E.', '')

    return text


"""
Translates POS tags from english to french
"""
def transform_tag(tag):
    tag = tag.lower()
    if tag == 'adp' or tag == 'p' or (len(tag) > 1 and tag[0] == 'p' and tag[1] == '+') or tag == 'prep':
        tag = 'prep'
    elif tag == 'noun' or tag == 'nc':
        tag = 'nc'
    elif tag == 'propn' or tag == 'npp':
        tag = 'np'
    elif tag[0]== 'v':
        tag = 'v'
    elif tag == 'aux':
        tag = 'aux'
    elif tag == 'adv':
        tag = 'adv'
    elif tag == 'det':
        tag = 'det'
    elif tag == 'adj':
        tag = 'adj'
    elif tag == 'pron':
        tag = 'pron'
    elif tag == 'punct' or tag == 'ponct':
        tag = 'poncts'
    elif tag == 'cc' or tag == 'cconj':
        tag = 'coo'
    #  else:
    #   tag = 'other'

    return tag





def treat_this_document(text, nlp_transformer):
    start_time = time.time()

    doc_for_ngrams = ''
    tab_pos = []
    these_sentences_norms = []
    these_sentences_lemmas = []

    try:
        nlp_obj = nlp_transformer(text)

        for sent in nlp_obj.sents:
            norms = []
            lemmes = []
            
            for token in sent:
                pos = transform_tag(token.pos_)
                lemma = token.lemma_.lower()

                if pos == 'np':
                    lemma = unidecode.unidecode(lemma)

                if lemma not in [" ", "\n", "\t"]:
                    doc_for_ngrams += lemma + ' '
                    tab_pos.append([lemma, pos])
                    lemmes.append(lemma)
                    norms.append(token.norm_)

            these_sentences_norms.append(" ".join(norms))
            these_sentences_lemmas.append(" ".join(lemmes))

    except Exception as e:
        logging.info(f'treat_this_document crashed with {e}')
    runtime = time.time() - start_time
    logging.info(f"")
    return (doc_for_ngrams,
        tab_pos,
        these_sentences_norms, 
        these_sentences_lemmas)





def tokens_are_similar(tokens1, tokens2, threshold=10):
    differences = 0
    for t1, t2 in zip(tokens1, tokens2):
        if t1 != t2:
            differences += 1
            if differences > threshold:
                return False
    return True




def check_doublons(args):
    pairs, shared_indices = args
    for (i, tokens1), (j, tokens2) in pairs:
        if tokens_are_similar(tokens1, tokens2):
            shared_indices.append(max(i, j))




def tokenize_and_stem(args):
    atb, unigrams = args
    tokenized_sents = []
    for t in atb:
        if t[0] in unigrams:
            tokenized_sents.append(t[0])

    return tokenized_sents



