from pathlib import Path
from lab1.parser import Parser
from collections import deque
import re
import math

BOLD_TAG = 'BOLD'
BLOCK_TAG = 'BL'

ANNOTATION_WORDS = 'annotation'
MAIN_TEXT_WORDS = 'main'

ANNOTATION_COEF = 3
MAIN_TEXT_COEF = 1

NON_ALPHABETIC = '[^a-zA-Z]'


class SearchSystem:
    def __init__(self):
        self.files = {}
        self.docs = {}
        self.all_words = set()
        self.tf = {}
        self.idf = {}
        self.tf_idf = {}
        self.doc_len = {}
        self.avg_len = 0

    def build_from_mml(self, path):
        mml_files_path = Path(path).glob('*.mml')
        parser = Parser()

        for f in mml_files_path:
            self.files[f.name] = parser.parse(f)

        self.prepare_words()
        self.calc_doc_len()
        self.init_dict(self.tf)
        self.init_dict(self.tf_idf)
        self.calc_tf()
        self.calc_idf()
        self.calc_tf_idf()

    def calc_doc_len(self):
        avg_len = 0

        for name, doc in self.docs.items():
            annotations = doc[ANNOTATION_WORDS]['words']
            main_text = doc[MAIN_TEXT_WORDS]['words']

            doc_len = len(annotations) + len(main_text)
            self.doc_len[name] = doc_len
            avg_len += doc_len

        self.avg_len = avg_len / len(self.docs)

    def search(self, query):
        relevance = dict()

        if query and query.strip():
            query_tf = self.calc_query_tf(query)
            query_tf_idf = self.calc_query_tf_idf(query_tf)
            relevance = self.calc_doc_relevance(query_tf_idf)

        return relevance

    def calc_query_tf(self, query):
        query_tf = {}

        query = query.lower().strip()
        terms = re.split(NON_ALPHABETIC, query)

        terms_len = len(terms)
        terms_set = set(terms)

        if not terms_len:
            return

        for term in terms_set:
            query_tf[term] = terms.count(term) / terms_len

        return query_tf

    def calc_query_tf_idf(self, query_tf):
        query_tf_idf = {}

        for word, tf in query_tf.items():
            if word in self.idf:
                query_tf_idf[word] = tf * self.idf[word]
            else:
                query_tf_idf[word] = 0

        return query_tf_idf

    def calc_doc_relevance(self, query_tf_idf):
        docs_rating = {}
        k1 = 2
        b = 0.75

        for name, doc in self.docs.items():
            doc_score = 0
            for term in query_tf_idf:
                if term in self.idf:
                    div = self.idf[term] * self.tf[name][term] * (k1 + 1)
                    delim = (self.tf[name][term] + k1 * (1 - b + b * self.doc_len[name] / self.avg_len))
                    doc_score += div / delim

            docs_rating[name] = doc_score

        return sorted(docs_rating.items(), key=lambda a: a[1], reverse=True)

    def similarity(self, v1, v2):
        return - (v1 - v2) ** 2

    def prepare_words(self):
        self.parse_mml()
        self.populate_all_words()

    def populate_all_words(self):
        for name, words in self.docs.items():
            annotations = set(words[ANNOTATION_WORDS]['words'])
            main_text = set(words[MAIN_TEXT_WORDS]['words'])

            for word in list(annotations) + list(main_text):
                self.all_words.add(word)

    def parse_mml(self):
        for name, mml in self.files.items():
            q = deque()
            q.append(mml)

            annotation_words = []
            main_text = []

            while True:
                if not q:
                    break

                node = q.pop()

                if node.tag:
                    if BOLD_TAG == node.tag.upper():
                        annotation_words.extend(self.get_words(node))
                    if BLOCK_TAG == node.tag.upper():
                        main_text.extend(self.get_words(node))

                children = node.children

                if children:
                    for c in reversed(children):
                        q.append(c)

            self.docs[name] = {
                ANNOTATION_WORDS: {
                    'coef': ANNOTATION_COEF,
                    'words': list(filter(None, annotation_words))
                },
                MAIN_TEXT_WORDS: {
                    'coef': MAIN_TEXT_COEF,
                    'words': list(filter(None, main_text))
                }
            }

    def calc_tf(self):
        for name, words in self.docs.items():
            annotations_part = words[ANNOTATION_WORDS]
            annotations = annotations_part['words']
            annotations_coef = annotations_part['coef']
            annotations_set = set(annotations)

            main_text_part = words[MAIN_TEXT_WORDS]
            main_text = main_text_part['words']
            main_text_coef = main_text_part['coef']
            main_text_set = set(main_text)

            doc_len = len(annotations) + len(main_text)

            for term in list(annotations_set) + list(main_text_set):
                self.tf[name][term] = 0

            for term in annotations_set:
                self.tf[name][term] += annotations_coef * annotations.count(term) / doc_len

            for term in main_text_set:
                self.tf[name][term] += main_text_coef * main_text.count(term) / doc_len

    def calc_idf(self):
        len_docs = len(self.tf)

        for word in self.all_words:
            occurrence_in_docs = 0

            for tf in self.tf.values():
                if tf[word]:
                    occurrence_in_docs += 1

            self.idf[word] = math.log(len_docs / occurrence_in_docs) if occurrence_in_docs else 0

    def calc_tf_idf(self):
        for name, words in self.tf.items():
            for word in words:
                self.tf_idf[name][word] = self.tf[name][word] * self.idf[word]

    def init_dict(self, d):
        for name in self.docs:
            d[name] = {}

            for word in self.all_words:
                d[name][word] = 0

    def get_words(self, node):
        ch = node.children
        words = []

        if ch and len(ch) > 0:
            v = ch[0].value.lower().strip()
            words = re.split(NON_ALPHABETIC, v)

        return words
