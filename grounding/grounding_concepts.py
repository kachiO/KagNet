import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
import re 
from pathlib import Path
import jsbeautifier

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])

def camel_case_split(str): 
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str) 

class GroundConcepts(object):
    """
    Extract and match concepts from Ai2Thor objects to ConceptNet concepts.
    """
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.root= Path(config_path).parent.parent
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

        with open(self.root/ self.config["paths"]["concept_vocab"][3:], "r", encoding="utf8") as f:
            self.concept_vocab = [l.strip() for l in list(f.readlines())]

        self.concept_vocab = [c.replace("_", " ") for c in self.concept_vocab]

    def lemmatize(self, concept):
        doc = self.nlp(concept.replace("_"," "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
        return lcs

    def load_matcher(self,):
        with open(self.root/ self.config["paths"]["matcher_patterns"][3:], "r", encoding="utf8") as f:
            all_patterns = json.load(f)

        matcher = Matcher(self.nlp.vocab)
        for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
            matcher.add(concept, None, pattern)
        return matcher

    def ground_mentioned_concepts(self, s, ans = ""):
        s = s.lower()
        doc = self.nlp(s)
        matches = self.matcher(doc)
        mentioned_concepts = set()
        span_to_concepts = {}

        for match_id, start, end in matches:
            span = doc[start:end].text  # the matched span
            if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
                continue
            original_concept = self.nlp.vocab.strings[match_id]

            if len(original_concept.split("_")) == 1:
                original_concept = list(self.lemmatize(original_concept))[0]

            if span not in span_to_concepts:
                span_to_concepts[span] = set()

            span_to_concepts[span].add(original_concept)

        for span, concepts in span_to_concepts.items():
            concepts_sorted = list(concepts)
            concepts_sorted.sort(key=len)
            shortest = concepts_sorted[0:3] #
            for c in shortest:
                if c in blacklist:
                    continue
                lcs = self.lemmatize(c)
                intersect = lcs.intersection(shortest)
                if len(intersect)>0:
                    mentioned_concepts.add(list(intersect)[0])
                else:
                    mentioned_concepts.add(c)

        return mentioned_concepts

    def hard_ground(self, sent):
        sent = sent.lower()
        doc = self.nlp(sent)
        res = set()

        for t in doc:
            if t.lemma_ in self.concept_vocab:
                res.add(t.lemma_)

        sent = "_".join([t.text for t in doc])

        if sent in self.concept_vocab:
            res.add(sent)
        return res

    def match_mentioned_concepts(self, sents, answers):
        self.matcher = self.load_matcher()
        res = []

        for sid, s in tqdm(enumerate(sents), total=len(sents), desc="Grounding"):
            a = answers[sid]
            all_concepts = self.ground_mentioned_concepts(s, a)
            answer_concepts = self.ground_mentioned_concepts(a)
            question_concepts = all_concepts - answer_concepts
            if len(question_concepts)==0:
                question_concepts = self.hard_ground(s) # not very possible
            if len(answer_concepts)==0:
                print(a)
                answer_concepts = self.hard_ground(a) # some case
                print(answer_concepts)

            res.append({"sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)})
        return res
    
    def prune_concepts(self, data):
        """
        Looks for stop words and removes concepts that are not in Conceptnet.
        Originally in prune_qc.py
        """
        
        import nltk

        nltk.download('stopwords')
        nltk_stopwords = nltk.corpus.stopwords.words('english')
        nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]
        
        pruned_data = []
        for item in tqdm(data, desc='Pruning concepts'):
            qc = item["qc"]
            prune_qc = []
            for c in qc:
                if c[-2:] == "er" and c[:-2] in qc:
                    continue
                if c[-1:] == "e" and c[:-1] in qc:
                    continue
                have_stop = False
                
                for t in c.split("_"):
                    if t in nltk_stopwords:
                        have_stop = True
                if not have_stop and c in self.concept_vocab:
                    prune_qc.append(c)

            ac = item["ac"]
            prune_ac = []
            for c in ac:
                if c[-2:] == "er" and c[:-2] in ac:
                    continue
                if c[-1:] == "e" and c[:-1] in ac:
                    continue
                all_stop = True
                for t in c.split("_"):
                    if t not in nltk_stopwords:
                        all_stop = False
                if not all_stop and c in self.concept_vocab:
                    prune_ac.append(c)

            item["qc"] = prune_qc
            item["ac"] = prune_ac

            pruned_data.append(item)

        return pruned_data
    
    def process(self, objects_list, add_rooms=True, exclude_self=False, outfilename='ai2thor_objects_to_objects'):
        with open(objects_list, 'r') as f: 
            lines = f.read().split('\n')
    
        objects = [line for line in lines if not line == '']
        objects = [' '.join(np.unique([o] + camel_case_split(o))) for o in objects]
        objects_all = objects
        name = ''
        
        if add_rooms:
            objects_all += ['Kitchen', 'Bedroom', 'Bathroom', 'LivingRoom']
            name += '_rooms'
            #print('Added rooms')
            
        if exclude_self:
            _objects_all = [' '.join(objects_all[:ii] + objects_all[ii+1:]) for ii, oo in enumerate(objects)]
            name += '_exclude_self'
        else:
            _objects_all = [' '.join(objects_all) for ii, oo in enumerate(objects)]
            name += '_include_self'

        output_path = self.root/ 'datasets/ai2thor'
        output_path.mkdir(exist_ok=True, parents=True) 

        res = self.match_mentioned_concepts(sents=objects, answers=_objects_all)
        res = self.prune_concepts(res)

        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        outfilename += name
        out_fn  = str(output_path / f'{outfilename}_concepts.json')

        with open(out_fn, 'w') as fp:
            fp.write(jsbeautifier.beautify(json.dumps(res), opts))
        print(f'Saved to {out_fn}')


