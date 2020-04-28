import pickle
import json
from tqdm import tqdm
import configparser
from scipy import spatial
import numpy as np
import os
from os import sys, path
import random
from pathlib import Path

CONCEPT_EMBEDDING_PATH = "embeddings/concept_glove.max.npy"
RELATION_EMBEDDING_PATH = "embeddings/relation_glove.max.npy"

class ScorePaths(object):
    def __init__(self, config_path, concept_embedding_fn=None, relation_embedding_fn=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.root= Path(config_path).parent.parent
        self.concept_fn = self.root/ CONCEPT_EMBEDDING_PATH if concept_embedding_fn is None else concept_embedding_fn
        self.relation_fn = self.root/ RELATION_EMBEDDING_PATH if relation_embedding_fn is None else relation_embedding_fn
        self.load_resources()

    def load_resources(self,):
        self.concept2id = {}
        self.id2concept = {}

        with open(self.config["paths"]["concept_vocab"][3:], "r", encoding="utf8") as f:
            for w in f.readlines():
                self.concept2id[w.strip()] = len(self.concept2id)
                self.id2concept[len(self.id2concept)] = w.strip()

        self.concept_embs = np.load(self.concept_fn)
        self.relation_embs = np.load(self.relation_fn) 

    def score_triple(self, h, t, r, flag):
        res = -10

        for i in range(len(r)):
            temp_h, temp_t = (t, h) if flag[i] else (h, t)
            res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)
            
        return res

    def score_triples(self, concept_id, relation_id, debug=False):
        concept = self.concept_embs[concept_id]
        relation = []
        flag = []

        for i in range(len(relation_id)):
            embs = []
            l_flag = []

            if 0 in relation_id[i] and 17 not in relation_id[i]:
                relation_id[i].append(17)
            elif 17 in relation_id[i] and 0 not in relation_id[i]:
                relation_id[i].append(0)
            if 15 in relation_id[i] and 32 not in relation_id[i]:
                relation_id[i].append(32)
            elif 32 in relation_id[i] and 15 not in relation_id[i]:
                relation_id[i].append(15)

            for j in range(len(relation_id[i])):
                if relation_id[i][j] >= 17:
                    embs.append(self.relation_embs[relation_id[i][j] - 17])
                    l_flag.append(1)
                else:
                    embs.append(self.relation_embs[relation_id[i][j]])
                    l_flag.append(0)

            relation.append(embs)
            flag.append(l_flag)

        res = 1

        for i in range(concept.shape[0] - 1):
            h = concept[i]
            t = concept[i + 1]
            score = self.score_triple(h, t, relation[i], flag[i])
            res *= score

        if debug:
            print("Num of concepts:")
            print(len(concept_id))
            to_print = ""

            for i in range(concept.shape[0] - 1):
                h = self.id2concept[concept_id[i]]
                to_print += h + "\t"
                
                for rel in relation_id[i]:
                    if rel >= 17:
                        # 'r-' means reverse
                        to_print += ("r-" + self.id2relation[rel - 17] + "/  ")
                    else:
                        to_print += self.id2relation[rel] + "/  "

            to_print += self.id2concept[concept_id[-1]]
            print(to_print)
            print("Likelihood: " + str(res) + "\n")

        return res

    def path_scoring(self, path, context):
        path_concepts = self.concept_embs[path]

        # cosine distance, the smaller the more alike
        cosine_dist = np.apply_along_axis(spatial.distance.cosine, 1, path_concepts, context)
        cosine_sim = 1 - cosine_dist
        if len(path) > 2:
            return min(cosine_sim[1:-1]) # the minimum of the cos sim of the middle concepts
        else:
            return 1.0 # the source and target of the paths are qa concepts

    def context_per_qa(self, acs, qcs, pooling="mean"):
        '''
        calculate the context embedding for each q-a statement in terms of mentioned concepts
        '''
        for i in range(len(acs)):
            acs[i] = self.concept2id[acs[i]]

        for i in range(len(qcs)):
            qcs[i] = self.concept2id[qcs[i]]

        concept_ids = np.asarray(list(set(qcs) | set(acs)), dtype=int)
        concept_context_emb = np.mean(self.concept_embs[concept_ids], axis=0) if pooling=="mean" else np.maximum(self.concept_embs[concept_ids])

        return concept_context_emb

    def calc_context_emb(self, pooling="mean", filename =""):
        filename = Path(filename)
        fn_embedded_concepts = filename.parent / f'{filename.stem}.{pooling}.npy'

        if Path(fn_embedded_concepts).is_file():
            print(f'{fn_embedded_concepts} file exists! Loading.')
            embs = np.load(fn_embedded_concepts)
            return embs
        
        with open(filename, "rb") as f:
            concepts_data = json.load(f)

        embs = [self.context_per_qa(acs=s["ac"], qcs=s["qc"], pooling=pooling) for s in tqdm(concepts_data, desc='Computing concept-context embedding')]
        embs = np.asarray(embs)
        np.save(fn_embedded_concepts, embs)

        return embs
    
    def prune(self, scores, paths, threshold):
        assert len(paths) == len(scores)

        ori_len = 0
        pruned_len = 0
        for index, qa_pairs in tqdm(enumerate(paths[:]), desc="Pruning the paths", total=len(paths)):
            for qa_idx, qas in enumerate(qa_pairs):
                _paths = qas["paths"]
                if _paths is not None:
                    pruned_paths = []
                    for pf_idx, item in enumerate(_paths):
                        score = scores[index][qa_idx][pf_idx]
                        if score >= threshold:
                            pruned_paths.append(item)
                    ori_len += len(paths[index][qa_idx]["paths"])
                    pruned_len += len(pruned_paths)
                    assert len(paths[index][qa_idx]["paths"]) >= len(pruned_paths)
                    paths[index][qa_idx]["paths"] = pruned_paths

        print(f"Num paths: {ori_len}, \tAfter Pruning: {pruned_len}, keep rate: {pruned_len / ori_len:.4}")

        return paths 

    def process(self, paths_fn, concepts_fn, debug=False, prune_threshold=0.15):
        """
        concepts_fn:    file generated in grounding_concepts.py
        path_fn:        file generated by pathfinder.py
        """

        all_scores = []
        self.context_embeddings = self.calc_context_emb(filename=concepts_fn)

        with open(paths_fn, "rb") as f:
            paths_data = pickle.load(f)
 
        for index, qa_pairs in tqdm(enumerate(paths_data), desc="Scoring the paths", total=len(paths_data)):
            scores = []

            for qa_idx, qas in enumerate(qa_pairs):
                paths = qas["paths"]

                if paths is not None:
                    context_emb = self.context_embeddings[index]
                    path_scores = []
                    
                    for pf_idx, item in enumerate(paths):
                        assert len(item["path"]) > 1
                        score = self.score_triples(concept_id=item["path"], relation_id=item["rel"], debug=debug)
                        path_scores.append(score)
                    scores.append(path_scores)
                else:
                    scores.append(None)
            all_scores.append(scores)

        out_fn = Path(paths_fn).parent / f'{Path(paths_fn).stem}_scores.pickle'
        
        with open(out_fn, 'wb') as fp:
            pickle.dump(all_scores, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Scores saved to {out_fn}')

        # prune
        paths = self.prune(all_scores, paths_data, threshold=prune_threshold) 
        out_fn = Path(paths_fn).parent / f'{Path(paths_fn).stem}_scores_pruned.pickle'

        with open(out_fn, 'wb') as fp:
            pickle.dump(paths, fp, protocol=pickle.HIGHEST_PROTOCOL)
