import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import pickle
import sys
from pathlib import Path

class GenGraph(object):
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.root = Path(config_path).parent.parent
        self.load_cpnet()

    def load_resources(self):
        self.concept2id, self.id2concept = {}, {}

        with open(self.root / self.config["paths"]["concept_vocab"][3:], "r", encoding="utf8") as f:
            for w in f.readlines():
                self.concept2id[w.strip()] = len(self.concept2id)
                self.id2concept[len(self.id2concept)] = w.strip() 
        
        self.relation2id, self.id2relation = {}, {}
        with open(self.root / self.config["paths"]["relation_vocab"][3:], "r", encoding="utf8") as f:
            for w in f.readlines():
                self.id2relation[len(self.id2relation)] = w.strip() 
                self.relation2id[w.strip()] = len(self.relation2id)
            
        with open(self.paths_fn, "rb") as fi:
            self.paths_data = pickle.load(fi)

        with open(self.concepts_fn, "r") as f:
            self.concept_data = json.load(f)

    def load_cpnet(self):
        self.cpnet = nx.read_gpickle(self.root / self.config["paths"]["conceptnet_en_graph"][3:])
        self.cpnet_simple = nx.Graph()

        for u, v, data in self.cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if self.cpnet_simple.has_edge(u, v):
                self.cpnet_simple[u][v]['weight'] += w
            else:
                self.cpnet_simple.add_edge(u, v, weight=w)

    def get_edge(self, src_concept, tgt_concept):
        rel_list = self.cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]["rel"] for item in rel_list]))

    def plain_graph_generation(self, qcs, acs, paths, rels):
        """
        Plain graph generation
        """
        graph = nx.Graph()

        for index, p in enumerate(paths):
            for c_index in range(len(p)-1):
                h = p[c_index]
                t = p[c_index+1]
                # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
                graph.add_edge(h,t, weight=1.0)

        for qc1, qc2 in list(itertools.combinations(qcs, 2)):
            if self.cpnet_simple.has_edge(qc1, qc2):
                graph.add_edge(qc1, qc2, weight=1.0)

        for ac1, ac2 in list(itertools.combinations(acs, 2)):
            if self.cpnet_simple.has_edge(ac1, ac2):
                graph.add_edge(ac1, ac2, weight=1.0)

        if len(qcs) == 0:
            qcs.append(-1)

        if len(acs) == 0:
            acs.append(-1)

        if len(paths) == 0:
            for qc in qcs:
                for ac in acs:
                    graph.add_edge(qc,ac, rel=-1, weight=0.1)

        g = nx.convert_node_labels_to_integers(graph, label_attribute='cid') # re-index
        g_str = json.dumps(nx.node_link_data(g))
        return g_str

    def relational_graph_generation(self, qcs, acs, paths, rels):
        """
        Relational graph generation, multiple edge types.
        """
        graph = nx.MultiDiGraph()

        for index, p in enumerate(paths):
            rel_list = rels[index]
            for c_index in range(len(p)-1):
                h = p[c_index]
                t = p[c_index+1]

                if graph.has_edge(h,t):
                    existing_r_set = set([graph[h][t][r]["rel"] for r in graph[h][t]])
                else:
                    existing_r_set = set()

                for r in rel_list[c_index]:
                    # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
                    # TODO: do we need to add both directions?
                    if r in existing_r_set:
                        continue
                    graph.add_edge(h,t, rel=r, weight=1.0)

        for qc1, qc2 in list(itertools.combinations(qcs, 2)):
            if self.cpnet_simple.has_edge(qc1, qc2):
                rs = self.get_edge(qc1, qc2)
                for r in rs:
                    graph.add_edge(qc1, qc2, rel=r, weight=1.0)

        for ac1, ac2 in list(itertools.combinations(acs, 2)):
            if self.cpnet_simple.has_edge(ac1, ac2):
                rs = self.get_edge(ac1, ac2)
                for r in rs:
                    graph.add_edge(ac1, ac2, rel=r, weight=1.0)

        if len(qcs) == 0:
            qcs.append(-1)

        if len(acs) == 0:
            acs.append(-1)

        if len(paths) == 0:
            for qc in qcs:
                for ac in acs:
                    graph.add_edge(qc,ac, rel=-1, weight=0.1)

        g = nx.convert_node_labels_to_integers(graph, label_attribute='cid') # re-index
        g_str = json.dumps(nx.node_link_data(g))
        return g_str

    def process(self, concepts_fn, paths_fn):
        self.concepts_fn = concepts_fn
        self.paths_fn = paths_fn
        self.load_resources()
        final_text = ""

        for index, qa_pairs in tqdm(enumerate(self.paths_data), desc="Building Graphs", total=len(self.paths_data)):
            # print(self.concepts_data[index])
            # print(self.paths_data[index])
            # print(qa_pairs)
            statement_paths = []
            statement_rel_list = []

            for qa_idx, qas in enumerate(qa_pairs):
                if qas["paths"] is None:
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in qas["paths"]]
                    cur_rels = [item["rel"] for item in qas["paths"]]

                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)
                
            qcs = [self.concept2id[c] for c in self.concept_data[index]["qc"]]
            acs = [self.concept2id[c] for c in self.concept_data[index]["ac"]]
            gstr = self.plain_graph_generation(qcs=qcs, acs=acs, paths=statement_paths, rels=statement_rel_list)
            final_text += gstr + "\n"
            
        out_graph_fn = Path(self.paths_fn).parent / f'{Path(self.paths_fn).stem}_graph'

        with open(out_graph_fn, 'w') as fw:
            fw.write(final_text)

        print(f"Graph Done: {out_graph_fn}")
