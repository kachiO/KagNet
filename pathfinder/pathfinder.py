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
import numpy as np
from pathlib import Path
import jsbeautifier

opts = jsbeautifier.default_options()
opts.indent_size = 2

class PathFinder(object):
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.path = Path(config_path).parent.parent
        self.concept2id, self.id2concept = {}, {}
        self.id2relation, self.relation2id = {}, {}

        self.load_resources()
        self.load_cpnet()

    def load_resources(self,):
        with open(self.path / self.config["paths"]["concept_vocab"][3:], "r", encoding="utf8") as f:
            for w in f.readlines():
                self.concept2id[w.strip()] = len(self.concept2id)
                self.id2concept[len(self.id2concept)] = w.strip()

        with open(self.path / self.config["paths"]["relation_vocab"][3:], "r", encoding="utf8") as f:
            for w in f.readlines():
                self.id2relation[len(self.id2relation)] = w.strip()
                self.relation2id[w.strip()] = len(self.relation2id)

    def load_cpnet(self):
        print('loading conceptnet graph...')
        self.cpnet = nx.read_gpickle(self.path / self.config["paths"]["conceptnet_en_graph"][3:])
        self.cpnet_simple = nx.Graph()
        for u, v, data in self.cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if self.cpnet_simple.has_edge(u, v):
                self.cpnet_simple[u][v]['weight'] += w
            else:
                self.cpnet_simple.add_edge(u, v, weight=w)
        print('loading conceptnet graph...done')

    def get_edge(self, src_concept, tgt_concept):
        rel_list = self.cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]["rel"] for item in rel_list]))

    def find_paths(self, source, target, nhops=4, ifprint = False):
        try:
            s = self.concept2id[source]
            t = self.concept2id[target]
        except:
            if ifprint:
                print('missing concept ids')
            return

        if s not in self.cpnet_simple.nodes() or t not in self.cpnet_simple.nodes():
            return

        all_path = []
        all_path_set = set()

        for max_len in range(1, 1 + nhops):
            for p in nx.all_simple_paths(self.cpnet_simple, source=s, target=t, cutoff=max_len):
                path_str = "-".join([str(c) for c in p])
                if path_str not in all_path_set:
                    all_path_set.add(path_str)
                    all_path.append(p)
                if len(all_path) >= 100:  # top shortest 300 paths
                    break
            if len(all_path) >= 100:  # top shortest 300 paths
                break

        all_path.sort(key=len, reverse=False)
        pf_res = []
        for p in all_path:
            rl = []
            for src in range(len(p) - 1):
                src_concept = p[src]
                tgt_concept = p[src + 1]

                rel_list = self.get_edge(src_concept, tgt_concept)
                rl.append(rel_list)
                if ifprint:
                    rel_list_str = []
                    for rel in rel_list:
                        if rel < len(self.id2relation):
                            rel_list_str.append(self.id2relation[rel])
                        else:
                            rel_list_str.append(self.id2relation[rel - len(self.id2relation)]+"*")
                    print(self.id2concept[src_concept], "----[%s]---> " %("/".join(rel_list_str)), end="")
                    if src + 1 == len(p) - 1:
                        print(self.id2concept[tgt_concept], end="")
            if ifprint:
                print()

            pf_res.append({"path": p, "rel": rl})
        return pf_res

    def process(self, concepts_fn, nhops, verbose=False):
        with open(concepts_fn, 'r') as fp:
            concepts_data = json.load(fp)
       
        output_path = Path(concepts_fn).parent
        paths_found = []

        for item in tqdm(concepts_data):
            _object = sorted(item["ac"])
            _thor_objects = item["qc"]
            _found_paths = []  # path finding results
            
            for obj in _object:
                for th_obj in _thor_objects:
                    pf_res = self.find_paths(th_obj, obj, nhops, ifprint=verbose)
                    _found_paths.append({"object":obj, "thor_objects":th_obj, "paths":pf_res})

            paths_found.append(_found_paths)
        out_fn = output_path / f'{concepts_fn.name}.{nhops}hops_paths.json'
        
        with open(out_fn , 'w') as fi:
            fi.write(jsbeautifier.beautify(json.dumps(paths_found), opts))

        print(f'Found paths written to {out_fn}')
        



