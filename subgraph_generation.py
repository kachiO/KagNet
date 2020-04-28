
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/subgraph_generation.ipynb
from pathlib import Path
from grounding.grounding_concepts import GroundConcepts
from pathfinder.pathfinder import PathFinder
from pathfinder.path_scoring import ScorePaths
from graph_generation.graph_gen import GenGraph
from tqdm import tqdm
import networkx as nx, dgl, json, torch

print('\nGrounding...')
root = Path("~/projects/visual-navigation/KagNet")
config_path = root / 'grounding/paths.cfg'
ground = GroundConcepts(config_path)
ground.process(root / 'grounding/objects.txt')
print('\nGrounding...Done')

print('Path finding...')
config_fn = root / 'pathfinder/paths.cfg'
pathfinder = PathFinder(config_fn)
fn = Path('./datasets/ai2thor/ai2thor_objects_to_objects_rooms_include_self_concepts.json')
pathfinder.process(fn, nhops=1, beautify=True)
print('Path finding...Done')

print('Scoring Path...')
scorer = ScorePaths(config_fn)
paths_fn = 'datasets/ai2thor/ai2thor_objects_to_objects_rooms_include_self_concepts_1hops_paths.json.pickle'
concepts_fn = 'datasets/ai2thor/ai2thor_objects_to_objects_rooms_include_self_concepts.json'
scorer.process(paths_fn, concepts_fn)
print('Scoring Path...Done')

print('Graph Generation...')
config_fn = root / 'graph_generation/paths.cfg'
gen_graph = GenGraph(config_fn)
paths_fn = 'datasets/ai2thor/ai2thor_objects_to_objects_rooms_include_self_concepts_1hops_paths.json_scores_pruned.pickle'
concepts_fn = 'datasets/ai2thor/ai2thor_objects_to_objects_rooms_include_self_concepts.json'
gen_graph.process(concepts_fn, paths_fn)
print('Graph Generation...Done')