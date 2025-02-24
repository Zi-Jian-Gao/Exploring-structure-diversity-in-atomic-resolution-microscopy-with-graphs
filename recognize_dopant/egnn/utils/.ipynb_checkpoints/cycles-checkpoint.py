import cv2
import copy
import glob
import json
import torch
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, Dataset


def find_cycles(graph, length):
    def dfs(node, visited, path):
        if len(path) == length:
            # Check if it forms a cycle
            if path[0] in graph[path[-1]]:
                sorted_path = tuple(sorted(path))
                if sorted_path not in cycles:
                    cycles.add(sorted_path)
            return

        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, visited, path + [neighbor])
        visited[node] = False

    cycles = set()
    num_nodes = len(graph)
    visited = [False] * num_nodes

    for node in range(num_nodes):
        dfs(node, visited, [node])

    return cycles


def get_graph(edge_index):
    graph = {}
    for edge in edge_index:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
        
    return graph

# graph = get_graph(sample_edge_index.T.numpy())
# cycles = np.array(list(find_cycles(graph, 7)))