'''
Here i will implement graph class from scratch with adjacency matrix and implemented BFS and DFS traversals. 
Here i will calculate graph laplacian and use it's eigen values to detect components and cluster nodes
Demostration of graphy theory in action with applying pagerank algorithm, Dijkstra algorith with weights, and GNN style messaging. 
'''
from collections import deque
import heapq
import numpy as np

class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbours(self, node):
        return list(self.adj[node].keys())
    
    def degree(self, node):
        return len(self.adj[node])
    
    def adjacency_matrix(self):

        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A
    
    def degree_matrix(self):

        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.degree(i)
        return D
    
    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()
    

## BFS and DFS

def bfs(graph, start):
    visited = set()
    order = []
    distances = {}
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbour in graph.neighbours(node):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, dist + 1))
    return order, distances

def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbour in reversed(graph.neighbours(node)):
            if neighbour not in visited:
                stack.append(neighbour)
    return order


def connected_components(graph):
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)
            visited.update(order)
            components.append(order)

    return components

def laplacian_eigenvalues(graph):

    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues

def spectral_clustering(graph, k=2):

    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:k+1]

    labels = np.zeros(graph.n, dtype=int)
    for i in range(graph.n):
        if features[i, 0] >= 0:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels


## message passig
def message_passing(graph, features, weight_matrix):
    A = graph.adjacency_matrix()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums
    aggregated = A_norm @ features
    output = aggregated @ weight_matrix
    return output


## Exercise 1: Implement PageRank from scratch. Start with uniform scores. At each step: 
# score(v) = (1-d)/n + d * sum(score(u)/out_degree(u)) 
# for all u pointing to v. Use d=0.85. Run until convergence (change < 1e-6). Test on a small web graph.

def page_rank_formula(graph, d=0.85, tol=1e-6, max_iter = 100):
    n = graph.n

    scores = {i: 1.0 / n for i in range(n)}
    
    for _ in range(max_iter):
        new_scores = {}

        for v in range(n):
            inbound_sum = 0
            for u in range(n):
                # check if u points to v
                if v in graph.adj[u]:
                    inbound_sum += scores[u] / len(graph.adj[u])
            
            # applying the pagerank formula
            new_scores[v] = ((1 - d) / n) + (d * inbound_sum)

        change = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        if change < tol:
            return new_scores
    
        scores = new_scores
    return scores

# testing our function
web_graph = Graph(n_nodes=3, directed=True)

# adding links node 0 links to 1 and 2
web_graph.add_edge(0, 1)
web_graph.add_edge(0, 2)

# node 1 links to 2
web_graph.add_edge(1, 2)

# node 2 links back to 0
web_graph.add_edge(2, 0)

final_score = page_rank_formula(web_graph)
print(final_score)

## Exercise 2: Find communities using spectral clustering. Create a graph with two clearly 
# separated clusters (e.g., two cliques connected by a single edge). Run spectral clustering and verify it finds 
# the right split. What happens as you add more cross-cluster edges?

new_graph = Graph(n_nodes=6)

# cliques A (node 0, 1, 2)
new_graph.add_edge(0, 1)
new_graph.add_edge(1, 2)
new_graph.add_edge(2, 0)

# the bridge connecting them
new_graph.add_edge(2, 3)
# adding another cross cluster brdige
new_graph.add_edge(5, 0)
new_graph.add_edge(4, 2)

# cliques B nodes(3, 4, 5)
new_graph.add_edge(3, 4)
new_graph.add_edge(4, 5)
new_graph.add_edge(5, 3)

lables = spectral_clustering(new_graph)
print(lables)
# with above setup of one cross cluster edge from 2 to 3, i got clear cluster sepration of [111 000]. after that i added few more
# cross bridge(5-0) and (4-2) and i see the same results it is because still cutting the graph from middle is the best way possible
# for algorithm and cut least connection as possible, if it cuts anny triangle it is more connections than cutting 3 from middle, so 
# results stay the same.

# here i run expriment with all of them connecting together
clique_graph = Graph(n_nodes=6)
clique_graph.add_edge(0, 3)
clique_graph.add_edge(0, 4)
clique_graph.add_edge(0, 5)
clique_graph.add_edge(1, 3)
clique_graph.add_edge(1, 4)
clique_graph.add_edge(1, 5)
clique_graph.add_edge(2, 3)
clique_graph.add_edge(2, 4)
clique_graph.add_edge(2, 5)

lables_cliques = spectral_clustering(clique_graph)
print(f"the labels we got after connecting all nodes with each other is {lables_cliques}")


## exercise 3: Implement Dijkstra's algorithm for shortest paths in weighted graphs. 
# Compare results to BFS on the same graph with uniform weights.

def Dijkstra(graph, start):

    # initializing distance with infinity, start node with 0
    distances = {node: float('inf') for node in range(graph.n)}
    distances[start] = 0
    
    # track the actual path back to start
    predecessors = {node: None for node in range(graph.n)}

    # priority queue stores tuples of  (distance, node)
    # it automatically sorts by the first element (distance)

    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_dist > distances[current_node]:
            continue

        # check all neighbours of the current node
        for neighbor in graph.neighbours(current_node):
            weight = graph.adj[current_node][neighbor]
            
            path_cost = current_dist + weight

            # if shortest path is found update it
            if path_cost < distances[neighbor]:
                distances[neighbor] = path_cost
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (path_cost, neighbor))
    return distances, predecessors

def get_shortest_path(predecessors, start, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    return path if path[0] == start else []

## testing on toy example
geo_graph = Graph(n_nodes=4, directed=False)

# add weights
geo_graph.add_edge(0, 1, weight=1.0)
geo_graph.add_edge(0, 2, weight=4.0)
geo_graph.add_edge(1, 2, weight=2.0)
geo_graph.add_edge(1, 3, weight=6.0)
geo_graph.add_edge(2, 3, weight=1.0)

# RUNNIG DIJKSTRA ALGO FORM NODE 0

distances, predecessors = Dijkstra(geo_graph, start=0)
order_bfs, distances_bfs = bfs(geo_graph, start=0)
print("shortest distance from node 0 using bfs is :", distances_bfs)
print("Ordder using bfs is", order_bfs)
print("shortest distance from node 0:", distances)

print("Path to node 3: ", get_shortest_path(predecessors, start=0, target=3))
# bfs assumes equal weight so the shortest ditance it found from node 0 to other is : {0: 0, 1: 1.0, 2: 3.0, 3: 4.0}

## Exercise 4: Build a 2-layer message passing network. Apply message passing twice with different weight matrices. 
# Show that after 2 rounds, each node has information from its 2-hop neighborhood.

m_graph = Graph(n_nodes=3)
m_graph.add_edge(0, 1)
m_graph.add_edge(1, 2)


features = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
W1 = np.eye(3)
W2 = np.eye(3)

print(features) 
layer1_output = message_passing(m_graph, features, W1)
print("\n--- After 1 Round of Message Passing (1-Hop) ---")
print(layer1_output)

# --- Layer 2 Execution ---
layer2_output = message_passing(m_graph, layer1_output, W2)
print("\n--- After 2 Rounds of Message Passing (2-Hop) ---")
print(layer2_output)

## Exercise 5: Analyze a real-world graph. Use the Karate Club graph (34 nodes, 78 edges). Compute degree distribution, 
# Laplacian eigenvalues, and spectral clustering. Compare the spectral clustering result to the known ground truth split.

import networkx as nx

nx_G = nx.karate_club_graph()

num_nodes = nx_G.number_of_nodes()
my_graph = Graph(n_nodes = num_nodes, directed=False)

for u, v in nx_G.edges():
    my_graph.add_edge(u, v)

# 1 degree Distribution

degrees = [my_graph.degree(i) for i in range(my_graph.n)]
print(f"Average Node Degree: {np.mean(degrees)}")
print("Max Degree:", np.max(degrees))

# 2 Eigen values
eigenvalues = laplacian_eigenvalues(my_graph)
print("\n first 5 laplacian eigenvalues:", eigenvalues[:5])

# Analysis 3:
predicted_labels = spectral_clustering(my_graph, k=2)

# extrating Networkx's real-world ground tructh factions
ground_truth = [0 if nx_G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(num_nodes)]

print("------Comparison---------------")
print("predicted Factions: ", list(predicted_labels))
print("Ground truth Factions: ", ground_truth)

# calculate accuracy 
accuracy = np.mean(predicted_labels == np.array(ground_truth))

accuracy = max(accuracy, 1 - accuracy)

print(f"\nSpectral Clustering Accuracy: {accuracy * 100:.2f}%")