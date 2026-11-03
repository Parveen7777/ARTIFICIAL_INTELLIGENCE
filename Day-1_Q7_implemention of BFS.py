
from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

def bfs(start):
    q = deque([start])
    visited = []

    while q:
        node = q.popleft()
        if node not in visited:
            visited.append(node)
            q.extend(graph[node])

    print("BFS Traversal:", visited)

bfs('A')
