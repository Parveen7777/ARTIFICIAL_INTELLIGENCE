import heapq

graph = {'A': [('B', 1), ('C', 3)],
         'B': [('D', 1)],
         'C': [('D', 1)],
         'D': []}

h = {'A': 2, 'B': 1, 'C': 1, 'D': 0}

def astar(start, goal):
    q = [(h[start], 0, start, [start])]
    while q:
        f, g, node, path = heapq.heappop(q)
        if node == goal:
            return path
        for n, cost in graph[node]:
            heapq.heappush(q, (g+cost+h[n], g+cost, n, path+[n]))

print("Path:", astar('A', 'D'))
