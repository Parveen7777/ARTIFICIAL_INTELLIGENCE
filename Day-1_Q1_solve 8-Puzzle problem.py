
from collections import deque

start = (1,2,3,4,0,6,7,5,8)
goal = (1,2,3,4,5,6,7,8,0)

q = deque([start])
visited = {start}

while q:
    s = q.popleft()

    if s == goal:
        print("Puzzle Solved:", s)
        break

    i = s.index(0)

    for j in [i-3, i+3, i-1, i+1]:
        if 0 <= j < 9 and not (i%3==0 and j==i-1) and not (i%3==2 and j==i+1):
            a = list(s)
            a[i],a[j] = a[j],a[i]
            a = tuple(a)

            if a not in visited:
                visited.add(a)
                q.append(a)
