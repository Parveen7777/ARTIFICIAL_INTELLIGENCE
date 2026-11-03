
from collections import deque

def solve():
    start, goal = (3, 3, 0), (0, 0, 1)
    moves = [(1,0),(2,0),(0,1),(0,2),(1,1)]
    q = deque([(start, [start])])
    seen = {start}

    while q:
        (m,c,b), path = q.popleft()
        if (m,c,b) == goal:
            for s in path:
                print(s)
            print("Missionaries and Cannibals Problem Solved!")
            return

        for x,y in moves:
            nm, nc = m-x, c-y
            if b == 1: nm, nc = m+x, c+y
            s = (nm,nc,1-b)

            if 0 <= nm <= 3 and 0 <= nc <= 3:
                if (nm == 0 or nm >= nc) and (3-nm == 0 or 3-nm >= 3-nc):
                    if s not in seen:
                        seen.add(s)
                        q.append((s,path+[s]))

solve()
