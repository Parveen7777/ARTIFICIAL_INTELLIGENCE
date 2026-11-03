a, b, target = 4, 3, 2

x, y = 0, 0

while x != target:
    if x == 0:
        x = a
    elif y == b:
        y = 0
    else:
        t = min(x, b-y)
        x -= t
        y += t

    print(x, y)

print("Goal Reached!")
