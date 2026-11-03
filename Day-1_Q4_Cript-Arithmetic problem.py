
from itertools import permutations

words = ["SEND", "MORE", "MONEY"]
letters = set("".join(words))

for p in permutations(range(10), len(letters)):
    d = dict(zip(letters, p))
    if any(d[w[0]] == 0 for w in words):
        continue

    send = int("".join(str(d[c]) for c in "SEND"))
    more = int("".join(str(d[c]) for c in "MORE"))
    money = int("".join(str(d[c]) for c in "MONEY"))

    if send + more == money:
        print(send, "+", more, "=", money)
        break
