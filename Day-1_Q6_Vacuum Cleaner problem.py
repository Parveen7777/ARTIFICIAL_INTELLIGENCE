
room = [1, 0]
pos = 0

for i in range(2):
    if room[pos] == 1:
        print("Room", pos, "is Dirty - Cleaning")
        room[pos] = 0
    else:
        print("Room", pos, "is Clean")
    pos = 1 - pos

print("All Rooms Cleaned:", room)
