def getMinimumCost(k, c):
    c.sort(reverse = True)
    cost = 0
    n = len(c)
    m = 1 

    for i in range(n):
        cost += c[i] * m
        if (i+1) % k == 0:
            m += 1
    return cost

print(getMinimumCost(3, [1, 2, 3, 4]))