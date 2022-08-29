
def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
    n = len(keyName)    
    times = defaultdict(list)
    
    for i in range(n):
        name = keyName[i]
        time = keyTime[i]
        
        t = time.split(':')
        h = int(t[0])
        m = int(t[1])
        
        times[name].append(h*60+m)  # for each name, add the int of total minutes
    res = []
    
    for name in times:
        time_list = sorted(times[name])     # sorted the time list for that name
        queue = deque()     # keep track of that worker's entry for the hour
        
        for time in time_list:  # loop through entries
            queue.append(time)    # add each entry
            
            while time - queue[0] > 60: # if first entry was made over an hour ago, pop from the queue
                queue.popleft()
            if len(queue) >= 3: # if there were more than 3 entries in 1 hour, add name to res and break the count for that name
                res.append(name)
                break
    return sorted(res)  # return list of workers in alphabetical order
        
    
