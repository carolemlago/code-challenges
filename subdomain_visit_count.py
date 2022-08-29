def subdomainVisits(self, cpdomains):
    counts = collections.defaultdict(int)    # initialize a dictionaty to count visits and domain
    for domain in cpdomains:    # iterate through items in domain and split from counts
        count, domain = domain.split()

        count = int(count)
        fragments = domain.split('.')

        for i in range(len(fragments)):
            counts[".".join(fragments[i:])] += count

    return [f"{count} {domain}" for domain, count in counts.items()]
        
