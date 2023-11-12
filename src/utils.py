class Instance:
    """
    Class that represents an instance of the S-Plex problem
    """
    def __init__(self, s, n, m, edge_info):
        self.s = int(s)
        self.n = int(n)
        self.m = int(m)
        self.edge_ingo = edge_info

        # Create additional structures needed
        self.N = list(range(1, self.n + 1))
