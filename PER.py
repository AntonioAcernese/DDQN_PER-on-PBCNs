import random
import numpy as np


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    data_pointer = 0
    
    # Initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        # Generate the tree with all nodes values = 0
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        # If we reach bottom, end the search
        if left >= len(self.tree):
            return idx
        
        # else downward search, always search for a higher priority node
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    #total priority == root node
    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, priority, data):
        # Look at what index we want to put the experience
        idx = self.data_pointer + self.capacity - 1
        
        """ tree:
                0
               / \
              0   0
             / \ / \
           idx 0 0  0  We fill the leaves from left to right
        """
        # Update data frame
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(idx, priority)

        self.data_pointer += 1
        # If we're above the capacity, we go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Update the leaf priority score and propagate the change along the tree
    def update(self, idx, priorit):
        # PriorityChange = new priority score - former priority score
        change = priorit - self.tree[idx]

        self.tree[idx] = priorit
        
        # propagate the changePriority along the tree
        self._propagate(idx, change)

    # get a leaf from tree: returns index, priority and the sample
    def get_leaf(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
    
    
class Memory:  # stored as ( s, u, g', s' ) in SumTree
    zeta = 0.01 # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    omega = 0.6 # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    beta = 0.4 # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.zeta) ** self.omega
    
    # Function to store a new experience in the tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def add(self,  sample):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: # no priority updates to tree yet
            max_p = 1  
        self.tree.add(max_p, sample) 
        
    # Sample function, used to pick batch from tree memory
    # - First, sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then uniformly sample a value from each range.
    # - Then search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contain the minibatch
        batch = []
        idxs = []
        # Calculate the priority segment
        # Here, we divide the Range[0, ptotal] into n ranges
        priority_segment  = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            # A value is uniformly sampled from each range
            a = priority_segment  * i
            b = priority_segment  * (i + 1)
            value = random.uniform(a, b)
            
            # Experience that corresponds to each value
            (idx, pry, data) = self.tree.get_leaf(value)
            priorities.append(pry)
            batch.append(data)
            idxs.append(idx)
        
        #P=p_t^omega/(sum(p_t^omega))
        sampling_probabilities = priorities / self.tree.total()
        #IS: w = (1/(N*P))^beta
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
    
    # Update the priorities on the tree
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)