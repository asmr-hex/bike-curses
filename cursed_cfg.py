import numpy as np

class CFG:
    def __init__(self):
        self.structures = {}

    def add_structure(self, pattern):
        structure_id = "".join(pattern)
        if pattern == []:
            return

        if structure_id in self.structures:
            self.structures[structure_id].count += 1
        else:
            self.structures[structure_id] = Structure(pattern)

    def get_sample_grammar(self):
        patterns = list(self.structures.values())
        probs = [p.probability for p in patterns]

        return np.random.choice(patterns, p=probs)

    def compute_probability(self):
        for pattern in self.structures.values():
            pattern.compute_probability(len(self.structures))


class Structure:
    def __init__(self, pattern):
        print(" ".join(pattern))
        self.pattern = pattern  # this should be a list of POS tags
        self.count = 1
        self.probability = 0

    def compute_probability(self, total):
        self.probability = float(self.count) / float(total)
