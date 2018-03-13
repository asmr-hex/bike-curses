class CFG:
    def __init__(self):
        self.structures = {}
        self.total = 0

    def add_structure(self, pattern):
        structure_id = "".join(pattern)
        if structure_id in self.structures:
            self.structures[structure_id].count += 1
        else:
            self.structures[structure_id] = Structure(pattern)

        self.total += 1

    def compute_probability(self):
        for pattern in self.structures.values():
            pattern.compute_probability(self.total)


class Structure:
    def __init__(self, pattern):
        print(" ".join(pattern))
        self.pattern = pattern  # this should be a list of POS tags
        self.count = 1
        self.probability = 0

    def compute_probability(self, total):
        self.probability = float(self.count) / float(total)
