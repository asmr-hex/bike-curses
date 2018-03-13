class CFG:
    def __init__(self):
        self.structures = {}
        self.total = 0

    def add_structure(self, structure):
        structure_id = "".join(structure)
        if structure.id in self.structures:
            self.structures[structure_id].count += 1
        else:
            self.structures[structure_id] = Structure(structure)

        self.total += 1

    def compute_probability(self):
        for structure in self.structures:
            structure.compute_probability(self.total)


class Structure:
    def __init__(self, pattern):
        self.pattern = pattern  # this should be a list of POS tags
        self.count = 1
        self.probability = 0

    def compute_probability(self, total):
        self.probability = float(self.count) / float(total)
