class Token:
    """A class to describe tokens in our model"""
    def __init__(self, word, phonemes):
        self.word = word
        self.freq = 0
        self.pos = None
        self.phonemes = phonemes

        # maps from token to count
        self.previous_tokens = {}
        self.n_previous_tokens = 0
        self.next_tokens = {}
        self.n_next_tokens = 0

    def set_part_of_speech(self, pos):
        self.pos = pos

    def make_observation(self, prev_token, next_token):
        self.increment_frequency()
        self.add_previous(prev_token)
        self.add_next(next_token)

    def increment_frequency(self):
        self.freq += 1

    def add_previous(self, token):
        self.n_previous_tokens += 1
        self.add_token(self.previous_tokens, token)

    def add_next(self, token):
        self.n_next_tokens += 1
        self.add_token(self.next_tokens, token)

    def add_token(self, collection, token):
        if token in collection:
            collection[token] = 1
        else:
            collection[token] += 1

    def compute_probabilities(self):
        """run this method at the end of training"""
        self.previous_tokens = map(
            lambda k, v: float(v) / float(self.n_previous_tokens),
            self.previous_tokens)
        self.next_tokens = map(
            lambda k, v: float(v) / float(self.n_next_tokens),
            self.previous_tokens)
