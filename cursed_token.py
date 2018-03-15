import nltk


class Token:
    """A class to describe tokens in our model"""
    def __init__(self, word):
        self.word = word
        self.freq = 0
        self.probability = 0
        self.pos = []
        self.phonemes = []
        #if word in nltk.corpus.cmudict.dict():
        #    self.phonemes = nltk.corpus.cmudict.dict()[word][0]
        #    print("\t\t"+word+" ("+" ".join(self.phonemes)+")")

        # maps from token to count
        self.previous_tokens = {}
        self.n_previous_tokens = 0
        self.next_tokens = {}
        self.n_next_tokens = 0

    def set_part_of_speech(self, pos):
        pos_set = set(self.pos)
        pos_set.add(pos)
        self.pos = list(pos_set)

    def make_observation(self, prev_token, next_token):
        print(
            (prev_token or "None") +
            " --> "+self.word +
            " --> "+next_token)
        self.freq += 1
        self.add_previous(prev_token)
        self.add_next(next_token)

    def add_previous(self, token):
        if not token:
            return

        self.n_previous_tokens += 1
        self.add_token(self.previous_tokens, token)

    def add_next(self, token):
        if not token:
            return

        self.n_next_tokens += 1
        self.add_token(self.next_tokens, token)

    def add_token(self, collection, token):
        """adds a token to a collection of previous or next tokens. """
        if token in collection:
            collection[token] += 1
        else:
            collection[token] = 1

    def compute_probabilities(self, total):
        """run this method at the end of training"""
        # compute conditional probabilities
        for k, v in self.previous_tokens.items():
            self.previous_tokens[k] = float(v) / float(self.n_previous_tokens)

        for k, v in self.next_tokens.items():
            self.previous_tokens[k] = float(v) / float(self.n_next_tokens)

        # compute marginal probability of this token
        self.probability = float(self.freq) / float(total)
