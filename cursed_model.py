import nltk


class Model:
    def __init__(self):
        self.phonemes = nltk.corpus.cmudict.dict()
        self.markov_states = {}

    def train(self):
        #  get all corpora
        corpora = ["a", "b"]

        for corpus in corpora:
            self.train_on_corpus(corpus)

        #  compute all probabilities
        #  TODO

    def train_on_corpus(self, corpus):
        # preprocessing

        # initialize prev_token to line break token
        prev_token = Token("\n")
        current_token = None
        next_token = None

        # tokens without parts of speech tracked
        no_pos = {}

        # start streaming coprus line by line
        for line in corpus:
            # tokenize this line
            tokens = []  # TODO tokenize here

            for idx, token in enumerate(tokens):
                if idx == 0:
                    # first token of line
                    # when starting a new line, make observation for
                    # last token of last line (presumably "\n")
                    # unless current_token is None (which means its the
                    # beginning of a corpus)
                    first_token = tokens[0]
                    if current_token:
                        current_token.make_observation(previous_tokens, first_token)

                    current_token = first_token

                    break
                elif idx == len(tokens)-1:
                    # last token of line (presumably \n)
                    break
                else:
                    # interior token of line
                    break

            
        while:
            prev, curr, next
            markov_states ={}

            if curr in markov_states:
                curr.make_observation(prev, next)
            else:
                # compute part of speech

                # compute phonemes

                # add 

        for token in markov_states:
            token.compute_probabilities()
