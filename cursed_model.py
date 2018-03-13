import nltk
from cursed_token import Token


class Model:
    def __init__(self):
        self.markov_states = {}

    def train(self, filename, mode="markov"):
        # open training file
        with open(filename) as corpus:
            if mode is "markov":
                self.train_markov_states_on_corpus(corpus)
            else:
                self.train_cfg_on_corpus(corpus)

        #  compute all probabilities
        for token in self.markov_states.values():
            token.compute_probabilities()

    def train_cfg_on_corpus(self, corpus):
        return

    def train_markov_states_on_corpus(self, corpus):
        # preprocessing

        # initialize prev_token to line break token
        prev_token = Token("\n")
        current_token = None
        next_token = None

        # keep track of sequence of tokens for each sentencey thing
        sentence = []

        # start streaming coprus line by line
        for line in corpus:
            # tokenize this line
            tokens = nltk.word_tokenize(line)

            # ignore single word lines
            if len(tokens) < 2:
                continue

            # appending new line since word_tokenize takes it out
            tokens += ["\n"]

            for idx, token in enumerate(tokens):
                if idx == 0:
                    # first token of line
                    # when starting a new line, make observation for
                    # last token of last line (presumably "\n")
                    # unless current_token is None (which means its the
                    # beginning of a corpus)
                    #
                    # current_token should be "\n"
                    first_token = token
                    if current_token:
                        # tokenize previous lines \n token since we now
                        # have all its info w.r.t. prev and next
                        self.update_markov_state(
                            prev_token, current_token, first_token)

                    prev_token = current_token
                    current_token = first_token
                    next_token = tokens[1]

                    # tokenize this current token
                    self.update_markov_state(
                            prev_token, current_token, next_token)

                elif idx == len(tokens)-1:
                    # last token of line (presumably \n)
                    prev_token = current_token
                    current_token = token  # this should be \n
                else:
                    # interior token of line
                    prev_token = current_token
                    current_token = token
                    next_token = tokens[idx+1]

                    self.update_markov_state(
                        prev_token, current_token, next_token)

                # update parts of speech if necessary
                sentence = self.track_pos(token, sentence)

    def track_pos(self, token, sentence):
        # check whether we are at the end of a sentencey thing
        if token in [".", "!", "?", ";"]:
            # pos tag here
            for posses in nltk.pos_tag(sentence):
                self.markov_states[posses[0]].set_part_of_speech(
                    posses[1])

            # clear sentence buffer
            sentence = []
        else:
            # continue adding to the sentence thingy
            sentence.append(token)

        return sentence

    def update_markov_state(self, prev_token, curr_token, next_token):
        if curr_token not in self.markov_states:
            # if this is a new token, assign phonemes
            self.markov_states[curr_token] = Token(curr_token)

        # make observation for this token
        self.markov_states[curr_token].make_observation(
            prev_token,
            next_token)
