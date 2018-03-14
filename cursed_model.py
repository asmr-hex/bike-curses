import nltk
from collections import defaultdict
from cursed_token import Token
from cursed_cfg import CFG
import sys
import json
import random
import pickle
import math
import re


class Model:
    def __init__(self):
        self.markov_states = {}
        self.cfg = CFG()
        # ugh -___- we can't pickle lambda because they are nameless...
        # so instead we need to define a named function to pass to the
        # defaultdict. *--*
        def idk():
            return []
        self.rhymes = defaultdict(idk)
        # since the constraints imposed on picking a word based on the
        # pos and conditional probability of the next/prev word, we want
        # to be able to set a minimum probability for every token in the
        # corpus based on the pos. (does that sentence make any sense...??)
        self.pos_lookup = defaultdict(idk)

    def load_pretrained(self, filename):
        pretrained = json.loads(open(filename, 'r'))
        # TODO: json to model?
        self.markov_states = pretrained['???']
        self.cfg = pretrained['???']

    def train(self, filename, mode="markov"):
        print("hi, welcome friend. we're currently training on " + filename)

        # open training file
        with open(filename) as corpus:
            if mode is "markov":
                self.train_markov_states_on_corpus(corpus)
            else:
                self.train_cfg_on_corpus(corpus)

        #  compute all probabilities
        for token in self.markov_states.values():
            token.compute_probabilities(len(self.markov_states))

    def train_cfg_on_corpus(self, corpus):
        print("training context-free grammar model.")

        for line in corpus:
            tokens = nltk.word_tokenize(line)
            tags = [tup[1] for tup in nltk.pos_tag(tokens)]
            self.cfg.add_structure(tags)

        self.cfg.compute_probability()

    def train_markov_states_on_corpus(self, corpus):
        print("training markov model.")

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

                    # track the rhyme at the end of the line
                    self.update_rhymes(self.get_token(prev_token))

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
            print(" ".join(sentence))
            # remove \n tokens so the pos'ing works as expected
            sentence = re.sub('\n', '', sentence)
            # pos tag here
            for posses in nltk.pos_tag(sentence):
                self.markov_states[posses[0]].set_part_of_speech(
                    posses[1])
                # include in pos_lookup table
                self.pos_lookup[posses[1]].append(posses[0])
                self.pos_lookup[posses[1]] = list(set(self.pos_lookup[posses[1]]))
                print(posses, end="")

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

    def update_rhymes(self, token):
        # the rhyming dictionary sorts words by their phonemes backwards
        rhyme_part = self.get_rhyme_part(token)
        self.rhymes[rhyme_part].append(token)

    def get_rhyme_part(self, token):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER',
                  'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
        rhyme_part = []
        # the rhyme part ends at the first vowel in the backwards phonemes
        for phoneme in token.phonemes[::-1]:
            rhyme_part.append(phoneme)
            # nltk's phonemes sometimes mark variants with numbers
            if re.sub('\d', '', phoneme) in vowels:
                break
        return ''.join(rhyme_part)

    def get_rhyme(self, token, required_pos=None):
        rhyme_part = self.get_rhyme_part(token)
        candidates = [{
            "token": x,
            "prob": x.probability
        } for x in self.rhymes[rhyme_part]]

        return self.weighted_choice(candidates, required_pos)

    def get_previous_token(self, token, required_pos=None):
        ''' traverse backwards in the markov model '''
        candidates = []
        # only include options that fit part of speech constraints
        for (candidate, probability) in token.previous_tokens.items():
            candidates.append({
                "token": self.get_token(candidate),
                "prob": probability})

        return self.weighted_choice(candidates, required_pos)

    def get_token(self, text):
        return self.markov_states[text]

    def weighted_choice(self, candidates, required_pos):
        ''' choose a token probabilistically biased by constraints
        constraints:
         (1) part of speech
         (2) conditional transition probability (prev/next word given word)
        '''
        if required_pos:
            candidates = [w for w in candidates if required_pos in w['token'].pos]
        if len(candidates) == 0:
            # lookup an alternate word with the required part of speech
            return random.choice(self.pos_lookup[required_pos])

        distribution = []
        for c in candidates:
            distribution += [c["token"]] * math.ceil(c["prob"] * len(candidates))

        return random.choice(distribution)


if __name__ == "__main__":
    # we are gunna run this
    mode = sys.argv[1]

    model = Model()

    if mode == "train":
        model.train("corpora/tirukkural_couplets.txt", mode="cfg")
        model.train(
            "corpora/paradiselost-normalized-abridged.txt", mode="markov")

        with open("pretrained.model", "wb") as fd:
            pickle.dump(model, fd)

    if mode == "generate":
        with open("pretrained.model", "rb") as fd:
            model = pickle.load(fd)
