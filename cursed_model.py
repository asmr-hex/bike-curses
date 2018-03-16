import nltk
from cursed_token import Token
from cursed_cfg import CFG
import random
import pickle
import math
import re
import time


class Model:
    def __init__(self):
        self.markov_states = {}
        self.cfg = CFG()
        # ugh -___-
        self.rhymes = {}
        # since the constraints imposed on picking a word based on the
        # pos and conditional probability of the next/prev word, we want
        # to be able to set a minimum probability for every token in the
        # corpus based on the pos. (does that sentence make any sense...??)
        self.pos_lookup = {}

    def train(self, filename, mode="markov"):
        print("hi, welcome friend. we're currently training on " + filename)

        start_time = time.time()

        # open training file
        with open(filename) as corpus:
            if mode is "markov":
                self.train_markov_states_on_corpus(corpus)
                #  compute all probabilities
                for token in self.markov_states.values():
                    token.compute_probabilities(len(self.markov_states))

                # discard cfg patterns which contain POS which are not represented in
                # the markov corpus.
                safe_structures = {}
                for idx, structure in self.cfg.structures.items():
                    valid_structure = True
                    for pos in structure.pattern:
                        if pos not in self.pos_lookup:
                            valid_structure = False
                            break
                    if valid_structure:
                        safe_structures[idx] = structure
                self.cfg.structures = safe_structures
                self.cfg.compute_probability()
            else:
                self.train_cfg_on_corpus(corpus)

        end_time = time.time()
        print('training took: ' + str(end_time - start_time) + " seconds")

    def train_cfg_on_corpus(self, corpus):
        print("training context-free grammar model.")

        for line in corpus:
            tokens = nltk.word_tokenize(line)
            tags = [tup[1] for tup in nltk.pos_tag(tokens)]
            self.cfg.add_structure(tags)

        #self.cfg.compute_probability()

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
            tokens = nltk.word_tokenize(line.lower())

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
            sentence = [s for s in sentence if s != '\n']
            # pos tag here
            for word, pos in nltk.pos_tag(sentence):
                self.markov_states[word].set_part_of_speech(pos)
                # include in pos_lookup table
                if pos not in self.pos_lookup:
                    self.pos_lookup[pos] = []
                self.pos_lookup[pos].append(word)
                self.pos_lookup[pos] = list(set(self.pos_lookup[pos]))
                #print(word, pos, end="")

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
        if rhyme_part not in self.rhymes:
            self.rhymes[rhyme_part] = []
        self.rhymes[rhyme_part].append(token)

    def get_rhyme_part(self, token):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER',
                  'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
        rhyme_part = []
        # the rhyme part ends at the first vowel in the backwards phonemes
        found_vowel = False
        for phoneme in token.phonemes[::-1]:
            if found_vowel and re.sub(r'\d', '', phoneme) not in vowels:
                break
            rhyme_part.append(phoneme)
            # nltk's phonemes sometimes mark variants with numbers
            if re.sub(r'\d', '', phoneme) in vowels:
                # once we find a vowel, we should include all following vowels
                # and stop when we hit a consonant
                found_vowel = True
        return ''.join(rhyme_part)

    def get_rhyme(self, token, required_pos=None):
        rhyme_part = self.get_rhyme_part(token)
        candidates = [{
            "token": x,
            "prob": x.probability
        } for x in self.rhymes[rhyme_part] if x.word != token]

        rhyme = self.weighted_choice(candidates, required_pos)

        if rhyme_part == self.get_rhyme_part(rhyme) and len(rhyme_part) > 0:
            return rhyme

        return False

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
            candidates = [w for w in candidates \
                if required_pos in w['token'].pos]
        if len(candidates) == 0:
            # lookup an alternate word with the required part of speech
            return self.get_token(random.choice(self.pos_lookup[required_pos]))

        distr = []
        for c in candidates:
            distr += [c["token"]] * math.ceil(c["prob"] * len(candidates))

        return random.choice(distr)


if __name__ == "__main__":
    # we are gunna run this
    model = Model()

    model.train("corpora/tirukkural_couplets.txt", mode="cfg")
    model.train(
        "corpora/paradiselost-normalized.txt", mode="markov")

    with open("pretrained.model", "wb") as fd:
        pickle.dump(model, fd)

