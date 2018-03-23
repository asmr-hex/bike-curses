""" generate curse text """
from cursed_model import Model
import pickle

class CursedException(Exception):
    """ an exception for when poems just don't turn out how you want """
    pass

class Curser:
    ''' May the sword of anathema slay
        If anyone steals this bike away. '''

    def __init__(self):
        with open('pretrained.model', 'rb') as fd:
            self.model = pickle.load(fd)

    def write_curse(self):
        ''' compose a couplet of two rhyming lines '''
        success = False
        while not success:
            first_line = Line(self.model)
            try:
                second_line = Line(self.model, rhyme=first_line.terminal_token)
            except CursedException:
                continue

            success = True

        return [first_line.text, second_line.text]


class Line:
    def __init__(self, model, rhyme=None):
        self.model = model
        cfg = self.model.cfg.get_sample_grammar()
        self.grammar = cfg.pattern

        # temporarily remove punctuation from cfg pattern
        punc = self.remove_trailing_punc()

        if rhyme:
            self.terminal_token = self.model.get_rhyme(
                rhyme, required_pos=self.grammar[-1])
        else:
            linebreak_token = self.model.get_token('\n')
            self.terminal_token = self.model.get_previous_token(
                linebreak_token, required_pos=self.grammar[-1])
        if not self.terminal_token:
            raise(CursedException('Poem failed to even begin'))

        self.text = self.write_line()

        # add back punctuation if necessary
        if len(punc) != 0:
            self.text.append(punc)

    def remove_trailing_punc(self):
        last_pos = self.grammar[-1]

        if last_pos in ['.', ',', '!', ';', ':', '?', 'POS']:
            self.grammar = self.grammar[:-1]
            return last_pos
        return ''

    def write_line(self):
        ''' traverse the markov chain constrained by the grammar '''
        current = self.terminal_token
        text = []
        # traverse the grammar backwards, ignoring the already-set terminal pos
        for pos in self.grammar[-2::-1]:
            text.append(current.word)
            current = self.model.get_previous_token(current, required_pos=pos)
            if not current:
                raise(CursedException('Line failed'))

        return text[::-1]


if __name__ == '__main__':
    curser = Curser()
    for i in range(20):
        curse = curser.write_curse()
        print(' '.join(curse[0]))
        print(' '.join(curse[1]))
        print('\n')
