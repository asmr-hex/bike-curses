""" generate curse text """
from cursed_model import Model
import pickle


class Curser:
    ''' May the sword of anathema slay
        If anyone steals this bike away. '''

    def __init__(self):
        with open('pretrained.model', 'rb') as fd:
            self.model = pickle.load(fd)

    def write_curse(self):
        ''' compose a couplet of two rhyming lines '''
        first_line = Line(self.model)
        second_line = Line(self.model, rhyme=first_line.terminal_token)
        return [first_line.text, second_line.text]

class Line:
    def __init__(self, model, rhyme=None):
        self.model = model
        cfg = self.model.cfg.get_sample_grammar()
        self.grammar = cfg.pattern

        if rhyme:
            self.terminal_token = self.model.get_rhyme(
                rhyme, required_pos=self.grammar[-1])
        else:
            linebreak_token = self.model.get_token('\n')
            self.terminal_token = self.model.get_previous_token(
                linebreak_token, required_pos=self.grammar[-1])
        if not self.terminal_token:
            raise(Exception('Poem failed to even begin'))
        self.text = self.write_line()

    def write_line(self):
        ''' traverse the markov chain constrained by the grammar '''
        current = self.terminal_token
        text = []
        # traverse the grammar backwards, ignoring the already-set terminal pos
        for pos in self.grammar[-2::-1]:
            text.append(current.word)
            current = self.model.get_previous_token(current, required_pos=pos)
            if not current:
                raise(Exception('Line failed'))

        return text[::-1]


if __name__ == '__main__':
    curser = Curser()
    for i in range(20):
        curse = curser.write_curse()
        print(' '.join(curse[0]))
        print(' '.join(curse[1]))
        print('\n')
