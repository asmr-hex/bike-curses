""" generate curse text """
import cursed_model

class Curser:
    ''' May the sword of anathema slay
        If anyone steals this bike away. '''

    def __init__(self):
        self.model = cursed_model.Model()
        self.model.load_pretrained('parameters.json')

    def write_curse(self):
        ''' compose a couplet of two rhyming lines '''
        first_line = Line(self.model)
        second_line = Line(self.model, rhyme=first_line.terminal_token)
        return [first_line.text, second_line.text]

class Line:
    def __init__(self, model, rhyme=None):
        self.model = model
        self.grammar = self.model.get_sample_grammar()

        if rhyme:
            self.terminal_token = self.model.get_rhyme(
                rhyme, pos=self.grammar[-1])
        else:
            linebreak_token = self.model.get_token('\n')
            self.terminal_token = self.model.get_previous_token(
                linebreak_token, pos=self.grammar[-1])
        self.text = self.write_line()

    def write_line(self):
        ''' traverse the markov chain constrained by the grammar '''
        current = self.terminal_token
        text = []
        # traverse the grammar backwards, ignoring the already-set terminal pos
        for pos in self.grammar[-2::-1]:
            text.append(current.word)
            current_token = self.model.get_previous_token(current, pos=pos)
            if not current_token:
                print('Line failed')
                return False

        return text[::-1]


if __name__ == '__main__':
    curser = Curser()
    print(curser.write_curse())
