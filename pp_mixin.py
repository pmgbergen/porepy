
import numpy as np
import pdb



class One():
    def __init__(self):
        print('init ONE')

    def method_one(self):
        print('method ONE')

class Two():
    def __init__(self):
        print('init TWO')

    def method_two(self):
        print('method TWO')

class Three():
    def __init__(self):
        print('init THREE')

    def method_three(self):
        print('method THREE')

class Mixin_one(Two, One, Three):
    """"""
    def method_two(self):
        print('bla bla bla')

class Four():
    def __init__(self):
        print('init FOUR')

    def method_three(self):
        print('method three wrong')


class Mixin_two(Mixin_one, Four):
    """ """

class Mixin_three(Four, Mixin_one):
    """ """



gigi = Mixin_one()
mario = Mixin_two()
luigi = Mixin_three()


pdb.set_trace()
print('\nDone!')



