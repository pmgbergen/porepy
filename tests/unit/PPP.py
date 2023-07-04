
import numpy as np
import pdb


# SHALLOW COPY
def myprint(var):
    print("\n" + var + " = ", eval(var))


aaa = np.array([1,2,3])
bbb = aaa 

aaa[0] = 10
myprint('aaa')
myprint('bbb')

myprint('id(aaa)')
myprint('id(bbb)')


aaa = np.array([1,2,3,4])
myprint('aaa')
myprint('bbb')


aaa = [4,5,6]
bbb = aaa # they are exaclty the same
aaa.append('mela')
myprint('aaa')
myprint('bbb')

aaa = [4,5,6]
bbb = list(aaa) # shallow copy, but since there is one level this is equivalent to a deep copy
aaa[0] = 'fulmina'
aaa.append('mela')
myprint('aaa')
myprint('bbb')

aaa = [[4,5,6]]
bbb = list(aaa) # only the original objs inside aaa are the same of bbb
aaa[0][0] = 'fulmina'
aaa.append('mela')
myprint('aaa')
myprint('bbb')
myprint('id(aaa)') # different objects
myprint('id(bbb)')
myprint('id(aaa[0])') # they are the same objs
myprint('id(bbb[0])')








pdb.set_trace()

