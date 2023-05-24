
#%%
class A(object):
    def __init__(self):
        self.a1 = 1.234

class B(object):
    def __init__(self):
        self.b1 = 0

class C(object):
    def __init__(self):
        self.c1 = 0.123

    def read(self):
        self.a = A()
        self.b = B()

        x = self.a.a1 + 8888
        y = self.b.b1
        print(x, y)

CV = C()
CV.read()


# %%
