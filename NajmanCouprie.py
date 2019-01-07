"""
Building the component tree in quasi-linear time.
L. Najman and M. Couprie

IEEE Transactions on Image Processing, Institute of Electrical and Electronics Engineers,
2006, 15 (11), pp.3531-3539. <hal-00622110>

Implementation
C. Meyer
"""


class NajmanCouprie():

    Q = []
    Par = []
    Rnk = []

    def makeset(self, x):
        self.Par[x] = x
        self.Rnk[x] = 0

    def find(self, x):
        if (self.Par[x] != x):
            self.Par[x] = self.find(self, self.Par[x])
        return self.Par[x]

    def link(self, x, y):
        if(self.Rnk[x] > self.Rnk[y]):
            x, y = y, x

        if(self.Rnk[x] == self.Rnk[y]):
            self.Rnk[y] = self.Rnk[y] + 1

        self.Par[x] = y

        return y


