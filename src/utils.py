from numpy import zeros as z, sqrt

class RunningStat(object):
    """ Online computation of mean and variance from
    list of tensor

    :param shape: define the shape of the matrix

    """
    def __init__(self, shape):
        self.u, self._u = z(shape), z(shape)
        self.v, self._v = z(shape), z(shape)
        self.steps = 0

    def update_states(self, image):
        step = (image - self._u)
        self.u = self._u + step/self.steps
        self.v = self._v + step * (image - self.u)
        self._u , self._v = self.u, self.v

    def update(self, image):
        if self.steps == 1:
            self._u = self.u = image
        else:
            self.update_states(image)

    @property
    def variance(self):
        return self.v/(self.steps)

    @property
    def std(self):
        return sqrt(self.variance)

    @property
    def mean(self):
        return self.u

    def __call__(self, image):
        self.steps += 1
        self.update(image)
