class CanonicalCorrelation(object):
    """
    Compute the canonical correlation exposed in Cnn
    similarity activations  tensors.
    """

    @staticmethod
    def pairwise_distance(x):
        """
        :param x:
        :return:
        """
        linear = torch.matmul(x, x.T)
        difference = linear.diag() - linear
        return difference + difference.T

    @staticmethod
    def median_band(distances):
        return -1 / (2 * torch.median(distances))

    def rbf(self, x):
        """
        :param x:
        :return:
        """
        norm = self.pairwise_distance(x)
        bandwidth = self.median_band(norm)
        return torch.exp(bandwidth * norm)

    @staticmethod
    def center(gram, n):
        """
        :param gram:
        :param n:
        :return:
        """

        gram.fill_diagonal_(0)
        means = gram.sum(0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)

        return gram

    def __call__(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        gram_x, gram_y = self.rbf(x), self.rbf(y)
        cx = self.center(gram_x, len(gram_x))
        cy = self.center(gram_y, len(gram_y))
        scaled = cx.flatten().dot(cy.flatten())
        return scaled / (cx.norm() * cy.norm())
