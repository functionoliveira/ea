class Mutation:
    def __call__(self):
        raise NotImplementedError("Method 'generate' is not implemented.")

class CurrentToPBestOneBin(Mutation):
    def __call__(self, xi, xbest, xr1, xr2, F):
        return xi + F * (xbest - xi) + F * (xr1-xr2)