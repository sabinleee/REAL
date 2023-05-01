# define a transform
class Transform(object):
    def __init__(self, K, R, t, img, depth):
        self.K = K
        self.R = R
        self.t = t
        self.img = img
        self.depth = depth
    
    def __call__(self, sample):
        sample = {
            'K': self.K,
            'R': self.R,
            't': self.t,
            'img': self.img,
            'depth': self.depth
        }
        return sample