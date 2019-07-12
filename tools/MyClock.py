import time
class MyClock:
    def __init__(self,fmt='mm:ss'):
        self.fmt = fmt
    def tic(self):
        self.start = time.time()
    def toc(self,p = True):
        t = time.time()
        elapsed = t-self.start
        if p:
            print('TIME ELAPSED: {}'.format(self._get_print(elapsed)))
        else:
            return self._get_print(elapsed)
    def _get_print(self,elapsed):
        if self.fmt == 'mm:ss':
            return '{:02d}:{:02d}'.format(int(elapsed//60),int(elapsed%60))
