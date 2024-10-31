import numpy as np

class SlideSpliter:
    def __init__(
            self,
            slide_size = 2048,
            patch_size = 256,
            roi_size = 128
        ):

        assert(slide_size == 2048, 'Slide size mismatch!')
    
        self.slide_size = slide_size
        self.patch_size = patch_size
        self.roi_size = roi_size
        self.nums = int(slide_size / roi_size)

        margin_size = int((patch_size - roi_size) / 2)
        self.slide_pts = np.array(
            [0] + list(range(margin_size, slide_size-patch_size, roi_size)) + [slide_size-patch_size]
        )
        self.slide_pts = [(hs, ws) for hs in self.slide_pts for ws in self.slide_pts]

        self.patch_pts = np.array(
            [0] + [margin_size]*(self.nums-2) + [roi_size]
        )
        self.patch_pts = [(hs, ws) for hs in self.patch_pts for ws in self.patch_pts]

    
    def split(self, x):
        assert(x.shape[0] == 2048, 'Slide size mismatch!')
        
        xs = []
        for hs, ws in self.slide_pts:
            xs += [x[hs:hs+self.patch_size, ws:ws+self.patch_size, :]]
            
        return np.array(xs)
        

    def recover(self, xs):
        rec_imgs = np.zeros((self.slide_size, self.slide_size))
        for i, (hs, ws) in enumerate(self.patch_pts):
            rec_hs = int(i / self.nums) * self.roi_size
            rec_ws = int(i % self.nums) * self.roi_size
            rec_imgs[rec_hs:rec_hs+self.roi_size, rec_ws:rec_ws+self.roi_size] = xs[i][hs:hs+self.roi_size, ws:ws+self.roi_size]

        return rec_imgs