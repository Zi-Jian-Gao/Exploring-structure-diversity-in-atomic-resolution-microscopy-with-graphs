import numpy as np

class SlideSpliter:
    def __init__(
            self,
            slide_size = 2048,
            patch_size = 256,
            roi_size = 128
        ):

        # assert(slide_size == 2048, 'Slide size mismatch!')
    
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
        # assert(x.shape[0] == 2048, 'Slide size mismatch!')
        
        xs = []
        for hs, ws in self.slide_pts:
            xs += [x[hs:hs+self.patch_size, ws:ws+self.patch_size, :]]
            
        return np.array(xs)

    def split2(self, x):
        # assert (x.shape[0] == 2048, 'Slide size mismatch!')

        xs = []
        for hs, ws in self.slide_pts:
            xs += [x[hs:hs + self.patch_size, ws:ws + self.patch_size, :]]

        return np.array(xs)

    def recover(self, xs):
        rec_imgs = np.zeros((self.slide_size, self.slide_size))
        for i, (hs, ws) in enumerate(self.patch_pts):
            rec_hs = int(i / self.nums) * self.roi_size
            rec_ws = int(i % self.nums) * self.roi_size
            rec_imgs[rec_hs:rec_hs+self.roi_size, rec_ws:rec_ws+self.roi_size] = xs[i][hs:hs+self.roi_size, ws:ws+self.roi_size]

        return rec_imgs

class SlideSpliter_unequal:
    def __init__(self, patch_size=256, roi_size=128):
        self.patch_size = patch_size
        self.roi_size = roi_size

    def _calculate_split_points(self, size):
        split_pts = list(range(0, size, self.roi_size))
        if split_pts[-1] + self.patch_size < size:
            split_pts.append(size - self.patch_size)
        return split_pts

    def _pad_patch(self, patch):
        padded_patch = np.zeros((self.patch_size, self.patch_size, patch.shape[2]))
        padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
        return padded_patch

    def split2(self, x):
        h, w, _ = x.shape
        self.h_split_pts = self._calculate_split_points(h)
        self.w_split_pts = self._calculate_split_points(w)

        patches = [
            self._pad_patch(x[hs:hs + self.patch_size, ws:ws + self.patch_size, :])
            for hs in self.h_split_pts
            for ws in self.w_split_pts
        ]
        patches = np.array(patches)
        patches = np.transpose(patches, (0, 3, 1, 2))  # (num_patches, c, h, w)
        return patches

    def recover(self, patches, h, w, ps, roi):
        self.patch_size = ps
        self.roi_size = roi
        rec_img = np.zeros((h, w))
        self.h_split_pts = self._calculate_split_points(h)
        self.w_split_pts = self._calculate_split_points(w)
        num_h_patches = len(self.h_split_pts)
        num_w_patches = len(self.w_split_pts)


        for i, (hs_idx, ws_idx) in enumerate(
            (h_idx, w_idx)
            for h_idx in range(num_h_patches)
            for w_idx in range(num_w_patches)
        ):
            hs = self.h_split_pts[hs_idx]
            ws = self.w_split_pts[ws_idx]
            rec_hs_start = hs
            rec_hs_end = min(hs + self.roi_size, h)
            rec_ws_start = ws
            rec_ws_end = min(ws + self.roi_size, w)

            patch_hs_start = max(0, self.roi_size // 2 - (rec_hs_end - rec_hs_start) // 2)
            patch_ws_start = max(0, self.roi_size // 2 - (rec_ws_end - rec_ws_start) // 2)

            rec_img[rec_hs_start:rec_hs_end, rec_ws_start:rec_ws_end] = \
                patches[i][patch_hs_start:patch_hs_start + (rec_hs_end - rec_hs_start),
                           patch_ws_start:patch_ws_start + (rec_ws_end - rec_ws_start)]

        return rec_img

