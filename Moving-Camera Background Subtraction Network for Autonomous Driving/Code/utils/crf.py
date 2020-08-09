import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


class DenseCRF(object):
    def __init__(self, iter_max, scale, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.scale = scale
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape # softmax output

        U = utils.unary_from_softmax(probmap, scale=self.scale)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w)

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q



class xyt_DenseCRF(object):
    def __init__(self, iter_max, scale, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_t_std):
        self.iter_max = iter_max
        self.scale = scale
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_t_std = bi_t_std
        
    def __call__(self, label, probmap):
        C, H, W = probmap.shape  # softmax layer output

        U = utils.unary_from_softmax(probmap, scale=self.scale)
        U = np.ascontiguousarray(U)
        label = np.ascontiguousarray(label)

        # create DenseCRF dims with (int N, int C)
        d = dcrf.DenseCRF(label.shape[0]*label.shape[1], C)
        d.setUnaryEnergy(U)
        
        # This creates the previousmasks-indepedent features and then add them to the CRF
        feats = utils.create_pairwise_gaussian(sdims=self.pos_xy_std, shape=label.shape[:2])
        d.addPairwiseEnergy(feats, compat=self.pos_w,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
                            
        # This creates the previousmasks-dependent features and then add them to the CRF
        feats = utils.create_pairwise_bilateral(sdims=self.bi_xy_std, schan=self.bi_t_std,
                                          img=label, chdim=2)
        d.addPairwiseEnergy(feats, compat=self.bi_w,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
                            
        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q