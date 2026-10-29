import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomGaussianBlur(BaseTransform):
    """Blur image pixels without changing annotations or image geometry.

    Args:
        prob (float): Probability of applying blur.
        kernel_size (int): Positive odd Gaussian kernel width and height.
        sigma_range (tuple[float, float]): Bounds for sampling the blur sigma.
    """

    def __init__(self, prob=0.5, kernel_size=5, sigma_range=(0.1, 2.0)):
        if not 0 <= prob <= 1:
            raise ValueError('prob must be between 0 and 1')
        if (isinstance(kernel_size, bool) or not isinstance(kernel_size, int)
                or kernel_size <= 0 or kernel_size % 2 == 0):
            raise ValueError('kernel_size must be a positive odd integer')
        if (len(sigma_range) != 2 or not np.isfinite(sigma_range).all()
                or not 0 < sigma_range[0] <= sigma_range[1]):
            raise ValueError(
                'sigma_range must contain two positive ordered bounds')
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma_range = tuple(sigma_range)

    @cache_randomness
    def _random_disable(self):
        return np.random.rand() >= self.prob

    @cache_randomness
    def _sample_sigma(self):
        return np.random.uniform(*self.sigma_range)

    def transform(self, results):
        if self._random_disable():
            return results
        image = results['img']
        sigma = self._sample_sigma()
        blurred = cv2.GaussianBlur(
            image, (self.kernel_size, self.kernel_size), sigmaX=sigma,
            sigmaY=sigma)
        if image.ndim == 3 and blurred.ndim == 2:
            blurred = blurred[..., None]
        results['img'] = blurred
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'kernel_size={self.kernel_size}, '
                f'sigma_range={self.sigma_range})')
