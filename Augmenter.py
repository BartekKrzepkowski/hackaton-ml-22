import numpy as np
from imgaug import augmenters as iaa


sometimes = lambda aug: iaa.Sometimes(0.2, aug)

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            sometimes(iaa.Add((-40, 40))),
            sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))),
            sometimes(iaa.Cutout(nb_iterations=2)),
            sometimes(iaa.Dropout(p=(0, 0.2))),
            sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))),
            sometimes(iaa.MotionBlur(k=15)),
            sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))),
            sometimes(iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)),
            sometimes(iaa.GammaContrast((0.5, 2.0))),
            sometimes(iaa.Fliplr(0.5)),
            sometimes(iaa.Flipud(0.5)),
            sometimes(iaa.Affine(scale=(0.5, 1.5))),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.15)))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)