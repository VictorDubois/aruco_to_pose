
import pygame as pg


class Marker:
    def __init__(self, img, scale, size):
        self.size = size
        self.img = self.load_scale_img(img, scale, size)
        self.sprite = self.img.copy()
        self.rect = self.sprite.get_rect()

    @staticmethod
    def load_scale_img(img, scale, size):
        im = pg.image.load(img).convert_alpha()
        l = int(size * scale)
        im = pg.transform.scale(im, (l, l))
        return im

    def set_position_orientation(self, position, angle):
        self.sprite = pg.transform.rotate(self.img, angle % 360)
        self.rect = self.sprite.get_rect()
        self.rect.center = position
