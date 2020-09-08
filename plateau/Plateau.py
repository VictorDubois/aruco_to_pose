
from .Marker import Marker
import os
import pygame as pg


class Plateau:
    def __init__(self, dim, resources, scale, origin_id, center):
        pg.init()
        self.screen = pg.display.set_mode(dim)
        self.scale = scale
        self.plateau, self.markers = self.load_resources(resources)
        self.origin_id = origin_id
        self.center = center

    @staticmethod
    def find_resources(folder):
        assert os.path.isdir(folder), 'The resource folder does not exist'
        resources = []
        for root, _, files in os.walk(folder, topdown=False):
            for name in files:
                resources.append(os.path.join(root, name))
        return resources

    def load_resources(self, folder):
        plateau = None
        markers_dict = {}
        resources = self.find_resources(folder)
        for name in resources:
            head, tail = os.path.split(name)
            if tail.startswith('plateau'):
                plateau = pg.image.load(name).convert_alpha()
            elif tail.startswith('marker'):
                id_ = int(tail.split('.')[0].replace('marker', ''))
                size = 0.07
                if id_ == 42:
                    size = 0.1
                markers_dict[id_] = Marker(name, self.scale, size)
        return plateau, markers_dict

    def update(self, ids=None, posrot=None):
        self.screen.blit(self.plateau, (0, 0))
        if ids is not None:
            for id in ids:
                if posrot.get(id):
                    p, r = posrot[id]
                    self.markers[id].set_position_orientation(p, r)
                    self.screen.blit(self.markers[id].sprite, self.markers[id].rect)
        pg.display.flip()

    def draw_marker(self, id, pos, rot):
        self.markers[id].set_position_orientation(pos, rot)
        self.screen.blit(self.markers[id].sprite, self.markers[id].rect)
        pg.display.flip()



