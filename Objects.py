import pygame
import numpy as np
pygame.font.init()
font = pygame.font.SysFont("ariel", 24)

colors_list = [(c) for c, v in pygame.color.THECOLORS.items()]

class Box():
    def __init__(self, pos, BOX_LENGTH):
        self.pos = pos
        self.rect = pygame.Rect(pos[0], pos[1], BOX_LENGTH, BOX_LENGTH)
        self.val = 0

    def draw(self, surface):
        color = (int(255*self.val), 0, 0)
        pygame.draw.rect(surface, color, self.rect)

class Neuron:
    def __init__(self, x, y, val=0, bias=False):
        self.x = x
        self.y = y

    def draw(self, surface, color=None):
        if not color:
            color = (0, 255, 0)
        pygame.draw.circle(surface, color, (self.x, self.y), 10, 2)

class Connection:
    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.val = 0
    
    def draw(self, surface):
        pygame.draw.line(surface, colors_list[int(self.val*len(colors_list))], (self.first.x, self.first.y), (self.second.x, self.second.y), 1)
