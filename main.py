#! /usr/bin/env python

import time
import math
import numpy as np
import scipy
import pygame as pg
from pygame.locals import *


class Player:
    def __init__(self, m, x, v, a, angle, r):
        self.m = m
        self.x = np.array([x[0],x[1]])
        self.v = np.array([v[0],v[1]])
        self.a = np.array([a[0],a[1]])
        self.angle = angle
        self.r = r

    def update(self, dt: float):
        """Update given a time in seconds"""
        self.x += dt * self.v + .5 * (dt ** 2) * self.a
        self.v += dt * self.a

    def draw(self, surf):
        w, h = surf.get_size()
        phi = self.angle
        l = self.r + 5 # Length of cannon in pixels

        # End of line position
        lp = self.x + [l*math.cos(phi)/w, l*math.sin(phi)/h]

        # Pixel positions
        x = int(w * self.x[0]) 
        y = int(h * (1 - self.x[1]))
        lx = int(w * lp[0]) 
        ly = int(h * (1 - lp[1]))

        # Draw the circle and the line
        pg.draw.circle(bg, (255,255,255), (x,y), self.r)
        pg.draw.line(bg, (255,255,255), (x,y), (lx,ly), 2)


if __name__ == '__main__':
    # Initialize
    pg.init()
    screen = pg.display.set_mode((900, 900))

    # Intialize surface
    bg = pg.Surface(screen.get_size())
    bg = bg.convert()

    # Create player
    p = Player(
        m=1.0,
        x=(0.5,0.5),
        v=(0.0,0.0),
        a=(0.0,0.0),
        angle=math.pi,
        r=10,
    )

    prev_t = time.time()
    curr_t = None
    dt = 0

    key_down = {}
    while True:
        # Event loop
        for e in pg.event.get():
            if e.type == QUIT:
                exit(0)
            elif e.type == KEYDOWN:
                key_down[e.key] = True
            elif e.type == KEYUP:
                key_down[e.key] = False


        if key_down.get(pg.K_UP):
            p.v = 0.3 * np.array([
                math.cos(p.angle),
                math.sin(p.angle),
            ])
        elif key_down.get(pg.K_DOWN):
            p.v = -0.3 * np.array([
                math.cos(p.angle),
                math.sin(p.angle),
            ])
        else:
            p.v = np.zeros(2)

        if key_down.get(pg.K_LEFT): p.angle += math.pi/250
        if key_down.get(pg.K_RIGHT): p.angle -= math.pi/250

        # Update loop
        curr_t = time.time()
        dt = curr_t - prev_t
        prev_t = curr_t

        p.update(dt)


        # Draw loop
        bg.fill((0, 0, 0))
        p.draw(bg)

        screen.blit(bg, (0, 0))
        pg.display.flip()
        time.sleep(.00166)
