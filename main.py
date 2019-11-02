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
        '''Update given a time in seconds'''
        self.v += dt * self.a
        self.x += dt * self.v

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
        pg.draw.circle(surf, (255,255,255), (x,y), self.r)
        pg.draw.line(surf, (255,255,255), (x,y), (lx,ly), 2)


class Bullets:
    x = np.empty(shape=(0, 2), dtype=np.float32)
    v = np.empty(shape=(0, 2), dtype=np.float32)
    a = np.empty(shape=(0, 2), dtype=np.float32)

    def add(self, x, v=[0.0, 0.0], a=[0.0, 0.0]):
        '''Add a bullet and return its index.'''
        self.x = np.append(self.x, [x], axis=0)
        self.v = np.append(self.v, [v], axis=0)
        self.a = np.append(self.a, [a], axis=0)
        return len(self.x) - 1

    def rem(self, index):
        self.x = np.delete(self.x, index, axis=0)
        self.v = np.delete(self.v, index, axis=0)
        self.a = np.delete(self.a, index, axis=0)

    def update(self, dt):
        self.v += dt * self.a
        self.x += dt * self.v

    def draw(self, bg):
        for pos in self.x:
            w, h = bg.get_size()
            x = int(pos[0] * w)
            y = int((1-pos[1]) * h)
            pg.draw.circle(bg, (255,255,255), (x, y), 3)


class Game:
    def __init__(self, dim):
        pg.init()

        self.prev_t = time.time()
        self.curr_t = None
        self.dt = 0

        # Intialize surface
        self.screen = pg.display.set_mode(dim)
        self.bg = pg.Surface(self.screen.get_size()).convert()

    def step(self):
        self.curr_t = time.time()
        self.dt = self.curr_t - self.prev_t
        self.prev_t = self.curr_t

    def clear(self):
        self.bg.fill((0, 0, 0))

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        pg.display.flip()


if __name__ == '__main__':
    g = Game(dim=(900, 900))
    p = Player(
        m=1.0,
        x=(0.5,0.5),
        v=(0.0,0.0),
        a=(0.0,0.0),
        angle=math.pi,
        r=10,
    )
    b = Bullets()

    key_down = {}
    while True:
        g.step() # Compute dt
        p.update(g.dt)
        b.update(g.dt)

        # Event loop
        for e in pg.event.get():
            if e.type == QUIT:      exit(0)

            if e.type == KEYDOWN:
                if e.key == K_SPACE:
                    b.add(
                        x=p.x,
                        v=(p.v + [
                            .2 * math.cos(p.angle),
                            .2 * math.sin(p.angle),
                        ]),
                        a=[0, 0]
                    )

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

        if key_down.get(pg.K_LEFT):  p.angle += math.pi*g.dt
        if key_down.get(pg.K_RIGHT): p.angle -= math.pi*g.dt

        g.clear()
        p.draw(g.bg)
        b.draw(g.bg)

        g.draw()
        time.sleep(.0166) # 60FPS
