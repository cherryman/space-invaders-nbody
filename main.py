#! /usr/bin/env python

import os
import sys
import time

# Disable pygame cancer
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import math
import numpy as np
import pygame as pg
from pygame.locals import *

from nbody import takeStepRK4


IMAGES = {
    'player': './assets/player.png',
    'enemy': './assets/enemy.png',
    'bullet': './assets/bullet.png',
    'bg': './assets/bg.jpg',
    'heart': './assets/heart.png',
    'clock': './assets/clock.png',
}

GREEN = (0, 255, 0)
BLUE = (0, 0, 128)


class Entity(pg.sprite.Sprite):
    def __init__(self, sprite, size, hp, m, x, v, a, angle):
        pg.sprite.Sprite.__init__(self)

        self.m = m
        self.x = np.array([x[0],x[1]])
        self.v = np.array([v[0],v[1]])
        self.a = np.array([a[0],a[1]])
        self.angle = angle
        self.hp = hp
        self.r = size/2

        self.image = pg.image.load(IMAGES[sprite])
        self.image = pg.transform.scale(self.image, (size, size))
        self.rect = self.image.get_rect()

    def draw(self, surf):
        w, h = surf.get_size()

        x = int(w * self.x[0])
        y = int(h * (1 - self.x[1]))

        img = pg.transform.rotate(
            self.image,
            math.degrees(self.angle) - 90.)
        img_w, img_h = img.get_size()
        surf.blit(img, (x - img_w/2, y - img_h/2))

    def update(self, game):
        '''Update given a time in seconds'''
        self.v += game.dt * self.a
        self.x += game.dt * self.v
        self.rect.center = (self.x[0]*game.pw/game.w,
                            self.x[1]*game.ph/game.h)


class Enemy(Entity):
    def __init__(self, size, hp, m, x, v, a, angle):
        super().__init__('enemy', size, hp, m, x, v, a, angle)


class Player(Entity):
    def __init__(self, size, hp, m, x, v, a, angle):
        super().__init__('player', size, hp, m, x, v, a, angle)

    def update(self, game):
        if pg.key.get_pressed()[pg.K_UP]:
            self.a = 0.35 * np.array([
                math.cos(self.angle),
                math.sin(self.angle),
            ])
        elif pg.key.get_pressed()[pg.K_DOWN]:
            self.a = -0.35 * np.array([
                math.cos(self.angle),
                math.sin(self.angle),
            ])

        if pg.key.get_pressed()[pg.K_LEFT]:
            self.angle += 1.5*math.pi*game.dt
        if pg.key.get_pressed()[pg.K_RIGHT]:
            self.angle -= 1.5*math.pi*game.dt

        super().update(game)
        self.a = (-4. / self.m) * self.v

        if self.rect.left <= 0 or self.rect.right >= game.pw:
            self.v[0] *= -1
        if self.rect.bottom <= 0 or self.rect.top >= game.ph:
            self.v[1] *= -1

class Bullets:
    x = np.empty(shape=(0, 2), dtype=np.float32)
    v = np.empty(shape=(0, 2), dtype=np.float32)
    m = np.empty(shape=(0, 1), dtype=np.float32)
    r = np.empty(shape=(0, 1), dtype=np.int32)

    img = pg.image.load(IMAGES['bullet'])

    def add(self, m, x, v=[0.0, 0.0], a=[0.0, 0.0]):
        '''Add a bullet and return its index.'''
        self.x = np.append(self.x, [x], axis=0)
        self.v = np.append(self.v, [v], axis=0)

        self.m = np.append(self.m, m)
        self.r = np.append(self.r, int(round(math.sqrt(m) * 10.)))
        return len(self.x) - 1

    def rem(self, index):
        self.x = np.delete(self.x, index, axis=0)
        self.v = np.delete(self.v, index, axis=0)

        self.m = np.delete(self.m, index, axis=0)
        self.r = np.delete(self.r, index, axis=0)

    def update(self, game):
        self.rem(
            np.where((self.x[:,0] < -.5) | (self.x[:,0] > 1.5)
                   | (self.x[:,1] < -.5) | (self.x[:,1] > 1.5))[0]
        )

    def draw(self, surf):
        w, h = surf.get_size()

        for pos, rad in zip(self.x, self.r):
            img = pg.transform.scale(self.img, (rad, rad))
            img_w, img_h = img.get_size()

            x = int(pos[0] * w + img_w/2)
            y = int((1-pos[1]) * h + img_h/2)
            surf.blit(img, (x, y))


class Game:
    def __init__(self, dim):
        pw, ph = dim
        pg.init()
        pg.display.set_caption('Space Invaders NBody')

        self.w, self.h = 1., 1.
        self.pw, self.ph = dim

        self.font = pg.font.Font('freesansbold.ttf', 36)

        self.prev_t = time.time()
        self.curr_t = None
        self.dt = 0
        self.stepCount = -1

        self.img_clock = pg.image.load(IMAGES['clock'])
        self.img_clock = pg.transform.scale(self.img_clock, (pw//10, pw//10))

        self.img_heart = pg.image.load(IMAGES['heart'])
        self.img_heart = pg.transform.scale(self.img_heart, (pw//10, pw//10))

        self.image = pg.image.load(IMAGES['bg'])
        self.screen = pg.display.set_mode(dim)
        self.bg = pg.Surface(self.screen.get_size()).convert()

        self.time_started = time.time()
        self.time_elapsed = 0

    def step(self):
        self.curr_t = time.time()
        self.dt = self.curr_t - self.prev_t
        self.prev_t = self.curr_t
        self.stepCount += 1

        # self.time_elapsed = time.time() - self.time_started

    def clear(self):
        self.bg.fill((0, 0, 0))
        self.bg.blit(self.image, (0, 0))

    def draw(self):
        clk_size = self.img_clock.get_rect()

        self.screen.blit(self.bg, (0, 0))
        # self.screen.blit(self.img_clock, (2, 2))
        # self.screen.blit(self.img_heart, (2, 2 + clk_size.h))
        pg.display.flip()


def main(g):
    p = Player(
        size=50,
        m=1.0,
        x=(0.5,0.5),
        v=(0.0,0.0),
        a=(0.0,0.0),
        angle=math.pi,
        hp=6000,
    )
    b = Bullets()
    enn = []

    frame = 0
    while True:
        g.step() # Compute dt

        b.x, b.v = takeStepRK4(g.dt, b.x, b.v, b.m)

        b.update(g)
        p.update(g)
        for en in enn: en.update(g)

        if p.hp <= 0: return

        if g.stepCount % 75 == 0: print('n particles: %s; FPS: ' % len(b.x), int(1/g.dt))

        # Event loop
        ks = 0
        for e in pg.event.get():
            if e.type == QUIT: sys.exit()
            if e.type == KEYDOWN: ks = g.stepCount

        if (g.stepCount-ks) % 4 == 0 and pg.key.get_pressed()[pg.K_SPACE]:
            b.add(
                m=2.0,
                x=(p.x + [
                    1.4 * p.r * math.cos(p.angle)/width,
                    1.4 * p.r * math.sin(p.angle)/height,
                ]),
                v=(p.v + [
                    .3 * math.cos(p.angle),
                    .3 * math.sin(p.angle),
                ]),
                a=[0, 0]
            )

        #Random chance of ennemy spawning
        spawn_chance = 0.02
        
        if np.random.rand() < spawn_chance:
            f = Enemy(
                size=50,
                m=10.0,
                x=(np.random.rand(),1),
                v=(0,-max(abs(-np.random.rand()/10),0.05)),
                a=(0.0,0.0),
                angle=math.pi,
                hp=2,
            )
            ang = math.atan2(p.x[1] - f.x[1], p.x[0] - f.x[0])
            f.angle = ang
            enn.append(f)
        
        if frame % 25 == 0:
            for en in enn:
                ang = math.atan2(p.x[1] - en.x[1], p.x[0] - en.x[0])
                en.angle = ang + np.random.rand()*2*math.pi/8
                b.add(
                    m=1.0,
                    x=en.x + [1.5*en.r*math.cos(en.angle)/width,
                              1.5*en.r*math.sin(en.angle)/height],
                    v=(en.v + [
                        .2 * math.cos(en.angle),
                        .2 * math.sin(en.angle),
                    ]),
                    a=[0, 0]
                )

        for en in enn:
            if en.x[1] < 0 or en.x[0] < 0\
                    or en.x[1] > 1 or en.x[0] > 1:
                enn.remove(en)

        # Check if there is a collision between player
        # and any particle in x or y
        test = abs(b.x - p.x)*g.bg.get_size() < p.r
        collis = np.array([x.all() for x in test])

        b_rem = np.where(collis)[0]
        if len(b_rem) > 0:
            b.rem(b_rem)
            p.hp -= 1

        # Check if enemies get hit
        for en in enn:
            # Check if there is a collision between player and
            # any particle in x or y
            test_enn = abs(b.x - en.x)*g.bg.get_size() < en.r
            collis_enn = np.array([x.all() for x in test_enn])
            b_rem = np.where(collis_enn)[0] 

            if len(b_rem) > 0:
                b.rem(b_rem)
                en.hp -= 1
            if en.hp <= 0:
                enn.remove(en)
                
        pg.display.update()
        g.clear()
        p.draw(g.bg)
        b.draw(g.bg)
        for en in enn: en.draw(g.bg)
        g.draw()

        frame += 1
        time.sleep(max(1/60 - g.dt, 0)) # 60FPS


if __name__ == '__main__':
    width, height = 1000, 1000
    g = Game(dim=(width, height))

    main(g)

    text = g.font.render('You died', True, (255, 0, 0))
    textRect = text.get_rect()
    textRect.center = (width/2, height/2)

    g.bg.fill((0, 0, 0))
    g.bg.blit(text, (textRect.x, textRect.y))
    g.draw()
    while True:
        time.sleep(10)
