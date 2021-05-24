import pygame
import numpy as np
from random import random,randrange
from typing import List


BLACK = pygame.Color("black")
WHITE = pygame.Color("white")

maxV = 3

w,h = 600,900


class Obstacle():
    def __init__(self, l, t, w , h):
        self.width = w
        self.height = h
        self.top_left = np.array((l,t))
        self.top_right = np.array((l+w,t))
        self.bottom_left = np.array((l,t+h))
        self.bottom_right = np.array((l+w,t+h))
        self.rect = pygame.Rect(l, t, w, h)

    def doesCollide(self,x, y):
        return x > self.top_left[0] and x < self.top_right[0] \
            and y > self.top_left[1] and y < self.bottom_left[1]

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)


class simpleRocket():
    def __init__(self, pos, lifespan, target, startVel=False, actions:List[np.ndarray]=[]):
        self.lifespan = lifespan
        self.target = np.array(target)
        self.fitness_bonus = 0
        self.alive = True
        if len(actions) == 0:
            self.actions = [np.array([2*random()-1,2*random()-1]) for _ in range(lifespan-1)]
            if startVel:
                self.actions = [np.array([0,-1])] + self.actions
        else:
            self.actions = actions
        self.pos = np.array((*pos,))
        self.start_pos = self.pos.copy()
        self.v = np.zeros(shape=2)
        self.fitness = 0
        self.dist = 0
        self.step = 0

    def update(self):
        if self.alive and not self.fitness_bonus:
            self.pos[0] += self.v[0]
            self.pos[1] += self.v[1]

            if np.linalg.norm(self.pos-self.target) < 10:
                self.fitness_bonus = len(self.actions) - self.step

            if len(self.actions):
                self.v += self.actions[self.step]
                self.v /= np.linalg.norm(self.v)
                self.v *= maxV
            else:
                self.v = np.zeros(shape=2)
        
        if self.pos[0] < 0 or self.pos[0] > w or self.pos[1] > h:
            self.alive=False

        self.step += 1

    def draw(self, screen):
        if self.alive and not self.fitness_bonus:
            pygame.draw.circle(screen, WHITE, self.pos, 5)

    def check_collisions(self, obs):
        for ob in obs:
            if ob.doesCollide(*self.pos):
                self.alive = False


def merge(a,b):
    c = []
    for i in range(len(a.actions)):
        c.append(a.actions[i] if i%2 else b.actions[i])
    return simpleRocket(a.start_pos, a.lifespan, a.target, actions=c)


def mutate(c):
    mutation_rate = 0.02
    for i in range(len(c.actions)):
        if random() < mutation_rate:
            c.actions[i] = np.array([2*random()-1,2*random()-1])
    return c


def natural_selection(rockets):
    maxFit = 100
    mating_pool = []
    new_pop = []
    max_d = min_d = None
    for r in rockets:
        r.dist = np.linalg.norm(r.pos - np.array(r.target))
        max_d = r.dist if max_d is None or r.dist>max_d else max_d
        min_d = r.dist if min_d is None or r.dist<min_d else min_d
        if r.fitness_bonus>0:
            min_d = 10
    
    death_count = 0
    avg_fitness = 0
    for r in rockets:
        if r.fitness_bonus>0:
            r.fitness = maxFit + r.fitness_bonus*5
        else:
            r.fitness = int(1/r.dist**2 * maxFit/(1/min_d**2)) 
        if not r.alive:
            # r.fitness=0
            death_count += 1
        avg_fitness += (r.dist)/len(rockets)
        mating_pool += [r]*r.fitness
    #     print(r.fitness, end=' ')
    # print(avg_fitness)
    pool_size = len(mating_pool)
    for _ in range(len(rockets)):
        p1 = mating_pool[randrange(pool_size)]
        p2 = mating_pool[randrange(pool_size)]
        child = merge(p1,p2)
        child = mutate(child)
        new_pop.append(child)

    return new_pop
