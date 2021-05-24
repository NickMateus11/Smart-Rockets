from smartRockets import simpleRocket, natural_selection, Obstacle
import pygame


w, h = 600, 900
pygame.init()
screen = pygame.display.set_mode((w, h))

BLACK = pygame.Color("black")
WHITE = pygame.Color("white")
GREEN = pygame.Color("green")

clock = pygame.time.Clock()

def main():

    goal_coords = (w//2,20)
    rocket_pop = 50
    steps = 500
    R = [simpleRocket((w//2,h-20), steps, target=goal_coords, startVel=True) for _ in range(rocket_pop)]
    gen = 1
    count = 0

    obstacles = [
        Obstacle(0,h//2,2*w/3,20),
        Obstacle(w//2,h//4,w//2,20)
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        for r in R:
            r.draw(screen)
            r.update()
            r.check_collisions(obstacles)
        
        for ob in obstacles:
            ob.draw(screen)

        count += 1
        
        if count == steps:
            R = natural_selection(R)
            count = 0
            gen += 1

        pygame.draw.rect(screen, GREEN, pygame.Rect(goal_coords[0]-20, 0, 40, 20))
        pygame.display.flip()
        # clock.tick(60)


if __name__ == "__main__":
    main()
