# Example file showing a circle moving on screen
import pygame
import pygame.gfxdraw

# pygame setup
pygame.init()
screen = pygame.display.set_mode((500.5, 500.5))
clock = pygame.time.Clock()
running = True
dt = 0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill((211, 179, 104))
    for i in range(100, 500, 100):
        pygame.gfxdraw.hline(screen, 0, 500, i, (0, 0, 0))
        pygame.gfxdraw.vline(screen, i, 0, 500, (0, 0, 0))

    for i in range(0, 3):
        for j in range(0, 3 - i):
            # white circles
            pygame.gfxdraw.aacircle(
                screen, 50 + (i * 100), 50 + (j * 100), 40, (255, 255, 255))
            pygame.gfxdraw.filled_circle(
                screen, 50 + (i * 100), 50 + (j * 100), 40, (255, 255, 255))
            # black circles
            pygame.gfxdraw.aacircle(
                screen, 450 - (i * 100), 450 - (j * 100), 40, (0, 0, 0))
            pygame.gfxdraw.filled_circle(
                screen, 450 - (i * 100), 450 - (j * 100), 40, (0, 0, 0))

    # pygame.gfxdraw.aacircle(
    #     screen, int(
    #         player_pos[0]), int(
    #         player_pos[1]), 10, (255, 255, 255))
    # pygame.gfxdraw.filled_circle(
    #     screen, int(
    #         player_pos[0]), int(
    #         player_pos[1]), 10, (255, 255, 255))

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_w]:
    #     player_pos.y -= 300 * dt
    # if keys[pygame.K_s]:
    #     player_pos.y += 300 * dt
    # if keys[pygame.K_a]:
    #     player_pos.x -= 300 * dt
    # if keys[pygame.K_d]:
    #     player_pos.x += 300 * dt

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
