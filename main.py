from numba import njit, prange
from numpy import arange, array, cos, r_, pi, sin, sqrt, vstack, zeros
from random import random
from os import environ
from screeninfo import get_monitors
import pygame as pg

# hand detection
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict


@njit
def raycast(terrain, player_params, distances):
    player_x, player_y, player_angle = player_params
    angles = arange(player_angle - fov / 2, player_angle + fov / 2, dtheta)

    for i in prange(len(angles)):
        angle = angles[i]
        x, y = player_x, player_y
        dx, dy = increment * cos(angle), increment * sin(angle)
        distance = 0

        while terrain[int(x)][int(y)] == 0:
            x += dx
            y += dy
            distance += increment

        distances[i] = sqrt((x - player_x) ** 2 + (y - player_y) ** 2)


def generate_terrain(player_x, terrain):
    if player_x + generate_ahead > len(terrain):
        current_x = len(terrain)

        while current_x < player_x + generate_ahead:
            new_track = None

            if current_x % obstacle_spacing == 0:
                new_track = array([1] + [1 if random() < obstacle_probability else 0 for _ in range(track_breadth)] + [1])
            else:
                new_track = array([1] + [0 for _ in range(track_breadth)] + [1])

            terrain = vstack((terrain, new_track))
            current_x += 1
        
    return terrain


def get_player_speed(distance):
    return player_base_speed + (max_additional_speed * distance ** 2) / (half_maximum ** 2 + distance ** 2)


def gameloop(screen):
    global res_downscale, res, dtheta

    dt = 1.0 / ideal_fps
    screen_width, screen_height = screen.get_size()

    ground_color = pg.Color("#667e2c")
    sky_color = pg.Color('#92b4f4')
    wall_color = pg.Color("#dd1c1a")

    # player params
    player_x = 1
    player_y = 2.5
    player_angle = 0
    player_move_speed = 5

    terrain = array([
        [1 for _ in range(track_breadth + 2)]
    ])

    for _ in range(50):
        terrain = vstack((terrain, [1] + [0 for _ in range(track_breadth)] + [1]))

    distances = zeros(screen_width)
    
    font = pg.font.SysFont("Arial", 30)
    left_text = font.render("Left", True, pg.Color("black"))
    right_text = font.render("Right", True, pg.Color("black"))

    pg.mixer.music.load("bg_music.mp3")
    timer = 0

    # hand detection model
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    while True:
        clock.tick(ideal_fps)

        keys_pressed = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return

            if event.type == pg.QUIT:
                return
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
            
        # move player forward
        player_x += get_player_speed(player_x) * dt

        # input from hands
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        label = None
        if results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2:
                pass
            else:
                for i in results.multi_handedness:
                    # Return whether it is Right or Left Hand
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Left':
                        player_y -= player_move_speed * dt
                    if label == 'Right':
                        player_y += player_move_speed * dt

        # constrain player to track
        if player_y < 1.5:
            player_y = 1.5
        if player_y > track_breadth + 0.5:
            player_y = track_breadth + 0.5

        # game over
        if terrain[int(player_x)][int(player_y)] == 1:
            print("Game Over")
            return

        terrain = generate_terrain(player_x, terrain)

        # draw sky and ground
        screen.fill(sky_color)
        pg.draw.rect(screen, ground_color, (0, screen_height // 2, screen_width, screen_height // 2))

        # draw walls
        raycast(terrain, (player_x, player_y, player_angle), distances)
        for i, distance in enumerate(distances):
            if abs(distance) < 0.001:
                continue

            if distance != render_distance:
                height = height_multiplier / distance

                red_shade = shader_min + int(shader_interval / (1 + distance) ** shader_pow)
                wall_color = pg.Color(red_shade, 32, 64)

                pg.draw.rect(screen, wall_color, (i * res_downscale, screen_height // 2 - height, res_downscale, 2 * height))

        dt = clock.get_time() / 1000
        timer += dt

        current_fps = clock.get_fps()
        fps_text = font.render(str(int(current_fps)), True, pg.Color("black"))
        screen.blit(fps_text, (0, 0))

        if label == 'Left':
            screen.blit(left_text, (screen_width // 2 - 100, screen_height // 2))
        if label == 'Right':
            screen.blit(right_text, (screen_width // 2 + 100, screen_height // 2))

        pg.display.flip()

        if timer > 5 and not pg.mixer.music.get_busy():
            pg.mixer.music.play(-1)

        cv2.imshow('Image', img)


if __name__ == '__main__':
    monitor_width, monitor_height = get_monitors()[0].width, get_monitors()[0].height  

    cap = cv2.VideoCapture(0)
    r, frame = cap.read()
    webcam_width, webcam_height = frame.shape[1], frame.shape[0]
    padding = 50

    pygame_window_width = monitor_width - webcam_width - padding
    height_ratio = 1.62
    pygame_window_height = int(pygame_window_width / height_ratio)
    environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (webcam_width + padding, padding)

    pg.init()
    clock = pg.time.Clock()
    screen = pg.display.set_mode((pygame_window_width, pygame_window_height))
    screen_width, screen_height = screen.get_size()
    pg.mouse.set_visible(False)

    # difficulty parameters
    player_base_speed = 10
    max_additional_speed = 20
    half_maximum = 1000

    # display parameters
    ideal_fps = 60
    res_downscale = 2
    fov = pi / 4
    increment = 0.02
    height_multiplier = 500
    render_distance = 20
    res = screen_width // res_downscale
    dtheta = fov / res

    # shaders
    shader_min = 80
    shader_interval = 255 - shader_min
    shader_pow = 0.5

    # gameplay parameters
    track_breadth = 7
    obstacle_spacing = 30
    obstacle_probability = 0.6
    generate_ahead = 200

    gameloop(screen)