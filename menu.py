import pygame as pg
from os import path

import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

from button import Button
from text import Text


def menu(screen, cap, num_plays):
    screen_width, screen_height = screen.get_size()
    bg_color = pg.Color("#222222")

    title_font = pg.font.Font(path.join("assets", "Orbitron-Regular.ttf"), 40)
    title = Text(screen_width // 2, 30, "Infinity Run", screen)
    title.set_font(title_font)

    instructions_image = pg.image.load(path.join("assets", "instructions.png"))

    detected_text = Text(screen_width // 2, 400, "Detected:", screen)
    detected_text.set_font(title_font)

    if num_plays == 0:
        play_button = Button(screen_width // 2, 500, 200, 50, screen, "Play")
    else:
        play_button = Button(screen_width // 2, 500, 200, 50, screen, "Play Again")
    quit_button = Button(screen_width // 2, 600, 200, 50, screen, "Quit")

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    while True:
        screen.fill(bg_color)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        label = None
        detected_text.set_text("Detected: Nothing")
        if results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2:
                detected_text.set_text("Detected: Both Hands")
            else:
                loop_trig = False
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Left':
                        detected_text.set_text("Detected: Left Hand")
                        loop_trig = True
                    elif label == 'Right':
                        detected_text.set_text("Detected: Right Hand")
                        loop_trig = True              

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return False
            if event.type == pg.MOUSEMOTION:
                mouse_pos = pg.mouse.get_pos()
                play_button.update(mouse_pos)
                quit_button.update(mouse_pos)
            if event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = pg.mouse.get_pos()
                play_button.check_clicked(mouse_pos, event.button)
                quit_button.check_clicked(mouse_pos, event.button)

                if play_button.left_clicked:
                    return True
                elif quit_button.left_clicked:
                    return False
                
        title.render()
        screen.blit(instructions_image, (screen_width // 2 - instructions_image.get_width() // 2, 100))
        detected_text.render()

        play_button.render()
        quit_button.render()

        pg.display.flip()
        cv2.imshow("Input Feed", img)