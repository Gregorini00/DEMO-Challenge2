# Act Component: Provide feedback to the user

import mediapipe as mp
import cv2
import numpy as np
import random # I don't think we are using it (???)
import pyttsx3
import pygame
from pathlib import Path
import time


# MICHELA:
# I added some comments to explain the things I've added to the code "unsupervised" :) 



# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self):
        # Balloon size and transition tracking for visualization
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent
        images_dir = project_root / "images"
        audio_dir  = project_root / "audio"


        self.win_left = "Sport Coaching Program"
        self.win_right = "Daily Activities"


        # initializing the first state
        self.mode = "choose"
        self.next_mode = None
        self.choose_img = cv2.imread(str(images_dir / "menu_choose.png"))


        # gardening images
        self.leaf_img = cv2.imread(str(images_dir / "leave.png"), cv2.IMREAD_UNCHANGED)
        self.can_img = cv2.imread(str(images_dir / "can.png"), cv2.IMREAD_UNCHANGED)
        self.carrot_img = cv2.imread(str(images_dir / "carrot.png"), cv2.IMREAD_UNCHANGED)

        # cooking images
        self.stew_img = cv2.imread(str(images_dir / "stew.png"), cv2.IMREAD_UNCHANGED)
        self.spoon_img = cv2.imread(str(images_dir / "spoon.png"), cv2.IMREAD_UNCHANGED)
        self.food_img = cv2.imread(str(images_dir / "food.png"), cv2.IMREAD_UNCHANGED)
        
        # other images
        self.choose_img   = cv2.imread(str(images_dir / "choose.png"))
        self.count_ready  = cv2.imread(str(images_dir / "ready.png"))
        self.count_3      = cv2.imread(str(images_dir / "3.png"))
        self.count_2      = cv2.imread(str(images_dir / "2.png"))
        self.count_1      = cv2.imread(str(images_dir / "1.png"))
        self.count_go     = cv2.imread(str(images_dir / "go.png"))

        self.instr_gardening = cv2.imread(str(images_dir / "gardening_instruction.png"))
        self.instr_cooking = cv2.imread(str(images_dir / "cooking_instruction.png"))

        # "good job!"
        self.good_job_duration = 1.5
        self.good_job_active = False
        self.good_job_end = 0.0

        self.good_job_img = cv2.imread(str(images_dir / "good_job.png"))


        self.transition_count = 0
        self.round_count = 0
        self.max_transitions = 6
        self.total_transitions = 0


        self.engine = pyttsx3.init()


        intro = audio_dir / "1 Introduction Part 1.wav"

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        if intro.exists():
            pygame.mixer.music.load(str(intro))
            pygame.mixer.music.play()
        else:
            print(f"[Act] Audio not found: {intro}")


        self.motivating_index = 0


        self.feedback_files = {
            'motivating': [
                str(audio_dir / 'goodjob1.wav'),
                str(audio_dir / 'goodjob2.wav'),
                str(audio_dir / 'goodjob3.wav'),
                str(audio_dir / 'transition.wav'),
                str(audio_dir / 'almost2.wav'),
                str(audio_dir / 'almost3.wav'),
            ]
        }

        #----------
        # countdown:


        # MICHELA:

        # self.countdown_steps is a list of couples (lable: str, duration: float)
        # with each lable we select a different image to show for the countdown
        # and how many seconds to show it

        self.countdown_steps = [
            ("instruction", 3.0),
            ("ready", 0.5),
            ("3", 1.0),
            ("2", 1.0),
            ("1", 1.0),
            ("go", 0.5),
        ]



        # MICHELA:

        # self.countdown_index = 0
        # index of the current step in the countdown_steps array
        
        # self.countdown_step_start = None
        # it's a timestamp for when the current countdown step started
        # (None = not started yet)

        self.countdown_index = 0
        self.countdown_step_start = None

    
    # MICHELA:

    # def start_countdown(self, next_mode: str):
    # method to start the countdown and also remembers which mode to enter
    # after it finishes
    
    def start_countdown(self, next_mode: str):
        if next_mode not in ("choose",  "gardening", "cooking"):
            return      # before the return, I previously added a print for debugging purpose
                        # (so if we add other modes we can see if everything's fine)

        self.next_mode = next_mode
        self.mode = "countdown"
        self.countdown_index = 0
        self.countdown_step_start = time.time()
    

    # MICHELA
    # def set_mode(self, mode: str):
    # - method used to set the mode (= "switch" the state)
    # - if you enter a task like "gardening" or "cooking", resent counters

    def set_mode(self, mode: str):
            if mode not in ("choose", "countdown", "gardening", "cooking"):
                return
            
            self.mode = mode
            

            if mode in ("gardening", "cooking"):

                self.transition_count = 0
                self.round_count = 0
                self.total_transitions = 0


    # to start the "good job" screen:
    def good_job(self):
        self.good_job_active = True
        self.good_job_end = time.time() + self.good_job_duration




               

        # Handles balloon inflation and reset after explosion

    def play_audio(self, file_path):
        if not file_path:
            return
        p = Path(file_path)
        if p.exists() and not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(str(p))
            pygame.mixer.music.play()



    def handle_balloon_inflation(self):
        """
        Increases the size of the balloon with each successful repetition.
        """
        self.transition_count += 1
        self.total_transitions += 1
        # Calculate the current round
        self.round_count = self.total_transitions // self.max_transitions
        clip = self.feedback_files['motivating'][self.motivating_index]
        self.motivating_index = (self.motivating_index + 1) % len(self.feedback_files['motivating'])
        self.play_audio(clip)




    def reset_balloon(self):
        """
        Resets the balloon after it explodes.
        """

        self.transition_count = 0
        # Play "end" audio before restarting
        end_audio = str((Path(__file__).resolve().parent.parent / "audio" / "end.wav"))  # usa audio_dir se preferisci
        self.play_audio(end_audio)

        # Create explosion fragments with random sizes and positions

        self.set_mode("choose")


    def visualize_smth(self):
        if self.good_job_active:
            self.visualize_good_job()
            return

        if self.mode == "choose":
            self.visualize_menu()
        elif self.mode == "countdown":
            self.visualize_countdown()
        elif self.mode == "gardening":
            self.visualize_balloon()
        elif self.mode == "cooking":
            self.visualize_cooking()



    def visualize_menu(self):
        
        """ if self.mode == "choose":
            img = np.full((500, 500, 3), (220, 245, 245), dtype=np.uint8)
            
            #title
            cv2.putText(img, "What would you like to start with?", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            
            #instructions/modes
            cv2.putText(img, "Press G  -  Gardening", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, "Press C  -  Cooking", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, "Press R  -  Back to this menu", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            
            #subtitle
            cv2.putText(img, "Make your choice!", (20, 480),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

            cv2.imshow(self.win_right, img)
            #cv2.waitKey(1) """
        
        if self.mode != "choose":
            return
        img = self.choose_img
        cv2.imshow(self.win_right, img)



    # MICHELA:

    # def visualize_countdown(self):
    # - it shows one image at a time (ready->3->2->1->go)
    #   for the assigned number of seconds
    # - when it's done, it goes to the selected mode

    def visualize_countdown(self):
        if self.mode != "countdown":
            return

        now = time.time()
        if self.countdown_step_start is None:
            self.countdown_step_start = now

        label, duration = self.countdown_steps[self.countdown_index]
        img = self.get_countdown_img(label)

        cv2.imshow(self.win_right, img)

        # with this if, we check if the current image has been shown long enough
        if (now - self.countdown_step_start) >= duration:

            # if yes, move to the next image in the countdown
            self.countdown_index += 1

            # reset the timer for the new image
            self.countdown_step_start = now

            # if we "run past" the last image...
            if self.countdown_index >= len(self.countdown_steps):

                # goes to the selected mode
                self.set_mode(self.next_mode)


    # MICHELA:
    # def get_countdown_img(self, label: str):
    # takes a label and returns the matching countdown image

    def get_countdown_img(self, label: str):

        if label == "instruction":
            if self.next_mode == "gardening":
                return self.instr_gardening
            elif self.next_mode == "cooking":
                return self.instr_cooking
                

        label_to_image = {
            "ready": self.count_ready,
            "3": self.count_3,
            "2": self.count_2,
            "1": self.count_1,
            "go": self.count_go,
        }

        return label_to_image.get(label) 


    def visualize_balloon(self):

        if self.transition_count >= self.max_transitions:
            self.good_job()
            return

        if self.transition_count >= 6:
            self.reset_balloon()
            return
        
        img = np.full((500, 500, 3), (220, 245, 245), dtype=np.uint8)
        
        if self.transition_count < 4:
            display_image = self.leaf_img
        else:
            display_image = self.carrot_img


        # Resize the image based on transition_count (like balloon size)
        scale = 0.15 + (self.transition_count * 0.08)
        new_w = int(display_image.shape[1] * scale)
        new_h = int(display_image.shape[0] * scale)

        # Clamp size so it never exceeds canvas
        new_w = min(new_w, img.shape[1])
        new_h = min(new_h, img.shape[0])

        # Resize properly after clamping
        resized_img = cv2.resize(display_image, (new_w, new_h))

        # Center placement
        x_offset = (img.shape[1] - new_w) // 2
        y_offset = (img.shape[0] - new_h) // 2
        # Split alpha and color channels
        alpha = resized_img[:, :, 3] / 255.0  # alpha channel
        rgb_carrot = resized_img[:, :, :3]  # RGB channels

        # Alpha blending into 3-channel canvas
        for c in range(3):
            img[y_offset:y_offset + resized_img.shape[0], x_offset:x_offset + resized_img.shape[1], c] = (
                    alpha * rgb_carrot[:, :, c] + (1 - alpha) * img[y_offset:y_offset + resized_img.shape[0],
                                                                x_offset:x_offset + resized_img.shape[1], c]
            )
        # Overlay text
        cv2.putText(img, f'Times watered: {self.transition_count}', (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Carrots grown: {self.round_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.imshow(self.win_right, img)
        #cv2.waitKey(1)


    def visualize_cooking(self):

        if self.transition_count >= self.max_transitions:
            self.good_job()
            return
        
        if self.transition_count >= 6:
            self.reset_balloon()
            return
        
        img = np.full((500, 500, 3), (220, 245, 245), dtype=np.uint8)

        # Choose which image to display
        if self.transition_count < 4:
            display_image = self.stew_img  # first image
        else:
            display_image = self.food_img



        # Resize the image based on transition_count (like balloon size)
        scale = 0.15 + (self.transition_count * 0.08)
        new_w = int(display_image.shape[1] * scale)
        new_h = int(display_image.shape[0] * scale)

        # Clamp size so it never exceeds canvas
        new_w = min(new_w, img.shape[1])
        new_h = min(new_h, img.shape[0])

        # Resize properly after clamping
        resized_img = cv2.resize(display_image, (new_w, new_h))

        # Center placement
        x_offset = (img.shape[1] - new_w) // 2
        y_offset = (img.shape[0] - new_h) // 2
        # Split alpha and color channels
        alpha = resized_img[:, :, 3] / 255.0  # alpha channel
        rgb_carrot = resized_img[:, :, :3]  # RGB channels

        # Alpha blending into 3-channel canvas
        for c in range(3):
            img[y_offset:y_offset + resized_img.shape[0], x_offset:x_offset + resized_img.shape[1], c] = (
                    alpha * rgb_carrot[:, :, c] + (1 - alpha) * img[y_offset:y_offset + resized_img.shape[0],
                                                                x_offset:x_offset + resized_img.shape[1], c]
            )

        
        cv2.putText(img, f'Times stirred: {self.transition_count}', (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Food cooked: {self.round_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.imshow(self.win_right, img)
        #cv2.waitKey(1)


    def visualize_good_job(self):
        img = self.good_job_img
        cv2.imshow(self.win_right, img)

        if time.time() >= self.good_job_end:
            self.good_job_active = False
            self.transition_count = 0
            self.total_transitions = 0
            self.round_count = 0

            self.set_mode("choose")
    

    def overlay(self, background, overlay_img, x, y, scale=1.0):
        """
        Simple helper to overlay a transparent RGBA image on a BGR background.
        """
        overlay_img = cv2.resize(overlay_img, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay_img.shape
        if h <= 0 or w <= 0:
            return background

        # Crop if overlay goes outside background
        if y + h > background.shape[0]: h = background.shape[0] - y
        if x + w > background.shape[1]: w = background.shape[1] - x
        if h <= 0 or w <= 0: return background

        alpha = overlay_img[:h, :w, 3:] / 255.0
        background[y:y + h, x:x + w] = alpha * overlay_img[:h, :w, :3] + (1 - alpha) * background[y:y + h, x:x + w]
        return background


    def provide_feedback(self, decision, frame, joints, elbow_angle_mvg):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.
        :param elbow_angle_mvg: The moving average from the left elbow angle.

        """

        mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        if joints.pose_landmarks:
            h, w, _ = frame.shape
            wrist = joints.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            
        # Define the number and text to display
        #number = elbow_angle_mvg

        
        # Set the position, font, size, color, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_color = (0, 0, 0)  # White color in BGR
        thickness = 2



        # Define the position for the number and text
        text_position = (50, 50)

        if self.mode == "choose":
            text = "Good morning, Eleanor!"
        elif self.mode == "gardening":
            text = "Water the plant!"
        elif self.mode == "cooking":
            text = "Stir the pot!"
        else:
            text = ""


        # Draw the text on the image
        cv2.putText(frame,text, text_position, font, font_scale, font_color, thickness)

        if joints and joints.pose_landmarks and self.mode in ("gardening", "cooking"):
            h, w, _ = frame.shape

            wrist = joints.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            index = joints.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
            index_x = int(index.x * w)
            index_y = int(index.y * h)

            overlay_img = None

            if self.mode == "gardening":
                overlay_img = self.can_img
                frame = self.overlay(frame, overlay_img, wrist_x - 20, wrist_y - 40, scale=0.2)
            elif self.mode == "cooking":
                overlay_img = self.spoon_img
                frame = self.overlay(frame, overlay_img, index_x - 10, index_y - 50, scale=0.2)
                

        # Display the frame (for debugging purposes)
        cv2.imshow('Sport Coaching Program', frame)
