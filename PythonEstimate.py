import unreal_engine as ue
from unreal_engine.classes import KismetSystemLibrary
import cv2
import mediapipe as mp
import numpy as np

ue.log('Python FaceEstimate')

class FaceEstimate:

    def begin_play(self):
        KismetSystemLibrary.PrintString(InString="Python_Estimate_BeginPlay", Duration=5.0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            ue.log("Camera initialization failed")
            return

        # Initialize Mediapipe FaceMesh with optimized settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize frame counter for frame skipping
        self.frame_counter = 0

    def tick(self, delta_time):
        self.frame_counter += 1
        if self.frame_counter % 5 != 0:  # Process every 5th frame
            return

        ret, image = self.cap.read()
        if not ret:
            ue.log("Failed to read frame from camera")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_left = tuple(map(int, [face_landmarks.landmark[33].x * image.shape[1], face_landmarks.landmark[33].y * image.shape[0]]))
                right_eye_right = tuple(map(int, [face_landmarks.landmark[263].x * image.shape[1], face_landmarks.landmark[263].y * image.shape[0]]))
                nose_top = tuple(map(int, [face_landmarks.landmark[4].x * image.shape[1], face_landmarks.landmark[4].y * image.shape[0]]))

                # Adjusting for misdetection
                if abs(nose_top[1] - (face_landmarks.landmark[33].y * image.shape[0])) < 20:
                    nose_top = tuple(map(int, [face_landmarks.landmark[6].x * image.shape[1], face_landmarks.landmark[6].y * image.shape[0]]))

                nose_bottom = tuple(map(int, [face_landmarks.landmark[168].x * image.shape[1], face_landmarks.landmark[168].y * image.shape[0]]))

                # Draw landmarks and features on the image
                cv2.circle(image, left_eye_left, 5, (0, 0, 255), -1)
                cv2.circle(image, right_eye_right, 5, (0, 0, 255), -1)
                cv2.circle(image, nose_top, 5, (0, 0, 255), -1)
                triangle_pts = np.array([left_eye_left, right_eye_right, nose_top])
                cv2.polylines(image, [triangle_pts], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.line(image, nose_top, nose_bottom, (0, 255, 255), 2)

                # Calculate the centroid and the face direction
                centroid = (
                    (left_eye_left[0] + right_eye_right[0] + nose_top[0]) / 3,
                    (left_eye_left[1] + right_eye_right[1] + nose_top[1]) / 3
                )
                distance_x = nose_top[0] - centroid[0]
                arrow_length = min(abs(distance_x), 100) * np.sign(distance_x)
                direction = np.array([arrow_length, 0])
                arrow_end = tuple(map(int, np.array(nose_top) + direction))
                cv2.arrowedLine(image, nose_top, arrow_end, (255, 0, 0), 2)

                #self.uobject.GetFaceAngleValue(distance_x)

                #Add
                self.uobject.FaceAngleX = distance_x
                ue.log(f"Set FaceAngleX to: {distance_x}")

        # Flip the image horizontally
        image = cv2.flip(image, 1)
        
        cv2.imshow('Face Orientation', image)
        
    def end_play(self, end_play_reason):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        ue.log("Released camera and destroyed all OpenCV windows")
