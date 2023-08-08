import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import torch
import time
from datetime import datetime

class YOLOv5App(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        self.image = Image(size_hint=(1, 0.8))  # Adjust size_hint to make the image larger
        self.layout.add_widget(self.image)
        
        self.is_running = False
        self.cap = None
        self.model = None  # Initialize the model
        self.frame_count = 0
        self.start_time = 0
        
        # Add a button to trigger the YOLOv5 execution
        self.button = Button(text='Run YOLOv5', size_hint=(1, 0.2))  # Adjust size_hint to make the button smaller
        self.button.bind(on_press=self.toggle_yolov5)
        self.layout.add_widget(self.button)
        
        return self.layout
    
    def load_yolov5_model(self):
        # Load YOLOv5 model here
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp20/weights/best.pt', force_reload=True)
        
    def toggle_yolov5(self, instance):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update, 1.0 / 30)  # 30 FPS update
            self.load_yolov5_model()  # Load YOLOv5 model
            self.button.text = 'Stop Running YOLOv5'
            self.start_time = time.time()
        else:
            self.is_running = False
            self.cap.release()
            Clock.unschedule(self.update)
            self.button.text = 'Run YOLOv5'
            self.image.texture = None  # Clear the image display
            self.frame_count = 0  # Reset frame count
    
    def update(self, dt):
        if not self.is_running:
            return
        
        success, frame = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            self.toggle_yolov5(None)  # Stop the execution
            return
        
        frame = cv2.resize(frame, (640, 480))
        results = self.model(frame)  # Use self.model for YOLOv5
        
        # Process YOLOv5 results and display them on the frame
        class_counts = {}
        for det in results.xyxy[0]:
            class_name = self.model.names[int(det[5])]
            confidence = det[4].item()
            
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        label_y = 30
        label_line_height = 30
        
        for class_name, count in class_counts.items():
            label = f'{class_name}: {count}'
            cv2.putText(frame, label, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            label_y += label_line_height
        
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        
        cv2.putText(frame, f'FPS: {fps:.2f}', (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the processed frame
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture1
        self.frame_count += 1
        
    def on_stop(self):
        if self.cap is not None:
            self.cap.release()

if __name__ == '__main__':
    YOLOv5App().run()
