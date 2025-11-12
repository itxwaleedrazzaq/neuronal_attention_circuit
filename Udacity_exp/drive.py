'''
The code is adopted from https://github.com/naokishibuya/car-behavioral-cloning.'
'''
import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import keras
from keras._tf_keras.keras.models import load_model
import utils
from neuronal_attention_circuit import NAC

keras.config.enable_unsafe_deserialization()

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 5
MIN_SPEED = 2

#and a speed limit
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        print("Steering Angle: " + str(steering_angle) + "  |  " + "Throttle: " + str(throttle) + "  |  " + "Speed: " + str(speed))
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # cv2.imshow('FPV of Car', np.array(image))
        # cv2.waitKey(20)

        
        try:
            raw_image = image
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1,verbose=0))
            #print('steering_angle : {}',format(steering_angle))
            if abs(steering_angle) >= 0.01:
                steering_angle =  steering_angle * 3
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 0.05

#            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
       
        except Exception as e:    
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            raw_image.save('{}.jpg'.format(image_filename))

    else:
        sio.emit('manual', data={}, skip_sid=True)

    # cv2.destroyAllWindows()
    
@sio.on('connect')
def connect(sid, environ):
    print("\n***CONNECTED TO SIMULATOR***\n\n", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model.h5 file. Model should be on the Same Path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to the Image Folder. This is where the Images from the Test Run will be Saved.'
    )
    args = parser.parse_args()

    #load model
    model = load_model(args.model,custom_objects={'NAC': NAC})

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("\n***RECORDING THIS RUN***\n")
    else:
        print("\n***NOT RECORDING THIS RUN***\n")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
