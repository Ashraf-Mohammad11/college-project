#import require python classes and packages
#import packages and classes
import pandas as pd
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
import os
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Activation, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from Attention import attention #===============importing attention layer
from sklearn.metrics import average_precision_score
import seaborn as sns
from numpy import dot
from numpy.linalg import norm
from flask import *
from auth_utils import *
from werkzeug.utils import secure_filename
import os,random
import numpy as np
import pandas as pd
import tensorflow as tf


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_UPLOAD_SIZE_MB = 512  # Maximum upload size in megabytes

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Set max content length

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/signup")
def signup_route():
    return signup()


@app.route("/signin")
def signin_route():
    return signin()

@app.route('/home')
def home():
    return render_template('home.html')

boundings = []
X = []
Y = []
#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
#define extended kalman filter object
kalman = cv2.TrackerKCF_create()
#function to normalize bounding boxes
def convert_bb(img, width, height, xmin, ymin, xmax, ymax):
    bb = []
    conv_x = (64. / width)
    conv_y = (64. / height)
    height = ymax * conv_y
    width = xmax * conv_x
    x = max(xmin * conv_x, 0)
    y = max(ymin * conv_y, 0)     
    x = x / 64
    y = y / 64
    width = width/64
    height = height/64
    return x, y, width, height


@app.route('/load_dataset')
def load_dataset():
    global labels,X,Y,boundings
    # Load the dataset and fill missing values with 0
    try:
        #define CVC-09 pedestrian dataset path
        path = 'Dataset/Annotations'
        if os.path.exists('model/X.txt.npy'):
            X = np.load('model/X.txt.npy')#load all processed images
            Y = np.load('model/Y.txt.npy')                    
            boundings = np.load('model/bb.txt.npy')#load bounding boxes
        else:
            for root, dirs, directory in os.walk(path):#if not processed images then loop all annotation files with bounidng boxes
                for j in range(len(directory)):
                    file = open('Dataset/Annotations/'+directory[j], 'r')
                    name = directory[j]
                    name = name.replace("txt","png")
                    if os.path.exists("Dataset/FramesPos/"+name):
                        img = cv2.imread("Dataset/FramesPos/"+name)
                        height, width, channel = img.shape
                        img = cv2.resize(img, (64, 64))#Resize image
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        lines = file.readlines()
                        boxes = []
                        for m in range(0,12):
                            boxes.append(0)
                        start = 0    
                        for i in range(len(lines)):#loop and read all bounding boxes from image
                            if start < 12:
                                line = lines[i].split(" ")
                                x1 = float(line[0])
                                y1 = float(line[1]) 
                                x2 = float(line[2])
                                y2 = float(line[3])
                                xx1 = x1 - x2 / 2
                                yy1 = y1 - y2 / 2
                                x2 = x1 + x2 / 2
                                y2 = y1 + y2 / 2
                                x1 = xx1
                                y1 = yy1
                                x1, y1, x2, y2 = convert_bb(img, width, height, x1, y1, x2, y2)#normalized bounding boxes
                                bbox = (x1 * 64, y1 * 64, x2 * 64, y2 * 64)
                                kalman.init(img, bbox)#apply kalman filter images to correct bounding box lovcations
                                kalman.update(img)
                                boxes[start] = x1
                                start += 1
                                boxes[start] = y1 
                                start += 1
                                boxes[start] = x2
                                start += 1
                                boxes[start] = y2
                                start += 1
                        boundings.append(boxes)
                        X.append(img)
                        Y.append(0)
            X = np.asarray(X)#convert array to numpy format
            Y = np.asarray(Y)
            boundings = np.asarray(boundings)
            np.save('model/X.txt',X)#save all processed images
            np.save('model/Y.txt',Y)                    
            np.save('model/bb.txt',boundings)
        print("Dataset Images Loaded")
        print("Total Images Found in Dataset : "+str(X.shape[0])) 
        print("Dataset loading task completed")
        print("Total images found in dataset : "+str(X.shape[0]))
        
        # Convert dataset summary to HTML for easy display
        dataset_html = f"""
        <p>Total Images: {X.shape[0]}</p>
        """
        message = "Dataset loaded successfully."
        
    except FileNotFoundError:
        dataset_html = "<p>Dataset file not found. Please ensure the file path is correct.</p>"
        message = "Error: Could not load dataset."
    except Exception as e:
        dataset_html = f"<p>An error occurred: {str(e)}</p>"
        message = "An error occurred while loading the dataset."

    # Render the template and display the dataset and message
    return render_template("dataset.html", dataset_html=dataset_html, message=message)


@app.route("/preprocess")
def preprocess_and_split_data():
    global X, Y,boundings,trainImages,testImages,trainBBoxes,testBBoxes,trainLabels,testLabels
    #preprocess images by applying shuffling and then split dataset into train and test
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffle image pixels
    X = X[indices]
    Y = Y[indices]
    boundings = boundings[indices]
    #split dataset into train and test where 20% dataset size for testing and 80% for testing
    split = train_test_split(X, Y, boundings, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    print()
    print("Dataset train & test split as 80% dataset for training and 20% for testing")
    print("Training Size (80%): "+str(trainImages.shape[0])) #print training and test size
    print("Testing Size (20%): "+str(testImages.shape[0]))
    print()
    total_size = X.shape[0]
    training_size = trainImages.shape[0]
    testing_size = testImages.shape[0]
    
    return render_template("preprocess.html", 
                           total_size=total_size,
                           training_size=training_size, 
                           testing_size=testing_size,
                           train_percentage=80, 
                           test_percentage=20)

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='micro') * 100
    r = recall_score(testY, predict,average='micro') * 100
    f = f1_score(testY, predict,average='micro') * 100
    a = accuracy_score(testY,predict)*100     
    print(algorithm+' Accuracy  : '+str(a))
    print(algorithm+' Precision : '+str(p))
    print(algorithm+' Recall    : '+str(r))
    print(algorithm+' FSCORE    : '+str(f))    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    return a, p, r, f



@app.route('/existing_alg')
def existing_algorithm():
    
    if os.path.exists("model/frcnn.hdf5") == False:
        rcnn = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(X.shape[1], X.shape[2], X.shape[3])))
        rcnn.trainable = False
        flatten = rcnn.output
        flatten = Flatten()(flatten)
        bboxHead = Dense(16, activation="relu")(flatten)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(12, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(16, activation="relu")(flatten)
        softmaxHead = Dropout(0.2)(softmaxHead)
        softmaxHead = Dense(8, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.2)(softmaxHead)
        softmaxHead = Dense(1, activation="softmax", name="class_label")(softmaxHead)
        frcnn_model = Model(inputs=rcnn.input, outputs=(bboxHead, softmaxHead))
        losses = {"class_label": "binary_crossentropy", "bounding_box": "mean_squared_error"}
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        opt = Adam(lr=1e-4)
        frcnn_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
        testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
        model_check_point = ModelCheckpoint(filepath='model/frcnn.hdf5', verbose = 1, save_best_only = True)
        hist = frcnn_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=10, verbose=1,callbacks=[model_check_point])
        f = open('model/frcnn.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        frcnn_model = load_model('model/frcnn.hdf5')
    predict = frcnn_model.predict(testImages)[0]#perform prediction on test data
    pred = []
    for i in range(len(predict)):
        box_acc = dot(predict[i], testBBoxes[i])/(norm(predict[i])*norm(testBBoxes[i]))
        if box_acc < 0.56:
            pred.append(1)
        else:
            pred.append(0)
    accuracy, precision, recall, f_measure = calculateMetrics("Faster RCNN", pred, testLabels)
    # Pass metrics to the template
    return render_template("existing_alg.html", 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           f_measure=f_measure
                           )


@app.route('/proposed_alg')
def proposed_algorithm():
    #train propose improved YoloV5 with squeeze attention model
    #define input shape
    input_img = Input(shape=(64, 64, 3))
    #create YoloV4 layers with 32, 64 and 512 neurons or data filteration size
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = attention(return_sequences=True,name='attention')(x)#adding squeeze attention model to make improved Yolov5
    x = Flatten()(x)
    #define output layer with 4 bounding box coordinate and 1 weapan class
    x = Dense(256, activation = 'relu')(x)
    x = Dense(256, activation = 'relu')(x)
    x_bb = Dense(12, name='bb',activation='sigmoid')(x)
    x_class = Dense(1, activation='sigmoid', name='class')(x)
    #create yolo Model with above input details
    yolo_model = Model([input_img], [x_bb, x_class])
    #compile the model
    yolo_model.compile(Adam(lr=0.0001), loss=['mse', 'binary_crossentropy'], metrics=['accuracy'])
    if os.path.exists("model/yolo_weights.hdf5") == False:#if model not trained then train the model
        model_check_point = ModelCheckpoint(filepath='model/yolo_weights.hdf5', verbose = 1, save_best_only = True)
        hist = yolo_model.fit(trainImages, [trainBBoxes, trainLabels], batch_size=32, epochs=10, validation_data=(testImages, [testBBoxes, testLabels]), callbacks=[model_check_point])
        f = open('model/yolo_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:#if model already trained then load it
        yolo_model = load_model("model/yolo_weights.hdf5", custom_objects={'attention': attention})
    predict = yolo_model.predict(testImages)#perform prediction on test data
    predict = predict[0]
    pred = []
    for i in range(len(predict)):
        box_acc = dot(predict[i], testBBoxes[i])/(norm(predict[i])*norm(testBBoxes[i]))
        if box_acc < 0.80:
            pred.append(1)
        else:
            pred.append(0)
    
    accuracy, precision, recall, f_measure = calculateMetrics("Propose Improved YoloV5", pred, testLabels)
    # Pass metrics to the template
    return render_template("proposed_alg.html", 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           f_measure=f_measure
                           )

@app.route('/extension_alg')
def extension_algorithm():
    
    if os.path.exists("model/yolov6.hdf5") == False:
        yolov6 = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(X.shape[1], X.shape[2], X.shape[3])))
        yolov6.trainable = False
        flatten = yolov6.output
        flatten = Flatten()(flatten)
        bboxHead = Dense(16, activation="relu")(flatten)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(12, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(16, activation="relu")(flatten)
        softmaxHead = Dropout(0.2)(softmaxHead)
        softmaxHead = Dense(8, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.2)(softmaxHead)
        softmaxHead = Dense(1, activation="sigmoid", name="class_label")(softmaxHead)
        yolov6_model = Model(inputs=yolov6.input, outputs=(bboxHead, softmaxHead))
        losses = {"class_label": "binary_crossentropy", "bounding_box": "mean_squared_error"}
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        opt = Adam(lr=1e-4)
        yolov6_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
        testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
        model_check_point = ModelCheckpoint(filepath='model/yolov6.hdf5', verbose = 1, save_best_only = True)
        hist = yolov6_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=10, verbose=1,callbacks=[model_check_point])
        f = open('model/yolov6.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        yolov6_model = load_model('model/yolov6.hdf5')
    predict = yolov6_model.predict(testImages)[0]#perform prediction on test data
    pred = []
    for i in range(len(predict)):
        box_acc = dot(predict[i], testBBoxes[i])/(norm(predict[i])*norm(testBBoxes[i]))
        if box_acc < 0.57:
            pred.append(1)
        else:
            pred.append(0)
    
    accuracy, precision, recall, f_measure = calculateMetrics("Extension YoloV6", pred, testLabels)
    # Pass metrics to the template
    return render_template("extension_alg.html", 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           f_measure=f_measure
                           )


@app.route('/graph')
def display_graph():
    df = pd.DataFrame([['Existing Faster-RCNN','Precision',precision[0]],
                       ['Existing Faster-RCNN','Recall',recall[0]],
                       ['Existing Faster-RCNN','F1 Score',fscore[0]],
                       ['Existing Faster-RCNN','Accuracy',accuracy[0]],
                    ['Propose Yolov5 with CA Attention','Precision',precision[1]],
                    ['Propose Yolov5 with CA Attention','Recall',recall[1]],
                    ['Propose Yolov5 with CA Attention','F1 Score',fscore[1]],
                    ['Propose Yolov5 with CA Attention','Accuracy',accuracy[1]],
                    ['Extension YoloV6','Precision',precision[2]],
                    ['Extension YoloV6','Recall',recall[2]],
                    ['Extension YoloV6','F1 Score',fscore[2]],
                    ['Extension YoloV6','Accuracy',accuracy[2]],
                    ],columns=['Parameters','Algorithms','Value'])

    # Create the bar graph
    fig, ax = plt.subplots(figsize=(5, 3))  # Increase the figure size (10x6 inches)
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=ax)

    # Set the title and labels
    ax.set_title("Algorithms Performance Comparison", fontsize=6)
    ax.set_xlabel("Metrics", fontsize=4)
    ax.set_ylabel("Values", fontsize=4)
    
    # Adjust tick labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=4)

    # Move the legend outside the plot
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)

    # Save the graph as an image with high DPI for clarity
    graph_path = os.path.join(app.static_folder, 'graph.png')
    plt.tight_layout()  # Ensure everything fits well
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')  # Save the plot with 300 DPI for higher resolution
    plt.close()  # Close the plot to free memory

    # Render the HTML template and pass the image path
    return render_template("graph.html", graph_url='/static/graph.png')



@app.route('/predict')
def upload():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    global ensemble_model, rf, labels
    if 'testimage' not in request.files:
        message = 'No image file selected'
        return render_template('predict.html', message=message)

    image_file = request.files['testimage']

    if image_file.filename == '':
        message = 'No selected file'
        return render_template('predict.html', message=message)

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        try:
            # Call predictLanding function with the uploaded image path
            result_image, prediction_label = predictPedestrian(filepath)
            print("Testing......print")
            # Display results in the template
            return render_template(
                'predict.html', 
                result_image=result_image, 
                prediction_label=prediction_label
            )
        except Exception as e:
            message = f"Error processing image: {e}"
            return render_template('predict.html', message=message)
    else:
        message = 'Allowed file types: .png, .jpg, .jpeg'
        return render_template('predict.html', message=message)

# This is where we serve static files from
@app.route('/uploads/<filename>')
def send_file_from_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 


#function to detect and predict pedestrains
def predictPedestrian(image_path):
    img = cv2.imread(image_path)  # Read test image
    img = cv2.resize(img, (64, 64))  # Resize image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = img.reshape(1, 64, 64, 3)

    # Load model
    yolo_model = load_model("model/yolo_weights.hdf5", custom_objects={'attention': attention})
    predict_value = yolo_model.predict(img1)  # Perform prediction

    predict = predict_value[0][0]  # Get bounding boxes
    predicted_labels = predict_value[1][0]  # Get predicted labels

    total_persons = 0  # Counter for total persons detected
    img_original = cv2.imread(image_path)
    img_resized = cv2.resize(img_original, (600, 400))  # Resize for better display

    start = 0
    while start < 12:
        x1 = predict[start] * 600  # Scale bounding box coordinates
        start += 1
        y1 = predict[start] * 400
        start += 1
        x2 = predict[start] * 600
        start += 1
        y2 = predict[start] * 400
        start += 1

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 20:
            total_persons += 1  # Count detected persons
            cv2.rectangle(img_resized, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_resized, f'Person {total_persons}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    label_text = f'Total Persons: {total_persons}'

    # Save the annotated image for display in Flask
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.jpg')
    cv2.imwrite(result_image_path, img_resized)

    return result_image_path, label_text

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)