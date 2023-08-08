# medical_project_9
#Inspiration

Skin diseases are more common than other diseases. Skin diseases may be caused by fungal infection, bacteria, allergy, or viruses, etc. They have a serious impact on people’s life and health. Current research proposes an efficient approach to identify singular type of skin diseases. It is necessary to develop automatic methods in order to increase the accuracy of diagnosis for multi type skin diseases. So, image processing techniques help to build automated screening system for dermatology at an initial stage. The experimental results demonstrate the effectiveness and feasibility of the proposed method.

We proposed an image processing-based method to detect skin diseases. This method takes the digital image of disease effect skin area, and then uses image analysis to identify the type of disease. Our proposed approach is simple, fast and does not require expensive equipment, it can run on any device which has internet access. Just upload the image of your skin and check whether you have any skin disease or not. Artificial intelligence (AI) has wide applications in healthcare, including dermatology. Machine learning (ML) is a subfield of AI involving statistical models and algorithms that can progressively learn from data to predict the characteristics of new samples and perform a desired task. Although it has a significant role in the detection of skin cancer, dermatology skill lags behind radiology in terms of AI acceptance. With continuous spread, use, and emerging technologies, AI is becoming more widely available even to the general population. AI can be of use for the early detection of skin cancer. For example, the use of deep convolutional neural networks can help to develop a system to evaluate images of the skin to diagnose skin cancer. Early detection is key for the effective treatment and better outcomes of skin cancer. Specialists can accurately diagnose the cancer, however, considering their limited numbers, there is a need to develop automated systems that can diagnose the disease efficiently to save lives and reduce health and financial burdens on the patients. ML can be of significant use in this regard.

###What it does:-


It basically takes a picture of the skin disease in terms of a web app , and the predicts the disease and mentions the accuracy along with which the disease has been predicted and opens up the link where the about and cure of the disease is mentioned.

###How we built it :- 

We took HAM10000 dataset and pre-processed and extracted the best features , after that the processed data was tested on various models , out of which MobileNet gave us the best accuarcy, The entire frontened was built on html,css,javascript and the web app was made in flask.

###Challenges we ran into:-

We face problems while connectinng the frontened and backened as the web app was running in flask, and the frontened had javascript and jquery part attached ,so to connect pythin with the frontened in html was a great issue that we faced.

###Accomplishments that we're proud of :- 

At last we were finally able to connect the frontend and the backend and the web app was running smoothly.

###What we learned :- 

Learning was about logic building for connection of model with the web app and also how to connect both the frontend and backened written completely in two different languages.

###What's next for Skin disease Prediction Using Deep Learning
We are going to test this database on various other models, for better accuracy and even segment the image data, further we are going to expan this web app and would form a website for early prediction and suggestion of almost all the common diseases like heart disease detection, thyroid detection etc.

What's next for Skin disease Prediction Using Deep Learning
We are going to test this database on various other models, for better accuracy and even segment the image data, further we are going to expan this web app and would form a website for early prediction and suggestion of almost all the common diseases like heart disease detection, thyroid detection etc .

###Build-in with
HTML,CSS,Javascript, Jquery,Python

#How to run :---
1) Clone this repository
2) Then run pip install -r requirements.txt
3) After that run python app.py
