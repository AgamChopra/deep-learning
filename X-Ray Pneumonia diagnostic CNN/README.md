# X-Ray Pneumonia diagnostic Convolutional Neural Network
The goal of this project is to develop a Convolutional neural network that can accurately diagnose a patient with Pneumonia and the type of Pneumonia(viral or bacterial) by analyzing the patient’s chest X-ray image.

Update: ResNet50 architecture gave test accuracy of 1.00 and Val accuracy of 0.98.

![cost](https://github.com/AgamChopra/deep-learning/blob/master/X-Ray%20Pneumonia%20diagnostic%20CNN/img/pnyn.png?raw=true)

## Procedure
The image dataset was cataloged and index information was stored in a table. This information was then used to perform some image processing and resized to a 400px x 400px. Normalized and randomized test, train, and validation sets were then generated. 
A basic Conv Model was then trained and tested as a proof of concept. Test acc = 0.48, Val acc = 0.42
A ResNet50 model was then tested that produced a desirable result. Test acc = 1.00, Val acc = 0.98
### To Do
More complex Conv Net Architectures to be tested... 

## Outcomes
Training loss: 1.1128e-05 - accuracy: 1.0000

Testing loss: 0.0000e+00 - accuracy: 1.0000

Validation loss: 0.2522 - accuracy: 0.9829

### Testing Predictions
    Model Predictions:
    
                   Scan 1:
                   NORMAL[1. 0. 0.]
                   True, Expected [1. 0. 0.]

                   Scan 2:
                   VIRAL PNEUMONIA[0. 0. 1.]
                   True, Expected [0. 0. 1.]

                   Scan 3:
                   BACTERIAL PNEUMONIA[0. 1. 0.]
                   True, Expected [0. 1. 0.]

                   Scan 4:
                   NORMAL[1. 0. 0.]
                   True, Expected [1. 0. 0.]

                   Scan 5:
                   VIRAL PNEUMONIA[0. 0. 1.]
                   True, Expected [0. 0. 1.]
    
![im](https://github.com/AgamChopra/deep-learning/blob/master/X-Ray%20Pneumonia%20diagnostic%20CNN/img/random%20test%20plot.png)
![im](https://github.com/AgamChopra/deep-learning/blob/master/X-Ray%20Pneumonia%20diagnostic%20CNN/img/random%20train%20plot%202.png)
![im](https://github.com/AgamChopra/deep-learning/blob/master/X-Ray%20Pneumonia%20diagnostic%20CNN/img/random%20train%20plot.png)

## License

**[The MIT License*](https://github.com/AgamChopra/deep-learning/blob/master/LICENSE.md)**
