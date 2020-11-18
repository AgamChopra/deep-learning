#**Simple Stock Perfomance Deep Learning Model**
Simple Stock Perfomance Predictor is a deep learning model developed to predict weather one should invest and hold a given stock for a year or not to generate profit.

#Disclosure
This is my first attempt to make a deep learning algorithem in python using numpy.
As part of my MBA program at Stevens Institute of Technology, I develeped this model for one of my Business Information Analysis class in Fall 2020.
I utilized a simplified and idealized data set to show that ANN can be used to predict future stock perfomance based on financial indicators from tax returns, the economy, and the happiness index of a country.
**Please note that this is a simple model to achieve my class' data analysis requirements only and should not be used to infer anything about the real world. Real data is noisy and very hard to model and this model simply will not work in a real world scenario due to hardware and algorithem limitations.**
 
#Skills Utilized
In this project, I utilized the following: 
 Markup :
            *[ADAM optimization](https://arxiv.org/pdf/1412.6980.pdf)
            *[Learning rate decay](https://arxiv.org/pdf/1908.01878.pdf)
            *[Input normalization](https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d)
            *[Swish over LeakyReLU](https://arxiv.org/pdf/1710.05941.pdf)[ activation function](https://arxiv.org/pdf/1901.02671.pdf)
            *[Vectorization](https://towardsdatascience.com/what-is-vectorization-in-machine-learning-6c7be3e4440a)
            *[Data Visualization](https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed)
            *[Decision threshold optimization](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)
            *[Random initialization](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78)

#Outcomes
  ![cost](https://github.com/AgamChopra/deep-learning/blob/master/Simple%20stock%20perfomance%20predictor/img/cost.png?raw=true)
  ![accuracy](https://github.com/AgamChopra/deep-learning/blob/master/Simple%20stock%20perfomance%20predictor/img/accuracy.png?raw=true)
  ![p out](https://github.com/AgamChopra/deep-learning/blob/master/Simple%20stock%20perfomance%20predictor/img/model_p_output.png?raw=true)
  ![learning decision boundry](https://github.com/AgamChopra/deep-learning/blob/master/Simple%20stock%20perfomance%20predictor/img/train_decision.png?raw=true)
  ![test output](https://github.com/AgamChopra/deep-learning/blob/master/Simple%20stock%20perfomance%20predictor/img/test_decision.png?raw=true)

#License
***[MIT](https://choosealicense.com/licenses/mit/)***
