# Television Gesture Recognition - Conv3D & CNN-RNN Based Classification

## Table of Contents
* [RNN Model Overview](#rnn-model-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for Modeling](#approach-for-modeling)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## RNN Model Overview

Recurrent Neural Networks (RNNs) are a class of neural networks designed for processing sequential data, such as time series, text, or video frames, where the order of data is crucial. Unlike traditional feedforward neural networks, RNNs incorporate loops within their architecture, allowing information to persist across time steps. This capability makes RNNs particularly effective at modeling temporal dependencies and patterns in sequential data.

In an RNN, the output at each time step depends not only on the current input but also on the hidden state, which encapsulates information from previous time steps. Mathematically, the hidden state at time $t ( h_t )$ is computed as follows:

$$
h_t = f(W_xx_t + W_hh_{t-1} + b)
$$

Here:
- $x_t$: Input at time step $t$
- $h_{t-1}$: Hidden state from the previous time step
- $W_x$, $W_h$: Weight matrices
- $b$: Bias term
- $f$: Activation function (typically tanh or ReLU)

Despite their effectiveness, standard RNNs face challenges when learning long-term dependencies due to the vanishing gradient problem. To overcome this limitation, variations such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) were introduced.

---

### Long Short-Term Memory (LSTM)

LSTMs are a specialized type of RNN designed to handle long-term dependencies more effectively. The LSTM cell includes three key components, called gates, which regulate the flow of information: **forget gate**, **input gate**, and **output gate**. These gates allow the LSTM to selectively retain or discard information over time, enabling the model to maintain long-term memory.

At each time step, the LSTM computes the following:
1. **Forget gate**: Decides which information to discard from the cell state.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

2. **Input gate**: Determines which new information to store in the cell state.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
C̃_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

3. **Cell state update**:

$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$

4. **Output gate**: Decides what to output based on the cell state.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$
$$
h_t = o_t * \tanh(C_t)
$$

LSTMs are highly effective for tasks involving long sequences, such as speech recognition, machine translation, and video analysis.

---

### Gated Recurrent Unit (GRU)

GRUs are a simplified version of LSTMs that also address the vanishing gradient problem. GRUs combine the forget and input gates into a single **update gate** and directly control the hidden state without using a separate memory cell. This simplification reduces the computational complexity of GRUs while still maintaining their effectiveness for capturing long-term dependencies.

The GRU cell is computed as follows:

1. **Update gate**:

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

2. **Reset gate**:

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

3. **Candidate hidden state**:

$$
h̃_t = \tanh(W \cdot [r_t * h_{t-1}, x_t] + b)
$$

4. **Hidden state update**:

$$
h_t = z_t * h_{t-1} + (1 - z_t) * \tilde{h}_t
$$

GRUs require fewer parameters than LSTMs, making them faster to train and suitable for smaller datasets while still achieving comparable performance.

 The approach described here represent the practical process utilised in industry to predict categorical target parameters for business.


## Problem Statement

This exercise aims to develop a cool feature in the smart-TV for the TV Manufacturers that can recognise **five different gestures** performed by the user which will help users control the TV without using a remote. The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

* Thumbs up:  Increase the volume
* Thumbs down: Decrease the volume
* Left swipe: 'Jump' backwards 10 seconds
* Right swipe: 'Jump' forward 10 seconds  
* Stop: Pause the movie

The training data consists of a few hundred videos categorised into one of the five classes corresponding to five gestures. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

Our task is to train a deep learning model (a Convolutional 3D or CNN-RNN combination) using the gesture clips in the 'train' dataset which performs well on the clips in the 'validation' dataset as well.

## Technologies Used

Python Jypyter Notebook in cloud environment with GPU support has been used for the exercise. Apart from ususal Python libraries like  Numpy and Pandas, there are machine Learning specific libraries used to prepare, analyse, build model and visualise data. The following model specific libraries have been used in the case study:

- tensorflow
- tensorflow.keras


## Approach for Modeling

Our approach for analysing videos using neural networks would include exploring two types of architectures:

1. **3D convolutional network** which is a natural extension to the 2D convolutions in which case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). The video becomes a 4-D tensor of shape 100x100x3x30. A 3-D kernel/filter (a 'cubic' filter) will '3D-convolve' on each of the three channels of the (100x100x30) tensor.

2. **Standard CNN + RNN architecture** in which we pass the images of a video through a Convolutional Neural Network (CNN) which extracts a feature vector for each image, and then pass the sequence of these feature vectors through a Recurrent Neural Network(RNN).

The **Steps** we will take in this project are:

1. Setup Data Access
2. Build Generator for Incremental Image Loading
3. Build and Validate Models
4. Conclusion

Some distinguishing processes in this approach include,

- Standardisation of image size from two different sizes in which input images were configured

- Building a `Generator` function to load images in memory only when the images are required for processing/ training

- Usage of `Dropouts` along with Dense layers of CNN architecture to prevent overfitting

- Usage of `Batch Normalisation` to normalise input data to a hidden layer in order to speed-up training and regularise the model

- Invokation of `callback` functions to save the results during model training and moderate the learning rate


## Classification Outcome

After experimenting with Conv3D and CNN+RNN based models, we successfully finalised a CNN-RNN model for gesture recognition, achieving a strong performance with a **training accuracy of 1.0 and a validation accuracy of 0.9**. The model architecture utilized five Conv2D layers for feature extraction from video frames, followed by a GRU layer with 32 units to capture temporal dependencies across frames. 

To mitigate overfitting and enhance model generalization, we applied dropout, batch normalization, and kernel regularization within each Conv2D layer. The input images were resized to 128x128 resolution, and a batch size of 32 was used for efficient training. This setup allowed the model to effectively recognize gestures, making it suitable for real-time gesture-controlled applications, such as operating a TV. 

Overall, the model demonstrates both high accuracy and robustness, providing a reliable solution for gesture recognition in video sequences.


## Conclusion

RNNs, along with their advanced variants like LSTMs and GRUs, are powerful tools for modeling sequential data. They enable the capture of both short-term and long-term dependencies, making them ideal for a wide range of tasks such as natural language processing, time series forecasting, and video analysis. By leveraging these models, gesture recognition tasks, like the one in this project, can effectively analyze temporal patterns across video frames to achieve accurate predictions.


## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.
