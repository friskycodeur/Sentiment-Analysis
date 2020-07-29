import nltk 
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')

filepath=f"{getcwd()}/../temp2"
nltk.data.path.append(filepath)

import numpy as np
from nltk.corpus import twitter_samples
from utils import process_tweets,build_freqs

## Preparing the Data

all_pos=twitter_samples.strings('postive_tweets.json')
all_neg=twitter_samples.string('negative_tweets.json')

test_pos=all_pos[4000:]
train_pos=all_pos[:4000]
test_neg=all_neg[4000:]
train_neg=all_neg[:4000]

X_train=train_pos+train_neg
X_test=test_pos+test_neg

y_train=np.append(np.ones((len(train_pos),1)),np.zeros((len(train_neg),0)),axis=0)
y_test=np.append(np.ones((len(test_pos),1)),np.zeros((len(test_neg),0)),axis=0)

# creating frequency dictionary
freqs=build_freqs(X_train,y_train)

# defining sigmoid function

def sigmoid(z):
    h= 1/(1+np.exp(-z))
    return h

# defining gradient descent function

def gradient_descent(x,y,theta,alpha,num_iters):
    m=len(x)
    for i in range(0,num_iters):
        z=np.dot(x,theta)
        h=sigmoid(z)
        J = (-1/m)*(((np.dot(np.transpose(y),np.log(h)))+np.dot(np.transpose(1-y),np.log(1-h))))
        theta=theta-((alpha/m)*(np.dot(np.transpose(x),(h-y))))

    J = float(J)
    return J,theta

# defining a function to extract the features

def extract_features(text,freq):
    word=process_tweets(text)
    x=np.zeros((1,3))
    X[0,0]=1

    posi=[]
    nega=[]
    for word,label in freq:
        if label==1:
            posi.append(word)
        else:
            nega.append(word)
    for w in word:
        if w in posi:
            x[0,1] += freq.get((word,1)) 
        if word in nega:
            x[0,2] += freq.get((word,0))

    assert(x.shape == (1, 3))
    return x

## Training the model

X=np.zeros((len(X_train),3))
for i in range(len(X_train)):
    X[i:]=extract_features(X_train[i],freqs)
    Y=y_train
    J,theta=gradient_descent(X,Y,np.zeros((3,1)),1e-9,1500)

## Predicting on personalized text/tweet

def predict_tweet(tweet,freq,theta):
    x=extract_features(tweet,freq)
    y_pred=sigmoid(np.dot(x,theta))
    return y_pred

## Accuracy function

def accuracy_func(x,y,freq,theta):
    y_hat=[]
    for tweet in x:
        y_pred=predict_tweet(tweet,freq,theta)

        if y_pred>0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    
    y_hat=np.asarray(y_hat)
    y=np.squeeze(y)
    accuracy=(np.sum(y_hat==y))/len(x)
    return accuracy

## Predicting our own tweet 
print(f'The accuracy of our model is {accuracy_func(X_test,y_test,freqs,theta)}')
my_tweet=input('Write you own tweet')
y_hat=predict_tweet(my_tweet,freqs,theta)
if y_hat>0.5:
    print('This tweet showcase Postive Sentiment')
else:
    print('This tweet showcase Negative Sentiment')

