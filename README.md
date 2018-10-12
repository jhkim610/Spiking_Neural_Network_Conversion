implementation of the paper

"Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification"

using mnist dataset under pytorch 0.3.1 setting

run the code in the following order

cnn.py
this code will train a continuous-valued neural newtork and then it will save the weights and normalization factors for SNN

snn.py
test SNN

implemented function list:

analog input to first layer (v)

max pooling (x) -> average pooling was used considering the lower H/W complexity

softmax (x)

reset-by-subtraction (v)

p% quantile normalization (v)

batch normalization (v)

bias (v)

under testing:

time-to-first encoding
