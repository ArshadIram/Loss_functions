# LossFunctions

The aim of this repository is to have all the loss functions at one place for reference. The loss functions provides small explanation followed by mathematical formula.


| Contents|
| ---------------------- |
| 1. [**Mean Square Error**](#mean-square-error) |
| 2. [**Mean Square Lograthmic Error**](#mean-square-lograthmic-error) |
| 3. [**Mean Absolute Error**](#mean-absolute-error) |
| 4. [**Mean Absolute Percentage Error**](#mean-absolute-percentage-error) |
| 5. [**Binary Cross Entropy Loss/Log Loss**](#binary-cross-entropy) |
| 6. [**KL Divergence**](#kl-divergence) |
| 7. [**Cross-Entropy Loss/Logistic Loss/Multinomial Logistic Loss**](#cross-entropy-loss) |
| 8. [**Sparse multi-class loss function**](#sparse-multi-class-loss-function) |
| 9. [**Categorical Cross Entropy Loss**](#categorical-cross-entropy-loss) |
| 10. [**Huber loss**](huber-loss) |
| 11. [**Hinge loss**](hinge-loss) |
| 12. [**Squared hinge loss**](squared-hinge-loss) |
| 13. [**Pairwise Ranking Loss**](pairwise-ranking) |
| 14. [**Triplet loss**](triplet-loss) |
| 15. [**Ranking Losses with Different Names**](ranking-losses-with-different-names) |
| 16. [**Center loss**](center-loss) |
| 17. [**Exponential loss**](exponential-loss) |
| 17. [**Taylor Cross Entropy**](taylor-cross-entropy-loss) |
| 18. [**Symmetric Cross Entropy**](symmetric-cross-entropy) |
| 19. [**Bi-Tempered Logistic Loss**](bi-tempered-logistic-loss) |
| 20. [**Bi-Tempered Loss**](bi-tempered-loss) |
| 21. [**Class Balanced Loss**](class-balanced-loss) |
| 22. [**Focal Cosign Loss**](focal-cosign-loss) |
| 23. [**Focal Loss**](focal-loss) |
| 24. [**Label Smoothing Loss**](label-smoothing-loss) |
| 25. [**Face loss function**](face-loss-function) |



## **Mean Square Error** ##
* The mean squared error (MSE) tells you how close a regression line is to a set of points. 
* It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. 
* The squaring is necessary to remove any negative signs. 
* It also gives more weight to larger differences. It’s called the mean squared error as you’re finding the average of a set of errors. 
* The lower the MSE, the better the forecast.


<p align="center">
<img src="https://latex.codecogs.com/png.latex?MSE%20=%20\frac{1}{n}\sum_{n}^{i=1}(Y_i-\hat{Y_i})^2" />
</p>

## **Mean Square Logarithmic Error** ##
* MSLE will treat small differences between small true and predicted values approximately the same as big differences between large true and predicted values
* Use MSLE when doing regression, believing that your target, conditioned on the input, is normally distributed 
* You don’t want large errors to be significantly more penalized than small ones, in those cases where the range of the target value is large.


<p align="center">
<img src="https://latex.codecogs.com/png.latex?L(y,\hat{y})%20=%20\frac{1}{N}\sum_{N}^{i=0}(log(y_i+1)-log(\hat{y_i}+1))^2" />
</p>



## **Mean Absolute Error** ##
* Absolute Error is the amount of error in your measurements.
* The Mean Absolute Error(MAE) is the average of all absolute errors.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?MAE%20=%20\frac{1}{n}\sum_{n}^{i=1}|Y_i-\hat{Y_i}|" />
</p>

## **Mean Percentage Absolute Error** ##
* The mean absolute percentage error (MAPE) is a measure of how accurate a forecast system is. 
* It measures this accuracy as a percentage,

<p align="center">
<img src="https://latex.codecogs.com/png.latex?M%20=%20\frac{1}{n}\sum_{n}^{i=1}\frac{A_t-F_t}{A_t}" />
</p>

M=mean absolute percentage error&emsp;&emsp; n=number of times the summation iteration

```At```=actual value&emsp;&emsp;```Ft```	=	forecast value


## **Binary Cross-Entropy Loss** ##
We will have 2 classes, positive and negative. In order to predict how good our predicted probabilities, we should return high values for bad predicitions and low values for good predictions. The formula for Binary Cross Entropy/Log loss can be given as below. 

<p align="center">
<img src="https://latex.codecogs.com/png.latex?H_p(q)=-\frac{1}{N}\sum_{i=1}^{N}y_i\cdot%20log(p(y_i))+(1-y_i)\cdot%20log(1-p(y_i))" />
</p>

Negative log helps in penalizing the postive loss if it is too low.

## **KL Divergence** ##
The Kullback-Leibler Divergence,or “KL Divergence” for short, is a measure of dissimilarity between two distributions.

This means that, the closer p(y) gets to q(y), the lower the divergence. So, we need to find a good p(y) to use. It looks for the best possible p(y), which is the one that minimizes the cross-entropy.
<p align="center">
<img src="https://latex.codecogs.com/png.latex?D_{KL}(q||p)=H_p(q)-H(q)=\sum_{c=1}^{C}q(y_c)\cdot%20[log(q(y_c))-log(p(y_c))]" />
</p>


## **Cross-Entropy Loss** ##
The cross entropy loss can be given as below.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?CE=-\sum_{C}^{i}t_ilog(s_i)" />
</p>

Where ```ti``` and ```si``` are the groundtruth and the CNN score for each class ```i``` in ```C```. As usually an activation function (Sigmoid / Softmax) is applied to the scores before the CE Loss computation, we write ```f(si)``` to refer to the activations.


## **Sparse multi-class loss function** ##
The only difference between sparse categorical cross entropy and categorical cross entropy is the format of true labels. When we have a single-label, multi-class classification problem, the labels are mutually exclusive for each data, meaning each data entry can only belong to one class. Then we can represent y_true using one-hot embeddings.

For example, y_true with 3 samples, each belongs to class 2, 0, and 2.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?y_{true}%20=%20[[0,0,1],%20\\\textrm{\quad\quad\quad\quad\quad%20}1,0,0],%20\\\textrm{\quad\quad\quad\quad\quad%20}%200,0,1]]" />
</p>

Will become:
<p align="center">
<img src="https://latex.codecogs.com/png.latex?y_{true-one-hot}%20=%20[2,%200,%202]" />
</p>


## **Categorical Cross-Entropy loss** ##
Also called Softmax Loss. It is a Softmax activation plus a Cross-Entropy loss. If we use this loss, we will train a CNN to output a probability over the ***C*** classes for each image. It is used for multi-class classification.

![image](/images/categorical_loss.png)

<p align="center">
<img src="https://latex.codecogs.com/png.latex?f(s)_i=\frac{e^s_i}{\sum_{C}^{j}e^s_j}%20\quad%20CE=-\sum_{C}^{i}t_ilog(f(s)_i)" />
</p>


In the specific (and usual) case of Multi-Class classification the labels are one-hot, so only the positive class ***Cp*** keeps its term in the loss. There is only one element of the Target vector t which is not zero  ***ti=tp***. So discarding the elements of the summation which are zero due to target labels, we can write:
<p align="center">
<img src="https://latex.codecogs.com/png.latex?CE=-log\left%20(%20\frac{e^{s_p}}{\sum_{C}^{j}e^{s_j}}%20\right%20)" />
</p>



## **Huber Loss** ##
Huber loss is a superb combination of ***linear*** as well as ***quadratic*** scoring methods. It has an additional hyperparameter ***delta***. Loss is linear for values above delta and quadratic below delta. This parameter is tunable according to your data, which makes the Huber loss special.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?l(y_i,\hat{y_i})%20=%20\left\{\begin{matrix}%20(y_i,\hat{y_i})%20&%20\forall%20\left%20|y_i-\hat{y_i}%20%20\right%20|\leq%20\delta\\%202\delta%20|y_i-\hat{y_i}|-\delta^{2}&%20otherwise%20\end{matrix}\right." />
</p>

The following figure shows the change in Huber loss for different values of the ***delta*** against error.

![image](/images/huber_loss.png)

## **Hinge Loss** ##
The x-axis represents the distance from the boundary of any single instance, and the y-axis represents the loss size, or penalty, that the function will incur depending on its distance.


![image](/images/hinge_loss.png)

Hinge loss is actually quite simple to compute. The formula for hinge loss is given by the following:

<img src="https://latex.codecogs.com/png.latex?l=max(0,1-y^i(x^i-b))" />
</p>


With l referring to the loss of any given instance, ```y[i]``` and ```x[i]``` referring to the ith instance in the training set and b referring to the bias term
<img src="https://latex.codecogs.com/png.latex?l=\left\{\begin{matrix}%200%20&%20\forall%20\quad%20y\cdot(w%20\cdot%20x)%20\geq%201\\%201-y\cdot(w%20\cdot%20x)%20%20&%20otherwise%20\end{matrix}\right." />
</p>

## **Squared Hinge Loss** ##
Suppose that you need to draw a very fine decision boundary. In that case, you wish to punish larger errors more significantly than smaller errors. Squared hinge loss may then be what you are looking for, especially when you already considered the hinge loss function for your machine learning problem.

![image](/images/squared_hinge_loss.png)

<img src="https://latex.codecogs.com/png.latex?L(y,\hat{y})=\sum_{i=0}^{N}(max(0,1-y_i%20\cdot%20\hat{y_i})^2)" />
</p>



## **Pairwise Ranking loss** ##
In this setup positive and negative pairs of training data points are used. Positive pairs are composed by an anchor sample ```xa``` and a positive sample ```xp```, which is similar to ```xa``` in the metric we aim to learn, and negative pairs composed by an anchor sample ```xa``` and a negative sample ```xn```, which is dissimilar to ```xa``` in that metric.

Pairwise Ranking Loss forces representations to have ```0``` distance for positive pairs, and a distance greater than a margin for negative pairs. Being ```ra```,```rp``` and ```rn``` the samples representations and ```d``` a distance function, we can write:


<img src="https://latex.codecogs.com/png.latex?L=\left\{\begin{matrix}%20d(r_a,r_p)%20&%20if%20\quad%20PositivePair\\%20max(0,m-d(r_a,r_n))%20&%20if%20\quad%20NegativePair%20\end{matrix}\right." />
</p>



For negative pairs, the loss will be ```0``` when the distance between the representations of the two pair elements is greater than the margin ```m```. But when that distance is not bigger than ```m```, the loss will be positive, and net parameters will be updated to produce more distant representation for those two elements. The loss value will be at most ```m```, when the distance between ```ra``` and ```rn``` is ```0```. The function of the margin is that, when the representations produced for a negative pair are distant enough, no efforts are wasted on enlarging that distance, so further training can focus on more difficult pairs.

If ```r0``` and ```r1``` are the pair elements representations, y is a binary flag equal to ```0``` for a negative pair and to ```1``` for a positive pair and the distance d is the euclidian distance, we can equivalently write:

<img src="https://latex.codecogs.com/png.latex?L(r_0,r_1,y)=y\left%20\|%20r_0-r_1%20\right%20\|+(1-y)max(0,m-\left%20\|%20r_0-r_1%20\right%20\|)" />
</p>







## **Triplet loss** ##
This setup outperforms the former by using triplets of training data samples, instead of pairs. The triplets are formed by an anchor sample ```xa```, a positive sample ```xp``` and a negative sample ```xn```. The objective is that the distance between the anchor sample and the negative sample representations ```d(ra,rn)``` is greater (and bigger than a margin ```m```) than the distance between the anchor and positive representations ```d(ra,rp)```.

<img src="https://latex.codecogs.com/png.latex?L(r_a,r_p,r_n)=max(0,m+d(r_a,r_p)-d(r_a,r_n))" />
</p>


The 3 situation to analyze
* Easy Triplets: ```d(ra,rn)>d(ra,rp)+m```. The negative sample is already sufficiently distant to the anchor sample respect to the positive sample in the embedding space. The loss is ```0``` and the net parameters are not updated.
 
* Hard Triplets: ```d(ra,rn)<d(ra,rp)```. The negative sample is closer to the anchor than the positive. The loss is positive (and greater than ```m```).

* Semi-Hard Triplets: ```d(ra,rp)<d(ra,rn)<d(ra,rp)+m```. The negative sample is more distant to the anchor than the positive, but the distance is not greater than the margin, so the loss is still positive (and smaller than ```m```).

## **Ranking Losses with Different Names** ##

* Ranking loss: This name comes from the information retrieval field, where we want to train models to rank items in an specific order.

* Margin Loss: This name comes from the fact that these losses use a margin to compare samples representations distances.

* Contrastive Loss: Contrastive refers to the fact that these losses are computed contrasting two or more data points representations. This name is often used for Pairwise Ranking Loss, but I’ve never seen using it in a setup with triplets.

* Triplet Loss: Often used as loss name when triplet training pairs are employed.

* Hinge loss: Also known as max-margin objective. It’s used for training SVMs for classification. It has a similar formulation in the sense that it optimizes until a margin. That’s why this name is sometimes used for Ranking Losses.

## **Center loss**
Helps in discriminating between features. Minimizing the intra-class variations while keeping the features of different classes separable is the key. To this end, we propose the center loss function

<img src="https://latex.codecogs.com/png.latex?\mathcal{L}_C=\frac{1}{2}%20\sum_{i=1}^{m}\left%20\|%20x_i-c_y_i%20\right%20\|^{2}_2" />
</p>

For this loss, you define a per class center which serves as the centroid of embeddings corresponding to that class. The gradient update is done over the mini-batch and a hyperparameter ```alpha``` controls the learning rates of the centers. The update is given by

<img src="https://latex.codecogs.com/png.latex?c^{t+1}_j=c^{t}_j-\alpha%20\cdot%20\Delta%20c^{t}_j" />
</p>

Another scalar ```lambda``` is used to balance the two loss functions. The total loss

<img src="https://latex.codecogs.com/png.latex?\mathcal{L}%20=%20\mathcal{L}_S%20+%20\lambda%20\mathcal{L}_C" />
</p>



Last Updated: Sep-12