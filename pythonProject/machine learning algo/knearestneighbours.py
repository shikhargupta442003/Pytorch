import numpy as np

class KNearestNeighbor:
    def __init__(self,k):
        self.k=k
        self.eps=1e-8
    def train(self,x,y):
        self.x_train=x
        self.y_train=y

    def predict(self,x_test,loop):
        if loop==1:
            distances=self.compute_distance_loop1(x_test)
        elif loop==2:
            distances=self.compute_distance_loop2(x_test)
        else:
            distances=self.compute_distance_vectorized(x_test)
        return self.predict_labels(distances)

    def compute_distance_loop2(self,x_test):
        #By following Naive and inefficient method
        num_test=x_test.shape[0]
        num_train=self.x_train.shape[0]
        distances=np.zeros((num_test,num_train))# It gives a metrix 0. Whose rows are the current example and its columns denote all examples or train labels and it is created to find the distance between two labels.

        for i in range(num_test):
            for j in range(num_train):
                distances[i,j]=np.sqrt(self.eps+np.sum((x_test[i,:]-self.x_train[j,:])**2))

        return distances

    def compute_distance_loop1(self,x_test):
        #instead of using i and j only using i

        num_test=x_test.shape[0]
        num_train=self.x_train.shape[0]
        distances=np.zeros((num_test,num_train))

        for i in range(num_test):
            distances[i,:]=np.sqrt(self.eps+np.sum((x_test[i,:]-self.x_train)**2,axis=1))

        return distances

    def compute_distance_vectorized(self,x_test):
        #calculated the distance metrix using vectors
        # As distance(test-train)^2=test^2-2*test*train+train^2
        # Here we will use transpose of matrix as for 2*test*train as we will perform it using vector hence as,
        #[10,4]*[5,4] is only possible when we do transpose of [5,4] matrix hence result shape=[10,5](no of test ex,no. of train examples)
        squared_test_sum=np.sum(x_test**2,axis=1,keepdims=True)
        squared_train_sum=np.sum(self.x_train**2,axis=1,keepdims=True)
        mul_test_train=np.dot(x_test,self.x_train.T)
        return np.sqrt(self.eps+squared_train_sum.T+squared_test_sum.T-2*mul_test_train)

    def predict_labels(self,distances):
        num_test=distances.shape[0]
        y_pred=np.zeros(num_test)

        for i in range(num_test):
            y_indices=np.argsort(distances[i,:])
            k_closest_classes=self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i]=np.argmax(np.bincount(k_closest_classes))
        return y_pred

if __name__=='__main__':

    x = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    knn=KNearestNeighbor(k=3)
    knn.train(x,y)
    y_pred=knn.predict(x,loop=3)
    print(f'accuracy:{sum(y_pred==y)/y.shape[0]}')
