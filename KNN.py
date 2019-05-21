from numpy import *  

# create a dataset which contains 4 samples with 2 classes  
def createDataSet():  
    # create a matrix: each row as a sample  
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])  
    labels = ['A', 'A', 'B', 'B'] # four samples and two classes  
    return group, labels

# classify using kNN (k Nearest Neighbors )  
# Input:      newInput: 1 x N
#             dataSet:  M x N (M samples N, features)
#             labels:   1 x M   
#             k: number of neighbors to use for comparison  
# Output:     the most popular class label   

def kNNClassify(newInput, dataSet, labels, k):  
    numSamples = dataSet.shape[0] # 得到数据矩阵的行数
  
    diff = tile(newInput, (numSamples, 1)) - dataSet 
    # tile() 将newInput行内复制 1次 列向复制4次
    # 如tile(newInput, (4, 1)) 就是生成[[1.2, 1.0],[1.2, 1.0],[1.2, 1.0],[1.2, 1.0]]
    # 为了方便接下来矩阵相减求距离平方
    squaredDiff = diff ** 2 # x,y坐标各自相减后平方
    squaredDist = sum(squaredDiff, axis = 1) # 按行求和 得到距离平方
    distance = squaredDist ** 0.5  # 开根号 得到距离
  
    sortedDistIndices = argsort(distance)  # 得到排序后的数组的索引值
    # 如[3, 1, 2] argsort([3, 1, 2]) 得到 [1, 2, 0] 这是索引值的数组
  
    classCount = {} # 字典 保存临近k点的信息
    for i in xrange(k):  # 和range（）差不多 这个不返回列表 提高性能
        
        voteLabel = labels[sortedDistIndices[i]]  
        # sortedDistIndices[i] 从k点中按距离顺序取出点的labels

        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
        '''
            dict.get(key, default=None)
            key -- 字典中要查找的键。
            default -- 如果指定键的值不存在时，返回该默认值值。

        '''
        # classCount字典类似{
        #                       "A": 0,
        #                       "B": 0
        #                   }
  
    # 输出结果
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex   
    
    
if __name__== "__main__":    
    dataSet, labels = createDataSet()  
      
    testX = array([1.2, 1.0])  
    k = 3  
    outputLabel = kNNClassify(testX, dataSet, labels, 3)  
    print "Your input is:", testX, "and classified to class: ", outputLabel  
      
    testX = array([0.1, 0.3])  
    outputLabel = kNNClassify(testX, dataSet, labels, 3)  
    print "Your input is:", testX, "and classified to class: ", outputLabel