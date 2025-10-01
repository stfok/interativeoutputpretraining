import tensorflow as tf
import numpy as np
import random
import textxls as xlst


#xlst.write_excel_xls_append('data.xls',[[1,3]])

inputX = np.loadtxt('inputX.csv', delimiter=',')#np.random.rand(10000,6)-0.5
#inputX[:,6]=inputX[:,3]
#inputX[:,7]=inputX[:,3]
#inputX[:,8]=inputX[:,3]
#inputX[:,9]=inputX[:,3]
noise = np.random.normal(0, 0.5, inputX.shape[0]) #inputX[:,0] *4++1.5*inputX[:,1]#
outputY=[]
outputY0 = (inputX[:,0]+ inputX[:,1]) **2*2 -6*np.exp(inputX[:,2])+2*np.sin(inputX[:,3])-inputX[:,4]**3+1.5*inputX[:,5]**2+25 + np.random.normal(0, 0.5, inputX.shape[0])#+ noiseinputX[:,0] * 4 + 1#np.loadtxt('outputY.csv', delimiter=',')
outputY1 = (inputX[:,0]) **2 -2*np.exp(inputX[:,2]+0.3)+2*np.cos(inputX[:,3])+noise-20
outputY2 = 2*(inputX[:,4]) **3 +np.exp(inputX[:,5])
outputY=np.transpose(np.array([outputY0, outputY1, outputY2]))
#outputY =np.reshape(outputY,[-1,1])
learning_rate=0.000025
weight1 = tf.Variable(tf.truncated_normal([6, 12], stddev=0.1,dtype=tf.float64))
bias1 = tf.Variable(tf.truncated_normal([12], stddev=0.1,dtype=tf.float64))#inputX.shape[1],
x1 = tf.placeholder(tf.float64, [None, 6])
is_training = tf.placeholder(tf.bool, name='is_training') #tf.placeholder(tf.bool, shape=())
'''
weightb = tf.Variable(np.zeros((1,inputX.shape[1])),trainable=True)
xb1=tf.multiply(x1,weightb)
xb2= tf.add(x1,xb1)
'''
'''
weightbb = tf.Variable(np.zeros((1,inputX.shape[1])),trainable=True)
xbb1=tf.multiply(xb2,weightbb)
xbb2= tf.add(xb2,xbb1)
'''

y10_ = tf.matmul(x1, weight1) + bias1
#y10_ = tf.cast(y10_, dtype=tf.float32)
#y10_=tf.layers.batch_normalization(y10_, axis=-1, training=is_training)
#y10_= tf.cast(y10_, dtype=tf.float64)
y11_= tf.nn.relu(y10_)
#y11_ = tf.cast(y11_, dtype=tf.float32)
#y11_=tf.layers.batch_normalization(y11_, axis=-1, training=is_training)
y1_ = y11_#tf.cast(y11_, dtype=tf.float64)
#is_training=False, 

#y1_=tf.layers.batch_normalization(y1_)
'''
weight2 = tf.Variable(tf.truncated_normal([8, 6], stddev=0.1,dtype=tf.float64))
bias2 = tf.Variable(tf.truncated_normal([6], stddev=0.1,dtype=tf.float64))#inputX.shape[1],
###
wwp = tf.Variable(tf.truncated_normal([6, 6], stddev=0.1,dtype=tf.float64))
###
y22 = tf.matmul(y1_, weight2) + bias2#+x1*wwp
###
y2v_=tf.matmul(x1,wwp)-y22
##
y2_=tf.nn.relu(y2v_)
'''


##########################################3
weightin1 = tf.Variable(tf.truncated_normal([12, 6], stddev=0.1,dtype=tf.float64))
biasin1 = tf.Variable(tf.truncated_normal([6], stddev=0.1,dtype=tf.float64))#inputX.shape[1]
yin1 = tf.matmul(y1_, weightin1) + biasin1
yin1=tf.nn.relu(yin1)
weightin2 = tf.Variable(tf.truncated_normal([6, 6], stddev=0.1,dtype=tf.float64))
biasin2 = tf.Variable(tf.truncated_normal([6], stddev=0.1,dtype=tf.float64))#inputX.shape[1]

wwpin = tf.Variable(tf.truncated_normal([12, 3], stddev=0.1,dtype=tf.float64))
#biaswwpin = tf.Variable(np.random.rand(50)/100)



#biasin3 = tf.Variable(np.random.rand(50)/100)#inputX.shape[1]
yin2 = tf.matmul(yin1, weightin2) + biasin2

yin20=tf.nn.relu(yin2)

weightin3 = tf.Variable(tf.truncated_normal([6, 3], stddev=0.1,dtype=tf.float64))
biasin3 = tf.Variable(tf.truncated_normal([3], stddev=0.1,dtype=tf.float64))#inputX.shape[1]
yin22 = tf.matmul(yin20, weightin3) + biasin3


yin3=tf.matmul(y1_, wwpin)+yin22#biaswwpin



x1er = tf.placeholder(tf.float64, [None, 6])
weightin1er = tf.Variable(tf.truncated_normal([6, 3], stddev=0.1,dtype=tf.float64))
biasin1er = tf.Variable(tf.truncated_normal([3], stddev=0.1,dtype=tf.float64))#inputX.shape[1]
yin1er = tf.matmul(x1er, weightin1er) + biasin1er
yin1er=tf.nn.relu(yin1er)
weightin2er = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1,dtype=tf.float64))
biasin2er = tf.Variable(tf.truncated_normal([1], stddev=0.1,dtype=tf.float64))#inputX.shape[1]

wwper = tf.Variable(tf.truncated_normal([6, 1], stddev=0.1,dtype=tf.float64))
#biasin3 = tf.Variable(np.random.rand(50)/100)#inputX.shape[1]


yin2er = tf.matmul(yin1er, weightin2er) + biasin2er
yin3er=tf.matmul(x1er, wwper)-yin2er

'''
x1erz = tf.placeholder(tf.float64, [None, 300])
weightin1erz = tf.Variable((np.random.rand(inputX.shape[1],50)-0.5)/10)
biasin1erz = tf.Variable(np.random.rand(50)/100)#inputX.shape[1],
yin1erz = tf.matmul(x1erz, weightin1erz) + biasin1erz
yin1erz=tf.nn.relu(yin1erz)

weightin2erz = tf.Variable((np.random.rand(50,300)-0.5)/10)
biasin2z = tf.Variable(np.random.rand(300)/100)#inputX.shape[1],
yin2erz = tf.matmul(yin1erz, weightin2erz) + biasin2z
'''
'''
x1erz = tf.placeholder(tf.float64, [None, 300])
weightin1erz = tf.Variable((np.random.rand(300,50)-0.5)/10)
biasin1erz = tf.Variable(np.random.rand(50)/100)#inputX.shape[1]
yin1erz = tf.matmul(x1erz, weightin1erz) + biasin1erz

weightin2erz = tf.Variable((np.random.rand(50,300)-0.5)/10)
biasin2erz = tf.Variable(np.random.rand(300)/100)#inputX.shape[1]

#wwper = tf.Variable((np.random.rand(300,50)-0.5)/10)
#biasin3 = tf.Variable(np.random.rand(50)/100)#inputX.shape[1]


yin2erz = tf.matmul(yin1erz, weightin2erz) + biasin2erz
#yin3er=tf.matmul(y2_, wwper)-yin2er

'''


##############################################
#yin3=tf.nn.relu(yin3)






#weight3 = tf.Variable((np.random.rand(50,1)-0.5)/10)
#bias3 = tf.Variable(np.random.rand(1)/100)#inputX.shape[1],
#y3_ = tf.matmul(yin3, weight3) + bias3

#grad=tf.gradients(xs=[y2v_], ys=yin3)
yy = tf.placeholder(tf.float64, [None, 6])
y = tf.placeholder(tf.float64, [None, 3])
yer = tf.placeholder(tf.float64, [None, 3])
#yyerz = tf.placeholder(tf.float64, [None, 300])
#lossss=tf.reduce_mean(tf.reduce_sum(tf.square((y2v_- yy)), reduction_indices=[1]))
loss = tf.reduce_mean(tf.reduce_sum(tf.square((yin3 - y)), reduction_indices=[1]))
losssser=tf.reduce_mean(tf.reduce_sum(tf.square((yin3er- yer)), reduction_indices=[1]))
#losssserz=tf.reduce_mean(tf.reduce_sum(tf.square((yin2erz- yyerz)), reduction_indices=[1]))

'''
grad_W1, grad_b1 = tf.gradients(xs=[weight1, bias1], ys=loss)
new_W1 = weight1.assign(weight1 - learning_rate * grad_W1)
new_b1 = bias1.assign(bias1 - learning_rate * grad_b1)

xgrad_W2, grad_b2 = tf.gradients(xs=[weight2, bias2], ys=loss)
new_W2 = weight2.assign(weight2 - learning_rate * grad_W2)
new_b2 = bias2.assign(bias2 - learning_rate * grad_b2)
'''
gr1=[wwpin] #wwp
gr2=[weightin1, weightin2, weightin3, biasin1, biasin2, biasin3]#,weight2, bias2
gr3=[ weight1, bias1] 
gr4=[weightin1er, biasin1er, weightin2er, biasin1er,wwper]
#gr3z=[weightin1erz, biasin1erz, weightin2erz, biasin1erz]
#train_op2q = tf.train.GradientDescentOptimizer(0.0025).minimize(lossss, var_list=gr1)
#train_op2 = tf.train.GradientDescentOptimizer(0.0025).minimize(lossss, var_list=gr2)
#train=tf.group(train_op2q, train_op2) #train_op1,

train_op21 = tf.train.GradientDescentOptimizer(0.0025).minimize(loss, var_list=gr2)
train_op22 = tf.train.GradientDescentOptimizer(0.00).minimize(loss, var_list=gr1+gr3)
train2=tf.group( train_op21, train_op22) #

train_op21z = tf.train.GradientDescentOptimizer(0.0025).minimize(loss, var_list=gr1+gr2)
train_op22z = tf.train.GradientDescentOptimizer(0.0025).minimize(loss, var_list=gr3)
train3=tf.group( train_op21z, train_op22z) #

#train3 = tf.train.GradientDescentOptimizer(0.0025).minimize(losssser, var_list=gr4)

#train_oper = tf.train.GradientDescentOptimizer(0.0025).minimize(losssser, var_list=gr3)



'''
gr1=[wwp]
gr2=[weight1, bias1,weight2, bias2,weight3, bias3, weightin1, biasin1, weightin2, biasin2, wwpin]
gr3=[weightin1er, biasin1er, weightin2er, biasin1er,wwper]
#gr3z=[weightin1erz, biasin1erz, weightin2erz, biasin1erz]
train_op2q = tf.train.GradientDescentOptimizer(0.0025).minimize(lossss, var_list=gr1)
train_op2 = tf.train.GradientDescentOptimizer(0.0025).minimize(lossss, var_list=gr2)
train=tf.group(train_op2q, train_op2) #train_op1,

train_op21 = tf.train.GradientDescentOptimizer(0.0025).minimize(loss, var_list=gr1)
train_op22 = tf.train.GradientDescentOptimizer(0.0025).minimize(loss, var_list=gr2)
train2=tf.group( train_op21, train_op22) #

train_oper = tf.train.GradientDescentOptimizer(0.0025).minimize(losssser, var_list=gr3)
'''
#train_operz = tf.train.GradientDescentOptimizer(0.00025).minimize(losssserz, var_list=gr3z)
#trainz=tf.group( train_operz) #train_op1,
D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

#asd=tf.assign(D_vars[2],tf.clip_by_value(D_vars[2], -1.1,0.1))
#train = tf.train.GradientDescentOptimizer(0.00025).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

'''
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))
for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train2], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputb[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_

    #print('wwp',sess.run(wwp))
    #print('weight1', sess.run(weight1))
    #print('bias1', sess.run(bias1))
    #print('weight2', sess.run(weight2))
    #print('bias2', sess.run(bias2))
    #print('weight3', sess.run(weight3))
    #print('bias3', sess.run(bias3))
    #sess.run(asd)
    #xinw = sess.run(weightb)
    #xinw2 = sess.run(weightbb)
    #print(xinw)
    #print(xinw2)
      #if xinw>0:
      #   weightb.load(0)
      #if xinw<-1:
      #   weightb.load(-1)
    print('lossss', aa)
'''
'''
inputXT = np.random.rand(10000, 300) - 0.5
for i in range(2000):
    list1 = range(0, 10000)
    list2 = random.sample(list1, 10000)
    for j in range(0, 100):
        loss_, _ = sess.run([losssserz, trainz], feed_dict={x1erz: inputXT[list2[j * 100:(j + 1) * 100], :], yyerz: inputXT[list2[j * 100:(j + 1) * 100], :]})

    # print('wwp',sess.run(wwp))
    # print('weight1', sess.run(weight1))
    # print('bias1', sess.run(bias1))
    # print('weight2', sess.run(weight2))
    # print('bias2', sess.run(bias2))
    # print('weight3', sess.run(weight3))
    # print('bias3', sess.run(bias3))
    # sess.run(asd)
    # xinw = sess.run(weightb)
    # xinw2 = sess.run(weightbb)
    # print(xinw)
    # print(xinw2)
    # if xinw>0:
    #   weightb.load(0)
    # if xinw<-1:
    #   weightb.load(-1)
    print('loss', loss_)

'''

'''
inputXT = (np.random.rand(10000, 300) - 0.5)*0.1

inputYT=np.zeros((10000,1))


for i in range(1000):
    list1 = range(0, 10000)
    list2 = random.sample(list1, 10000)
    for j in range(0, 100):
        loss_, _ = sess.run([losssser, train3], feed_dict={x1er: inputXT[list2[j * 100:(j + 1) * 100], :], yer: inputXT[list2[j * 100:(j + 1) * 100], :]})
    print('loss', loss_)





weightin1.load(weightin1er.eval(sess),sess)
weightin2.load(weightin2er.eval(sess),sess)
biasin1.load(biasin1er.eval(sess),sess)
biasin2.load(biasin2er.eval(sess),sess)
wwpin.load(wwper.eval(sess),sess)
'''
#biaswwpin.load(biasin1erz.eval(sess),sess)

    #_, _,_, _, c = sess.run([new_W1, new_b1, new_W2, new_b2, loss], feed_dict={x1: inputX, y: outputY})
'''
print(weight1.eval(sess))
print("---------------------")
print(weight2.eval(sess))
print("---------------------")
print(bias1.eval(sess))
print("---------------------")
print(bias2.eval(sess))
print("------------------result is------------------")
'''
'''
# 训练阶段
for i in range(50):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      # 训练时传入 is_training=True
      loss_,_=sess.run([loss,train3], feed_dict={
          x1: inputX[list2[j*100:(j+1)*100],:], 
          y: outputY[list2[j*100:(j+1)*100],:], 
          is_training: True  # 正确使用占位符
      })
      aa=aa+loss_
    print('loss', aa)

# 推理阶段
# 无需修改 is_training 变量，直接传入 False
x_data = np.loadtxt('inputX2.csv', delimiter=',')  # 加载测试数据
v10 = (x_data[:,0] + x_data[:,1])**2*2 -6*np.exp(x_data[:,2]) + 2*np.sin(x_data[:,3]) -x_data[:,4]**3 +1.5*x_data[:,5]**2 +15 
v11 = (x_data[:,0])**2 -2*np.exp(x_data[:,2]+0.3) +2*np.cos(x_data[:,3]) -3
v12 = 2*(x_data[:,4])**3 + np.exp(x_data[:,5])
v1 = np.transpose(np.array([v10, v11, v12]))

# 关键修正：保持 is_training 为占位符，传入 False
v2 = sess.run(yin3, feed_dict={
    x1: x_data, 
    is_training: False  # 此处 is_training 是原始占位符
})

# 计算误差
print(np.mean(np.linalg.norm(v1-v2, axis=1)))
'''
pa00=sess.run(weightin1)
pa01=sess.run(weightin2)
pa02=sess.run(weightin3)
pa03=sess.run(biasin1)
pa04=sess.run(biasin1)
pa05=sess.run(biasin1)
pa06=sess.run(wwpin)



for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train3], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputY[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_
    print('loss', aa)
pa10=sess.run(weightin1)
pa11=sess.run(weightin2)
pa12=sess.run(weightin3)
pa13=sess.run(biasin1)
pa14=sess.run(biasin1)
pa15=sess.run(biasin1)
pa16=sess.run(wwpin)
ss=np.sum((pa00-pa10)**2)
ss=ss+np.sum((pa01-pa11)**2)
ss=ss+np.sum((pa02-pa12)**2)
ss=ss+np.sum((pa03-pa13)**2)
ss=ss+np.sum((pa04-pa14)**2)
ss=ss+np.sum((pa05-pa15)**2)
ss=ss+np.sum((pa06-pa16)**2)
print("ss",ss)
x_data = np.loadtxt('inputX2.csv', delimiter=',')#np.random.rand(1000,6)-0.5
#x_data[:,6]=x_data[:,3]
#x_data[:,7]=x_data[:,3]
#x_data[:,8]=x_data[:,3]
#x_data[:,9]=x_data[:,3]
#print(x_data)
#v1= x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]#x_data[:,0] * 4 + 1
v10=(x_data[:,0]+ x_data[:,1]) **2*2-6*np.exp(x_data[:,2])+2*np.sin(x_data[:,3])-x_data[:,4]**3+1.5*x_data[:,5]**2+25 
v11=(x_data[:,0]) **2 -2*np.exp(x_data[:,2]+0.3)+2*np.cos(x_data[:,3])-20
v12 = 2*(x_data[:,4]) **3 +np.exp(x_data[:,5])
v1=np.transpose(np.array([v10, v11, v12]))
#np.loadtxt('outputY2.csv', delimiter=',')#x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]+10
#print(v1)
v2=sess.run(yin3,feed_dict={x1: x_data, is_training: False})
#print(v2.T)
print(np.mean(np.linalg.norm(v1-v2,axis=1)))
#pa1=[weightin1, weightin2, weightin3, biasin1, biasin2, biasin3, wwpin]
#print((pa1-pa0).)

#wwpin.load(pa06,sess)
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))
for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train2], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputb[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_

    #print('wwp',sess.run(wwp))
    #print('weight1', sess.run(weight1))
    #print('bias1', sess.run(bias1))
    #print('weight2', sess.run(weight2))
    #print('bias2', sess.run(bias2))
    #print('weight3', sess.run(weight3))
    #print('bias3', sess.run(bias3))
    #sess.run(asd)
    #xinw = sess.run(weightb)
    #xinw2 = sess.run(weightbb)
    #print(xinw)
    #print(xinw2)
      #if xinw>0:
      #   weightb.load(0)
      #if xinw<-1:
      #   weightb.load(-1)
    print('lossss', aa)
pa00=sess.run(weightin1)
pa01=sess.run(weightin2)
pa02=sess.run(weightin3)
pa03=sess.run(biasin1)
pa04=sess.run(biasin1)
pa05=sess.run(biasin1)
pa06=sess.run(wwpin)





for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train3], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputY[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_
    print('loss', aa)
pa10=sess.run(weightin1)
pa11=sess.run(weightin2)
pa12=sess.run(weightin3)
pa13=sess.run(biasin1)
pa14=sess.run(biasin1)
pa15=sess.run(biasin1)
pa16=sess.run(wwpin)
ss=np.sum((pa00-pa10)**2)
ss=ss+np.sum((pa01-pa11)**2)
ss=ss+np.sum((pa02-pa12)**2)
ss=ss+np.sum((pa03-pa13)**2)
ss=ss+np.sum((pa04-pa14)**2)
ss=ss+np.sum((pa05-pa15)**2)
ss=ss+np.sum((pa06-pa16)**2)
print("ss",ss)
x_data = np.loadtxt('inputX2.csv', delimiter=',')#np.random.rand(1000,6)-0.5
#x_data[:,6]=x_data[:,3]
#x_data[:,7]=x_data[:,3]
#x_data[:,8]=x_data[:,3]
#x_data[:,9]=x_data[:,3]
#print(x_data)
#v1= x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]#x_data[:,0] * 4 + 1
v10=(x_data[:,0]+ x_data[:,1]) **2*2-6*np.exp(x_data[:,2])+2*np.sin(x_data[:,3])-x_data[:,4]**3+1.5*x_data[:,5]**2+25 
v11=(x_data[:,0]) **2 -2*np.exp(x_data[:,2]+0.3)+2*np.cos(x_data[:,3])-20
v12 = 2*(x_data[:,4]) **3 +np.exp(x_data[:,5])
v1=np.transpose(np.array([v10, v11, v12]))
#np.loadtxt('outputY2.csv', delimiter=',')#x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]+10
#print(v1)
v2=sess.run(yin3,feed_dict={x1: x_data, is_training: False})
#print(v2.T)
print(np.mean(np.linalg.norm(v1-v2,axis=1)))
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))#+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))

#wwpin.load(pa06,sess)
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))
for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train2], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputb[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_

    #print('wwp',sess.run(wwp))
    #print('weight1', sess.run(weight1))
    #print('bias1', sess.run(bias1))
    #print('weight2', sess.run(weight2))
    #print('bias2', sess.run(bias2))
    #print('weight3', sess.run(weight3))
    #print('bias3', sess.run(bias3))
    #sess.run(asd)
    #xinw = sess.run(weightb)
    #xinw2 = sess.run(weightbb)
    #print(xinw)
    #print(xinw2)
      #if xinw>0:
      #   weightb.load(0)
      #if xinw<-1:
      #   weightb.load(-1)
    print('lossss', aa)
pa00=sess.run(weightin1)
pa01=sess.run(weightin2)
pa02=sess.run(weightin3)
pa03=sess.run(biasin1)
pa04=sess.run(biasin1)
pa05=sess.run(biasin1)
pa06=sess.run(wwpin)





for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train3], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputY[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_
    print('loss', aa)
pa10=sess.run(weightin1)
pa11=sess.run(weightin2)
pa12=sess.run(weightin3)
pa13=sess.run(biasin1)
pa14=sess.run(biasin1)
pa15=sess.run(biasin1)
pa16=sess.run(wwpin)
ss=np.sum((pa00-pa10)**2)
ss=ss+np.sum((pa01-pa11)**2)
ss=ss+np.sum((pa02-pa12)**2)
ss=ss+np.sum((pa03-pa13)**2)
ss=ss+np.sum((pa04-pa14)**2)
ss=ss+np.sum((pa05-pa15)**2)
ss=ss+np.sum((pa06-pa16)**2)
print("ss",ss)
x_data = np.loadtxt('inputX2.csv', delimiter=',')#np.random.rand(1000,6)-0.5
#x_data[:,6]=x_data[:,3]
#x_data[:,7]=x_data[:,3]
#x_data[:,8]=x_data[:,3]
#x_data[:,9]=x_data[:,3]
#print(x_data)
#v1= x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]#x_data[:,0] * 4 + 1
v10=(x_data[:,0]+ x_data[:,1]) **2*2-6*np.exp(x_data[:,2])+2*np.sin(x_data[:,3])-x_data[:,4]**3+1.5*x_data[:,5]**2+25 
v11=(x_data[:,0]) **2 -2*np.exp(x_data[:,2]+0.3)+2*np.cos(x_data[:,3])-20
v12 = 2*(x_data[:,4]) **3 +np.exp(x_data[:,5])
v1=np.transpose(np.array([v10, v11, v12]))
#np.loadtxt('outputY2.csv', delimiter=',')#x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]+10
#print(v1)
v2=sess.run(yin3,feed_dict={x1: x_data, is_training: False})
#print(v2.T)
print(np.mean(np.linalg.norm(v1-v2,axis=1)))
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))#+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))

#wwpin.load(pa06,sess)
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))
for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train2], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputb[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_

    #print('wwp',sess.run(wwp))
    #print('weight1', sess.run(weight1))
    #print('bias1', sess.run(bias1))
    #print('weight2', sess.run(weight2))
    #print('bias2', sess.run(bias2))
    #print('weight3', sess.run(weight3))
    #print('bias3', sess.run(bias3))
    #sess.run(asd)
    #xinw = sess.run(weightb)
    #xinw2 = sess.run(weightbb)
    #print(xinw)
    #print(xinw2)
      #if xinw>0:
      #   weightb.load(0)
      #if xinw<-1:
      #   weightb.load(-1)
    print('lossss', aa)
pa00=sess.run(weightin1)
pa01=sess.run(weightin2)
pa02=sess.run(weightin3)
pa03=sess.run(biasin1)
pa04=sess.run(biasin1)
pa05=sess.run(biasin1)
pa06=sess.run(wwpin)





for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train3], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputY[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_
    print('loss', aa)
pa10=sess.run(weightin1)
pa11=sess.run(weightin2)
pa12=sess.run(weightin3)
pa13=sess.run(biasin1)
pa14=sess.run(biasin1)
pa15=sess.run(biasin1)
pa16=sess.run(wwpin)
ss=np.sum((pa00-pa10)**2)
ss=ss+np.sum((pa01-pa11)**2)
ss=ss+np.sum((pa02-pa12)**2)
ss=ss+np.sum((pa03-pa13)**2)
ss=ss+np.sum((pa04-pa14)**2)
ss=ss+np.sum((pa05-pa15)**2)
ss=ss+np.sum((pa06-pa16)**2)
print("ss",ss)
x_data = np.loadtxt('inputX2.csv', delimiter=',')#np.random.rand(1000,6)-0.5
#x_data[:,6]=x_data[:,3]
#x_data[:,7]=x_data[:,3]
#x_data[:,8]=x_data[:,3]
#x_data[:,9]=x_data[:,3]
#print(x_data)
#v1= x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]#x_data[:,0] * 4 + 1
v10=(x_data[:,0]+ x_data[:,1]) **2*2-6*np.exp(x_data[:,2])+2*np.sin(x_data[:,3])-x_data[:,4]**3+1.5*x_data[:,5]**2+25 
v11=(x_data[:,0]) **2 -2*np.exp(x_data[:,2]+0.3)+2*np.cos(x_data[:,3])-20
v12 = 2*(x_data[:,4]) **3 +np.exp(x_data[:,5])
v1=np.transpose(np.array([v10, v11, v12]))
#np.loadtxt('outputY2.csv', delimiter=',')#x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]+10
#print(v1)
v2=sess.run(yin3,feed_dict={x1: x_data, is_training: False})
#print(v2.T)
print(np.mean(np.linalg.norm(v1-v2,axis=1)))
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))#+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))

#wwpin.load(pa06,sess)
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))
for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train2], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputb[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_

    #print('wwp',sess.run(wwp))
    #print('weight1', sess.run(weight1))
    #print('bias1', sess.run(bias1))
    #print('weight2', sess.run(weight2))
    #print('bias2', sess.run(bias2))
    #print('weight3', sess.run(weight3))
    #print('bias3', sess.run(bias3))
    #sess.run(asd)
    #xinw = sess.run(weightb)
    #xinw2 = sess.run(weightbb)
    #print(xinw)
    #print(xinw2)
      #if xinw>0:
      #   weightb.load(0)
      #if xinw<-1:
      #   weightb.load(-1)
    print('lossss', aa)
pa00=sess.run(weightin1)
pa01=sess.run(weightin2)
pa02=sess.run(weightin3)
pa03=sess.run(biasin1)
pa04=sess.run(biasin1)
pa05=sess.run(biasin1)
pa06=sess.run(wwpin)





for i in range(500):
    list1=range(0,10000)
    list2=random.sample(list1,10000)
    aa=0
    for j in range(0,100):
      loss_,_=sess.run([loss,train3], feed_dict={x1: inputX[list2[j*100:(j+1)*100],:], y: outputY[list2[j*100:(j+1)*100],:], is_training: False})
      aa=aa+loss_
    print('loss', aa)
pa10=sess.run(weightin1)
pa11=sess.run(weightin2)
pa12=sess.run(weightin3)
pa13=sess.run(biasin1)
pa14=sess.run(biasin1)
pa15=sess.run(biasin1)
pa16=sess.run(wwpin)
ss=np.sum((pa00-pa10)**2)
ss=ss+np.sum((pa01-pa11)**2)
ss=ss+np.sum((pa02-pa12)**2)
ss=ss+np.sum((pa03-pa13)**2)
ss=ss+np.sum((pa04-pa14)**2)
ss=ss+np.sum((pa05-pa15)**2)
ss=ss+np.sum((pa06-pa16)**2)
print("ss",ss)
x_data = np.loadtxt('inputX2.csv', delimiter=',')#np.random.rand(1000,6)-0.5
#x_data[:,6]=x_data[:,3]
#x_data[:,7]=x_data[:,3]
#x_data[:,8]=x_data[:,3]
#x_data[:,9]=x_data[:,3]
#print(x_data)
#v1= x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]#x_data[:,0] * 4 + 1
v10=(x_data[:,0]+ x_data[:,1]) **2*2-6*np.exp(x_data[:,2])+2*np.sin(x_data[:,3])-x_data[:,4]**3+1.5*x_data[:,5]**2+25 
v11=(x_data[:,0]) **2 -2*np.exp(x_data[:,2]+0.3)+2*np.cos(x_data[:,3])-20
v12 = 2*(x_data[:,4]) **3 +np.exp(x_data[:,5])
v1=np.transpose(np.array([v10, v11, v12]))
#np.loadtxt('outputY2.csv', delimiter=',')#x_data[:,0] *x_data[:,0] *4 + 2*x_data[:,1]-6*x_data[:,2]+2*x_data[:,3]-x_data[:,4]+1.5*x_data[:,5]+10
#print(v1)
v2=sess.run(yin3,feed_dict={x1: x_data, is_training: False})
#print(v2.T)
print(np.mean(np.linalg.norm(v1-v2,axis=1)))
inputXT=np.random.rand(10000,6)-0.5
outputb=np.zeros((10000,3))#+outputY.mean(axis=0)
print("pingjun",outputY.mean(axis=0))



