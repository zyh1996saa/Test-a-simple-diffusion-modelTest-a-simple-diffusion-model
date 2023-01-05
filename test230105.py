import numpy as np
import matplotlib.pyplot as plt
import random



# In[]
#生成原始数据集，10000个三角形，第一个点在0.1-0.3，第二个点在0.8-1.2，第三个点在0.2-0.5
sample_size = 100000
data_shape = 3
ori_data = np.zeros((sample_size,data_shape))
ori_data[:,0] = np.random.uniform(low=0.1, high=0.3, size=sample_size)
ori_data[:,1] = np.random.uniform(low=0.8, high=1.2, size=sample_size)
ori_data[:,2] = np.random.uniform(low=0.2, high=0.5, size=sample_size)
# 采样标准正态分布
def sample_guass_noise(scale=1,size=(data_shape,)):
    return np.random.normal(loc=0,scale=1,size=size)

# 连乘alpha
def PI_alpha(alpha,t):
    temp = 1
    for i in range(t+1):
        temp *= alpha[i]
    return temp
# In[]
'''
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        random_int = random.randint(0,9999)
        print('random_int=%s'%random_int)
        plt.plot(ori_data[random_int,:])
        plt.show()
del order,random_int'''
# In[]
T = 200
# 定义beta
beta_t = np.linspace(0.0001,0.02,T)

# 定义alpha = 1- beta
alpha_t = 1 - beta_t

# 定义alpha_ba_t = alpha_1 * alpha_2 * ... * aplha_t 
alpha_ba_t = np.array([PI_alpha(alpha_t,t) for t in range(T)])

# 定义x0的系数 = sqrt(alpha_ba)
x0_t_coeff = np.sqrt(alpha_ba_t)

# 定义噪声z0的系数 = sqrt(1-alpha_ba)
noise_t_coeff = np.sqrt(1-alpha_ba_t)

t_for_each_data = np.random.randint(low=0, high=T, size=sample_size)

# In[]

# diffusion_tri
diff_tri = np.zeros((sample_size,data_shape))
noise_t = np.zeros((sample_size,data_shape))
for data_label in range(sample_size):
    t = t_for_each_data[data_label]
    noise_t[data_label,:] = sample_guass_noise()
    diff_tri[data_label,:] = (x0_t_coeff[t]*ori_data[data_label]).reshape(1,3) + (noise_t_coeff[t]*noise_t[data_label,:]).reshape(1,3)
# In[]
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        random_int = random.randint(0,sample_size-1)
        print('random_int=%s'%random_int)
        t = t_for_each_data[random_int]
        print('t=%s'%t)
        plt.plot(ori_data[random_int,:],color='b')
        plt.plot(diff_tri[random_int,:],color='r')
        plt.plot(noise_t[random_int,:],color='c')
        #print(x0_t_coeff[t],)
        plt.show()
del order,random_int
# In[]
import tensorflow as tf
# 定义神经网络

input0 = tf.keras.Input(shape=(data_shape,))
input1 = tf.keras.Input(shape=(1,))
x1 = tf.keras.layers.Dense(16,name='dense1',activation='gelu')(input1)
x1 = tf.keras.layers.Dense(36,name='dense2',activation='gelu')(x1)
x1 = tf.keras.layers.Dense(16,name='dense3',)(x1)
x0 = tf.keras.layers.Dense(36,name='dense4',activation='gelu',)(input0)
x0 = tf.keras.layers.Dense(16,name='dense5',activation='gelu',)(x0)
x = tf.keras.layers.Concatenate(axis=1)([x0, x1])
x = tf.keras.layers.Dense(36,name='dense6',activation='gelu',)(x)
x = tf.keras.layers.Dense(36,name='dense7',activation='gelu',)(x)
output = tf.keras.layers.Dense(data_shape,name='dense8',)(x)
model = tf.keras.Model([input0,input1], output)
# In[]

model_input0 = diff_tri
model_input1 = t_for_each_data/T
model_output = noise_t
model.compile(optimizer='Adam',loss='MSE')
model.fit([model_input0,model_input1],noise_t,epochs=100,batch_size=1024)


# In[]
'''
diff_tri[data_label,:] = (x0_t_coeff[t]*ori_data[data_label]).reshape(1,3) + (noise_t_coeff[t]*noise_t[data_label,:]).reshape(1,3)
'''
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        random_int = random_int = random.randint(0,sample_size-1)
        x0_true = ori_data[random_int]
        t = t_for_each_data[random_int]
        xt = (x0_t_coeff[t]*ori_data[random_int]).reshape(1,3) + (noise_t_coeff[t]*noise_t[random_int,:]).reshape(1,3)
        xt_read = diff_tri[random_int,:]
        
        
        noise_predict = model.predict([xt.reshape(1,3),np.array([[t/1000]])])
        noise_true = noise_t[random_int,:]
        
        x0_predict = (xt-noise_t_coeff[t]*noise_predict.reshape(3))/x0_t_coeff[t]
        x0_true_cal = (xt-noise_t_coeff[t]*noise_true.reshape(3))/x0_t_coeff[t]
        
        plt.plot(x0_predict.reshape(3),color='red')    
        plt.plot(x0_true_cal.reshape(3),color='pink')
        plt.plot(noise_predict.reshape(3),color='c')
        plt.plot(noise_true.reshape(3),color='b')
        print('random_int',random_int)
        plt.show()
# In[]
while 1:
    order = input('input something:')
    if order == 'quit':
        break
    else:
        for t in range(T,0,-1):
            if t == T:
                xt = sample_guass_noise()
            else:
                xt = xt_red_1
            noise_t_pre = model.predict([xt.reshape(1,3),np.array([[t/1000]])],verbose = 0).reshape(3)
            alpha = alpha_t[t-1]
            alpha_ba = alpha_ba_t[t-1]
            z = sample_guass_noise()
            xt_red_1 = (1/np.sqrt(alpha))*(xt-(1-alpha)*noise_t_pre/np.sqrt(1-alpha_ba))+z*np.sqrt(1-alpha)
            print('\r t=%s'%(t-1),end='\r')
        plt.plot(xt)
        plt.show()