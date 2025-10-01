import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pickle
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

def create_data_generator(x_train, y_train):
    datagen = ImageDataGenerator(
        rotation_range=10,      # 随机旋转范围
        width_shift_range=0.1,  # 随机水平平移
        height_shift_range=0.1, # 随机竖直平移
        shear_range=0.1,        # 随机剪切变换
        zoom_range=0.1,         # 随机缩放
        horizontal_flip=True,   # 随机水平翻转
        fill_mode='nearest'     # 填充模式
    )
    
    # 适合生成器
    datagen.fit(x_train)
    return datagen
def load_cifar100(data_dir):
    """ Load CIFAR-100 dataset from the given directory. """
    with open(os.path.join(data_dir, 'train'), 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
    with open(os.path.join(data_dir, 'test'), 'rb') as f:
        test_data = pickle.load(f, encoding='latin1')

    x_train = train_data['data'].reshape((len(train_data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y_train = train_data['fine_labels']
    x_test = test_data['data'].reshape((len(test_data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y_test = test_data['fine_labels']

    return (x_train, y_train), (x_test, y_test)

# 加载数据集
data_dir = 'cifar-100-python'
(x_train, y_train), (x_test, y_test) = load_cifar100(data_dir)

# 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)
datagen = create_data_generator(x_train, y_train)
# 定义一个 Shortcut Block
# 图像分 Patch 函数
def create_patches(images, patch_size):
    batch_size = tf.shape(images)[0]  # 动态获取 batch_size
    height = images.shape[1]
    width = images.shape[2]
    channels = images.shape[3]

    # 计算每个图像中 patch 的数量
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    
    # 使用 extract_patches 提取小块
    patches = tf.image.extract_patches(
        images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # 正确 reshape 为 (batch_size, num_patches_height * num_patches_width, patch_size * patch_size * channels)
    patches = tf.reshape(patches, (batch_size, num_patches_height * num_patches_width, patch_size * patch_size * channels))
    return patches
'''
def create_patches(images, patch_size):
    batch_size, height, width, channels = images.shape
    # 计算每个图像中 patch 的数量
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    
    # 使用 extract_patches 提取小块
    patches = tf.image.extract_patches(
        images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # 正确 reshape 为 (batch_size, num_patches_height * num_patches_width, patch_size * patch_size * channels)
    patches = tf.reshape(patches, (batch_size, num_patches_height * num_patches_width, patch_size * patch_size * channels))
    return patches
'''
# 定义位置嵌入
def position_embedding(num_patches, embedding_dim):
    positions = tf.range(start=0, limit=num_patches, delta=1)
    return layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)

def create_vit_block(input_tensor, name):
    x = layers.LayerNormalization(epsilon=1e-6, name=name + '_norm1')(input_tensor)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64, name=name + '_attention')(x, x)
    x = layers.Dropout(0.1)(x)
    x = x+input_tensor
    x2 = layers.LayerNormalization(epsilon=1e-6, name=name + 'z'+'_norm2')(x)
    x_ffn = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu',  name=name + 'z'+'_ffn_1')(x2)
    
    x_ffn = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu', name=name + 'z'+'_ffn_2')(x_ffn)
    x_ffn = layers.Dropout(0.1)(x_ffn)
    x_ffn = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=name +'z'+ '_ffn_3')(x_ffn)
    output = x + x_ffn

    return output, x
def shortcut_block(inputs, filters, block_num):
    x = layers.BatchNormalization(name=f'block{block_num}_bn1')(inputs)
    x = layers.ReLU(name=f'block{block_num}_relu1')(x)
    x = layers.Conv2D(filters,kernel_regularizer=tf.keras.regularizers.l2(0.0001), kernel_size=(3, 3), padding='same', name=f'block{block_num}_conv1')(x)

    x = layers.BatchNormalization(name=f'block{block_num}_bn2')(x)
    x = layers.ReLU(name=f'block{block_num}_relu2')(x)
    x = layers.Conv2D(filters,kernel_regularizer=tf.keras.regularizers.l2(0.0001), kernel_size=(3, 3), padding='same', name=f'block{block_num}_conv2')(x)

    # Shortcut connection
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', name=f'block{block_num}_shortcut')(inputs)
    return layers.add([x, shortcut], name=f'block{block_num}_output')

# 定义全连接层的 Shortcut Block
def fc_shortcut_block(inputs, units, block_num):
    '''
    x0 = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=f'fcc_blockz{block_num}_dense0')(inputs)
    
    x0 = layers.BatchNormalization(name=f'fcc_blockz{block_num}_bn0')(x0)
    x0 = layers.ReLU(name=f'fcc_blockz{block_num}_relu0')(x0)
    x = layers.Dense(units*2, kernel_regularizer=tf.keras.regularizers.l2(0.0001),name=f'fc_blockz{block_num}_dense1')(x0)

    x = layers.BatchNormalization(name=f'fc_blockz{block_num}_bn1')(x)
    x = layers.ReLU(name=f'fc_blockz{block_num}_relu1')(x)
    x = layers.Dense(units*2, kernel_regularizer=tf.keras.regularizers.l2(0.0001),name=f'fc_blockz{block_num}_dense2')(x)

    x = layers.BatchNormalization(name=f'fc_blockz{block_num}_bn2')(x)
    x = layers.ReLU(name=f'fc_blockz{block_num}_relu2')(x)
    #x = layers.Dense(units, name=f'fc_blockz{block_num}_dense2')(x)
    x = layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=f'fc_blockz{block_num}_dense3')(x)
    '''
    print("inutsde shape",inputs.shape)
    x0 = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=f'fcc_blockz{block_num}_dense0')(inputs)
    x = layers.BatchNormalization(name=f'fc_blockz{block_num}_bn1')(x0)
    x = layers.ReLU(name=f'fc_blockz{block_num}_relu1')(x)
    x = layers.Dense(units*2, kernel_regularizer=tf.keras.regularizers.l2(0.0001),name=f'fc_blockz{block_num}_dense1')(x)

    x = layers.BatchNormalization(name=f'fc_blockz{block_num}_bn2')(x)
    x = layers.ReLU(name=f'fc_blockz{block_num}_relu2')(x)
    #x = layers.Dense(units, name=f'fc_blockz{block_num}_dense2')(x)
    x = layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=f'fc_blockz{block_num}_dense2')(x)
    
    #x = layers.BatchNormalization(name=f'fc_blockz{block_num}_bn3')(x)
    #x = layers.ReLU(name=f'fc_blockz{block_num}_relu3')(x)
    #x = layers.Dense(units, name=f'fc_blockz{block_num}_dense2')(x)
    #x = layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=f'fc_blockz{block_num}_dense3')(x)
    
    # Shortcut connection
    #fc_shortcut = layers.Dense(units, name=f'fc_blockz{block_num}_shortcut')(inputs)
    fc_shortcut = layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name=f'fcc_blockz{block_num}_shortcut')(x0)#inputs
    return x #layers.add([x, fc_shortcut], name=f'fc_blockz{block_num}_output')

# 创建模型结构
def create_vit_model(input_shape, patch_size):
    inputs = layers.Input(shape=input_shape)
    # 创建 Patch
    patches = create_patches(inputs, patch_size)
    num_patches = patches.shape[1]

    # 嵌入层
    patch_embedding = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name='patch_embedding')(patches)
    pos_embedding = position_embedding(num_patches, 128)
    embeddings = patch_embedding+pos_embedding
    #x = layers.Conv2D(16, (3, 3), padding='same',name='inputc')(inputs)
    # 3个 Shortcut Block
    blocks = []
    shortcuts=[]
    vit_1, shortcut1 = create_vit_block(embeddings, 'vit_block_1')

    blocks.append(vit_1)
    shortcuts.append(shortcut1)
    vit_2, shortcut2 = create_vit_block(vit_1, 'vit_block_2')

    blocks.append(vit_2)
    shortcuts.append(shortcut2)
    vit_3, shortcut3 = create_vit_block(vit_2, 'vit_block_3')
    blocks.append(vit_3)
    shortcuts.append(shortcut3)
    vit_4, shortcut4 = create_vit_block(vit_3, 'vit_block_4')
    blocks.append(vit_4)
    shortcuts.append(shortcut4)
    vit_5, shortcut5 = create_vit_block(vit_4, 'vit_block_5')
    blocks.append(vit_5)
    shortcuts.append(shortcut5)
    # 全连接 Shortcut Block
    #x = layers.BatchNormalization(name=f'blockadasd')(vit_5)
    #x = layers.ReLU(name=f'blockzxczxc')(x)
    x = layers.Flatten(name='flatten')(vit_5)#
    outputs = fc_shortcut_block(x, units=256, block_num=6)
    blocks.append(outputs)
    final_output = layers.Activation('softmax', name='output')(outputs)#layers.Dense(100, activation='softmax', name='output')(outputs)
    model = models.Model(inputs, final_output)
    return model, blocks, shortcuts

early_stopping = EarlyStopping(
    monitor='val_accuracy',      # 监控验证准确率
    min_delta=0.001,             # 指标的最小变化量
    patience=6,                  # 在性能不再改进的情况下，最多允许的连续 epoch 数
    verbose=1,                   # 控制输出的详细程度
    mode='max',                  # 监控最大值（例如准确率）
    restore_best_weights=True    # 恢复最佳权重，以防在早期停止时保存次优模型
)
# 加载数据集
#(x_train, y_train), (x_test, y_test) = load_cifar100()

# 创建模型
patch_size = 4  # 设置patch大小
model, blocks, shortcuts = create_vit_model((32, 32, 3), patch_size)


# 编译模型
learning_rate=0.001
batch_size=32
optimizerz = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test),verbose=1,callbacks=[early_stopping])
#print("Finished fine-tuning the model")
        
for ii in range(1):
 '''
 for block in (6,):
    # 冻结其他层，训练当前 Block
    for i in model.layers:
        print(i.name)
    if block<6:
      for layer in model.layers:
        if (f'vit_block_{block}' in layer.name) :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
    else:
      for layer in model.layers:
        if (f'fc_blockz6_dense' in layer.name) or (f'fc_blockz6_bn' in layer.name) :
            layer.trainable = True
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = None
    
            
    model0 = models.Model(inputs=model.input, outputs=blocks[block-1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model0.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练当前 Block
    #x_augmented = datagen.flow(x_train, batch_size=len(x_train))
    #x_augmented = next(x_augmented) 
  
    if block<6 :
      for j in range(1):
        gc.collect()
        
        y_train_zeros = tf.zeros(((y_train.shape[0],)+blocks[block-1].shape[1:])) #((y_train.shape[0], num_classes))
        steps_per_epoch = len(x_train) // batch_size
        #model0.fit(x_train, y_train_zeros, epochs=5, batch_size=32, steps_per_epoch=steps_per_epoch)  # 输出为零
        model0.fit(datagen.flow(x_train, y_train_zeros, batch_size=32), epochs=20, steps_per_epoch=steps_per_epoch)  #
    else:    
        y_train_zeros = tf.zeros(((y_train.shape[0],)+blocks[block-1].shape[1:])) #((y_train.shape[0], num_classes))
        steps_per_epoch = len(x_train) // batch_size
        #model0.fit(x_train, y_train_zeros, epochs=5, batch_size=32, steps_per_epoch=steps_per_epoch)  # 输出为零
        model0.fit(datagen.flow(x_train, y_train_zeros, batch_size=32), epochs=8, steps_per_epoch=steps_per_epoch)  # 输出为零
    
    print(f"Finished training block {block}")
 '''
 for layer in model.layers:
     #layer.trainable = True
     if (f'b' in layer.name) :
            layer.trainable = True
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
            print(layer.name+'TRUE')
     else:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
            layer.trainable = True
 model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])

 print("Finished fine-tuning the model")

    
    
    
'''
for ii in range(1):

 for block in (6,):
    # 冻结其他层，训练当前 Block
    for i in model.layers:
        print(i.name)
    if block<6:
      for layer in model.layers:
        if (f'vit_block_{block}' in layer.name) :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
    else:
      for layer in model.layers:
        if (f'fc_blockz6_dense' in layer.name) or (f'fc_blockz6_bn' in layer.name) :
            layer.trainable = True
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = None
    
            
    model0 = models.Model(inputs=model.input, outputs=blocks[block-1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model0.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练当前 Block
    #x_augmented = datagen.flow(x_train, batch_size=len(x_train))
    #x_augmented = next(x_augmented) 
  
    if block<6 :
      for j in range(1):
        gc.collect()
        
        y_train_zeros = tf.zeros(((y_train.shape[0],)+blocks[block-1].shape[1:])) #((y_train.shape[0], num_classes))
        steps_per_epoch = len(x_train) // batch_size
        #model0.fit(x_train, y_train_zeros, epochs=5, batch_size=32, steps_per_epoch=steps_per_epoch)  # 输出为零
        model0.fit(datagen.flow(x_train, y_train_zeros, batch_size=32), epochs=20, steps_per_epoch=steps_per_epoch)  #
    else:    
        y_train_zeros = tf.zeros(((y_train.shape[0],)+blocks[block-1].shape[1:])) #((y_train.shape[0], num_classes))
        steps_per_epoch = len(x_train) // batch_size
        #model0.fit(x_train, y_train_zeros, epochs=5, batch_size=32, steps_per_epoch=steps_per_epoch)  # 输出为零
        model0.fit(datagen.flow(x_train, y_train_zeros, batch_size=32), epochs=8, steps_per_epoch=steps_per_epoch)  # 输出为零
    
    print(f"Finished training block {block}")
 for layer in model.layers:
     #layer.trainable = True
     if (f'b' in layer.name) :
            layer.trainable = True
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
            print(layer.name+'TRUE')
     else:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
            layer.trainable = True
 model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])

 print("Finished fine-tuning the model")    
'''
'''
      for layer in model.layers:
        if (f'vit_block_{block}' in layer.name) :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False      

      model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
      #model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))#
      #model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])
      print("Finished pretraining")
      
      for layer in model.layers:
        buchong=False
        for jjj in range(block,6):
           if f'vit_block_{jjj}' in layer.name:
              buchong=True
        if (f'fc_blockz1_dense' in layer.name) or (f'fc_blockz1_bn' in layer.name) or buchong :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
      model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
      #model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))#
      model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])
      
     
      for layer in model.layers:
        if (f'vit_block_{block}' in layer.name) :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
      model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
      #model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))#
      #model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])
      print("Finished fine-tuning the model")
      
      for layer in model.layers:
        buchong=False
        for jjj in range(block,6):
           if f'vit_block_{jjj}' in layer.name:
              buchong=True
        if (f'fc_blockz1_dense' in layer.name) or (f'fc_blockz1_bn' in layer.name) or buchong :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False
      model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
      #model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))#
      model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])
      

     else:
      for layer in model.layers:
        if (f'fc_blockz1_dense' in layer.name) or (f'fc_blockz1_bn' in layer.name) :
            layer.trainable = True
            print(layer.name+'TRUE')
        else:
            layer.trainable = False


      model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
      #model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))#
      model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test), verbose=1,callbacks=[early_stopping])
      print("Finished fine-tuning the model") 
    else:
      for layer in model.layers:
        #if (f'fc_blockz1_dense' in layer.name) or (f'fc_blockz1_bn' in layer.name) :
        layer.trainable = True
        print(layer.name+'TRUE')
        #else:
        #    layer.trainable = False
      model.compile(optimizer=optimizerz, loss='categorical_crossentropy', metrics=['accuracy'])
      #model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))#
      model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50, validation_data=(x_test, y_test),verbose=1,callbacks=[early_stopping])
      print("Finished fine-tuning the model")
        
'''
'''
    if f'block1_shortcut' in layer.name:
       layer.trainable = True
       print(layer.name+'True')
 '''

