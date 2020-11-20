#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
#import ipdb
import time
import cv2
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import fire
from elapsedtimer import ElapsedTimer
import pathlib
from pathlib import Path
import gc;gc.collect()
from tensorflow.python import pywrap_tensorflow
#print('tensorflow version:',tf.__version__)


# In[6]:


class VideoCaptioning:
    
    
    def __init__(self,path_prj,caption_file,feat_dir,
                 cnn_feat_dim=4096,h_dim=512,
                 lstm_steps=80,video_steps=80,
                 out_steps=20, frame_step=80,
                 batch_size=1,learning_rate=1e-4,
                 epochs=1,model_path=None,
                 mode='test'):

        self.dim_image = cnn_feat_dim
        self.dim_hidden = h_dim
        self.batch_size = batch_size
        self.lstm_steps = lstm_steps
        self.video_lstm_step=video_steps
        self.caption_lstm_step=out_steps
        self.path_prj = Path(path_prj)
        self.mode = mode
        if mode == 'train':
            self.train_text_path = self.path_prj / caption_file
            self.train_feat_path = self.path_prj / feat_dir
        else:
            self.test_text_path = self.path_prj / caption_file
            self.test_feat_path = self.path_prj / feat_dir
        #self.test_text_path = self.path_prj / test_file
        #self.test_feat_path = self.path_prj / feat_dir    
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.frame_step = frame_step
        self.model_path = model_path

    def build_model(self):
        tf.compat.v1.reset_default_graph()
        #tf.Graph()
        tf.compat.v1.disable_eager_execution()
        # Defining the weights associated with the Network
        with tf.device('/cpu:0'): 
            #print(self.n_words)
            #print(self.dim_hidden)
            #return
            #random_tf=tf.random_uniform([8423,512], -0.1, 0.1)
            #self.word_emb=tf.Variable(random_tf)
            self.word_emb = tf.Variable(tf.random.uniform([self.n_words, self.dim_hidden], -0.1, 0.1), name='word_emb')
            print("word_emb",self.word_emb)

        
       
        self.lstm1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
        self.encode_W = tf.Variable( tf.random.uniform([self.dim_image,self.dim_hidden], -0.1, 0.1), name='encode_W')
        self.encode_b = tf.Variable( tf.zeros([self.dim_hidden]), name='encode_b')
        
        self.word_emb_W = tf.Variable(tf.random.uniform([self.dim_hidden,self.n_words], -0.1,0.1), name='word_emb_W')
        self.word_emb_b = tf.Variable(tf.zeros([self.n_words]), name='word_emb_b')
        
        # Placeholders 
        video = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.video_lstm_step, self.dim_image])
        video_mask = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.video_lstm_step])

        caption = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.caption_lstm_step+1])
        caption_mask = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        print("video_flat",video_flat)
        image_emb = tf.compat.v1.nn.xw_plus_b( video_flat, self.encode_W,self.encode_b ) 
        print("image_emb",image_emb)
        #using image embedding to reduce the dimension to 512
        image_emb = tf.reshape(image_emb, [self.batch_size, self.lstm_steps, self.dim_hidden])
        print("image_emb_reshaping",image_emb) 
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])
        print(self.lstm1.state_size) 
        print(self.lstm2.state_size) 
        probs = []
        loss = 0.0

        #  Encoding Stage 
        for i in range(0, self.video_lstm_step):
            if i > 0:
                tf.compat.v1.get_variable_scope().reuse_variables()

            with tf.compat.v1.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)
                print("encoding output1 state1", output1,state1)
            with tf.compat.v1.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)
                print("encoding output2 state2", output2,state2)
        #  Decoding Stage  to generate Captions 
        for i in range(0, self.caption_lstm_step):
            print("iteration:",i)
            with tf.device("/cpu:0"):# looks for the id's from word embedding
                current_embed = tf.compat.v1.nn.embedding_lookup(self.word_emb, caption[:, i])

            tf.compat.v1.get_variable_scope().reuse_variables()

            with tf.compat.v1.variable_scope("LSTM1"):
                print("decoding input state1 from previous loop", output1,state1)

                output1, state1 = self.lstm1(padding, state1)
                print("decoding output1 state1", output1,state1)   
            with tf.compat.v1.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1), state2)
                print("decoding output2 state2", output2,state2) 
                print("current_embed:",current_embed)
            labels = tf.expand_dims(caption[:, i+1], 1)
            print("labels:",labels)
            print("caption:",caption)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            print("indices:",indices)
            concated = tf.concat([indices, labels],1)
            onehot_labels = tf.compat.v1.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)
            print("onehot_labels:",onehot_labels)
            logit_words = tf.compat.v1.nn.xw_plus_b(output2, self.word_emb_W, self.word_emb_b)
            print(logit_words)      
        # Computing the loss     
            cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,logits=logit_words)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)
            print(logit_words)
            print(output2)
            print(probs)   
            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            print("current_loss",current_loss)
            loss = loss + current_loss
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),reuse=tf.compat.v1.AUTO_REUSE) as scope:
            train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(loss)    
        #allops=tf_graph. get_operations()
        #print(allops)
        #return
        return loss,video,video_mask,caption,caption_mask,probs,train_op


    def build_generator(self):
        #tf.Graph()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        with tf.device('/cpu:0'):
            self.word_emb = tf.Variable(tf.random.uniform([self.n_words, self.dim_hidden], -0.1, 0.1), name='word_emb')


        self.lstm1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)

        self.encode_W = tf.Variable( tf.random.uniform([self.dim_image,self.dim_hidden], -0.1, 0.1), name='encode_W')
        self.encode_b = tf.Variable( tf.zeros([self.dim_hidden]), name='encode_b')

        self.word_emb_W = tf.Variable(tf.random.uniform([self.dim_hidden,self.n_words], -0.1,0.1), name='word_emb_W')
        self.word_emb_b = tf.Variable(tf.zeros([self.n_words]), name='word_emb_b')
        video = tf.compat.v1.placeholder(tf.float32, [1, self.video_lstm_step, self.dim_image])
        video_mask = tf.compat.v1.placeholder(tf.float32, [1, self.video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.compat.v1.nn.xw_plus_b(video_flat, self.encode_W, self.encode_b)
        image_emb = tf.reshape(image_emb, [1, self.video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.video_lstm_step):
            if i > 0:
                tf.compat.v1.get_variable_scope().reuse_variables()

            with tf.compat.v1.variable_scope("LSTM1"):#,reuse=tf.compat.v1.AUTO_REUSE) as scope:
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.compat.v1.variable_scope("LSTM2"):#,reuse=tf.compat.v1.AUTO_REUSE) as scope:
                output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)

        for i in range(0, self.caption_lstm_step):
            tf.compat.v1.get_variable_scope().reuse_variables()

            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.compat.v1.nn.embedding_lookup(self.word_emb, tf.ones([1], dtype=tf.int64))

            with tf.compat.v1.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.compat.v1.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1), state2)

            logit_words = tf.compat.v1.nn.xw_plus_b(output2, self.word_emb_W, self.word_emb_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)
            #print("generated_words:",generated_words)

            with tf.device("/cpu:0"):
                current_embed = tf.compat.v1.nn.embedding_lookup(self.word_emb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)
        #var_list=[v.name for v in tf.compat.v1.get_variable(self.encode_W)]
        #print(var_list)
        return video,video_mask,generated_words,probs,embeds


    def get_data(self,text_path,feat_path):
        text_data = pd.read_csv(text_path, sep=',',engine='python')#read the csv into a dataframe
        text_data = text_data[text_data['Language'] == 'English']# filter for english
        #print(text_data.shape)
        #add a column contanating videoid,start and end time to match the video file names
        text_data['video_path'] = text_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.npy', axis=1)
        #print("first",text_data['video_path'] )
        # use the column name above and add the path or each videopath name
        text_data['video_path'] = text_data['video_path'].map(lambda x: os.path.join(feat_path, x))
        #print("second",text_data['video_path'] )

        #print(text_data.shape)
        text_data.to_csv(r'video_captioning_dataset\YouTubeClips\getdata_textdata.csv')
        
        text_data = text_data[text_data['video_path'].map(lambda x: os.path.exists(x))]#filter by video path
        #print("mapping",text_data)
        text_data = text_data[text_data['Description'].map(lambda x: isinstance(x, str))]#filter description as string
        #print("description",text_data)
        unique_filenames=sorted(text_data['video_path'].unique())
        #print("unique_filenames",unique_filenames)
        #text_data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
        data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
        #print(data)
        return data

    def train_test_split(self,data,test_frac=0.2):
        indices = np.arange(len(data))
        #np.random.shuffle(indices)
        train_indices_rec = int((1 - test_frac)*len(data))
        indices_train = indices[:train_indices_rec]
        indices_test = indices[train_indices_rec:]
        data_train, data_test = data.iloc[indices_train],data.iloc[indices_test]
        #print(data_train.head())
        #print(data_test.head())
        data_train.reset_index(inplace=True)
        data_test.reset_index(inplace=True)
        data_train.to_csv(r'video_captioning_dataset\YouTubeClips\traindata.csv')
        data_test.to_csv(r'video_captioning_dataset\YouTubeClips\testdata.csv')
        
        return data_train,data_test

    def get_test_data(self,text_path,feat_path):
        text_data = pd.read_csv(text_path, sep=',',engine='python')
        text_data = text_data[text_data['Language'] == 'English']
        text_data['video_path'] = text_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.npy', axis=1)
        text_data['video_path'] = text_data['video_path'].map(lambda x: os.path.join(feat_path, x))
        text_data = text_data[text_data['video_path'].map(lambda x: os.path.exists( x ))]
        text_data = text_data[text_data['Description'].map(lambda x: isinstance(x, str))]
    
        unique_filenames = sorted(text_data['video_path'].unique().tolist())
        #unique_filenames = text_data.apply(lambda x:text_data['video_path'].unique())
        test_data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
        #save data to csv
        test_data.to_csv(r'video_captioning_dataset\YouTubeClips\alltextdata.csv')
        return test_data       
        
    def create_word_dict(self,sentence_iterator, word_count_threshold=5):
        
        word_counts = {}
        sent_cnt = 0
        
        for sent in sentence_iterator:
            sent_cnt += 1
            for w in sent.lower().split(' '):
               word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        
        idx2word = {}
        idx2word[0] = '<pad>'
        idx2word[1] = '<bos>'
        idx2word[2] = '<eos>'
        idx2word[3] = '<unk>'
    
        word2idx = {}
        word2idx['<pad>'] = 0
        word2idx['<bos>'] = 1
        word2idx['<eos>'] = 2
        word2idx['<unk>'] = 3
    
        for idx, w in enumerate(vocab):
            word2idx[w] = idx+4
            idx2word[idx+4] = w
    
        word_counts['<pad>'] = sent_cnt
        word_counts['<bos>'] = sent_cnt
        word_counts['<eos>'] = sent_cnt
        word_counts['<unk>'] = sent_cnt
    
        return word2idx,idx2word
        
        
        
    
    def train(self):
        print(self.train_text_path)
        print(self.train_feat_path)
 
        data = self.get_data(self.train_text_path,self.train_feat_path)
        
        self.train_data,self.test_data = self.train_test_split(data,test_frac=0.2)
        self.train_data.to_csv(f'{self.path_prj}/train.csv',index=False)
        self.test_data.to_csv(f'{self.path_prj}/test.csv',index=False)

        print(f'Processed train file written to {self.path_prj}/train_corpus.csv')
        print(f'Processed test file written to {self.path_prj}/test_corpus.csv')
                

        train_captions = self.train_data['Description'].values
        test_captions = self.test_data['Description'].values
    
        captions_list = list(train_captions) 
        captions = np.asarray(captions_list, dtype=np.object)
        print("captions",captions)
        captions = list(map(lambda x: x.replace('.', ''), captions))
        captions = list(map(lambda x: x.replace(',', ''), captions))
        captions = list(map(lambda x: x.replace('"', ''), captions))
        captions = list(map(lambda x: x.replace('\n', ''), captions))
        captions = list(map(lambda x: x.replace('?', ''), captions))
        captions = list(map(lambda x: x.replace('!', ''), captions))
        captions = list(map(lambda x: x.replace('\\', ''), captions))
        captions = list(map(lambda x: x.replace('/', ''), captions))
        print("captions_cleaned",captions)
        self.word2idx,self.idx2word = self.create_word_dict(captions, word_count_threshold=0)
        
        np.save(self.path_prj/ "word2idx",self.word2idx)
        np.save(self.path_prj/ "idx2word" ,self.idx2word)
        self.n_words = len(self.word2idx)
        #print(len(self.word2idx))
    
        tf_loss,tf_video,tf_video_mask,tf_caption,tf_caption_mask,tf_probs,train_op= self.build_model()
        print("tf_video",tf_video)
        print("tf_video_mask",tf_video_mask)
        print("tf_caption",tf_caption)
        print("tf_caption_mask",tf_caption_mask)
        #sess_count=self.tf.compat.v1.InteractiveSession._active_session_count
        #if sess_count>0:
          #  print(sess_count)
           # self.tf.compat.v1.InteractiveSession.close()
        sess = tf.compat.v1.InteractiveSession()
        tf.compat.v1.global_variables_initializer().run()
        #saver = tf.compat.v1.train.Saver(max_to_keep=100, write_version=1)
        saver = tf.compat.v1.train.Saver(max_to_keep=100)
        var_name_list=[v.name for v in tf.compat.v1.global_variables()]
        #print(var_name_list)
        
    
    
        loss_out = open('loss.txt', 'w')
        val_loss = []
    
        for epoch in range(0,self.epochs):
            val_loss_epoch = []
    
            index = np.arange(len(self.train_data))

            self.train_data.reset_index()
            np.random.shuffle(index)
            self.train_data = self.train_data.loc[index]
    
            current_train_data = self.train_data.groupby(['video_path']).first().reset_index()
            print("current_train_data:",current_train_data)
    
            for start, end in zip(
                    range(0, len(current_train_data),self.batch_size),
                    range(self.batch_size,len(current_train_data),self.batch_size)):
    
                start_time = time.time()
    
                current_batch = current_train_data[start:end]
                current_videos = current_batch['video_path'].values
    
                current_feats = np.zeros((self.batch_size, self.video_lstm_step,self.dim_image))
                #print(self.video_lstm_step)
                #print(self.dim_image)
                current_feats_vals = list(map(lambda vid: np.load(vid),current_videos))
                #print("current_feats_vals",current_feats_vals[0:5])
                current_feats_vals = np.array(current_feats_vals) 
    
                current_video_masks = np.zeros((self.batch_size,self.video_lstm_step))
    
                for ind,feat in enumerate(current_feats_vals):
                    current_feats[ind][:len(current_feats_vals[ind])] = feat
                    current_video_masks[ind][:len(current_feats_vals[ind])] = 1
    
                current_captions = current_batch['Description'].values
                current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
                current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
                current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
                current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
                current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
                current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
                current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
                current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
                current_captions = list(map(lambda x: x.replace('/', ''), current_captions))
                print("current_captions",current_captions)

    
                for idx, each_cap in enumerate(current_captions):
                    word = each_cap.lower().split(' ')
                    if len(word) < self.caption_lstm_step:
                        current_captions[idx] = current_captions[idx] + ' <eos>'
                    else:
                        new_word = ''
                        for i in range(self.caption_lstm_step-1):
                            new_word = new_word + word[i] + ' '
                        current_captions[idx] = new_word + '<eos>'
    
                current_caption_ind = []
                for cap in current_captions:
                    current_word_ind = []
                    for word in cap.lower().split(' '):
                        if word in self.word2idx:
                            current_word_ind.append(self.word2idx[word])
                        else:
                            current_word_ind.append(self.word2idx['<unk>'])
                    current_caption_ind.append(current_word_ind)
    
                current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=self.caption_lstm_step)
                current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
                current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
                nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) ))
                print("current_caption_ind",current_caption_ind)
                print("current_caption_matrix",current_caption_matrix)
                for ind, row in enumerate(current_caption_masks):
                    row[:nonzeros[ind]] = 1
    
                probs_val = sess.run(tf_probs, feed_dict={
                    tf_video:current_feats,
                    tf_caption: current_caption_matrix
                    })
                print("current video feat",current_feats)
                print("current video mask",current_video_masks)
                print("current caption",current_caption_matrix)
                print("current caption mask",current_caption_masks)
                #print("probs_val",probs_val)
                _,loss_val = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_video: current_feats,
                            tf_video_mask : current_video_masks,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
                val_loss_epoch.append(loss_val)
    
                print('Batch starting index: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
                loss_out.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
    
            # draw loss curve every epoch
            print(val_loss_epoch)
            val_loss.append(np.mean(val_loss_epoch))
            #val_loss.append(val_loss_epoch)
            plt_save_dir = self.path_prj / "loss_imgs"
            print(plt_save_dir)
            print(val_loss)
            
            plt_save_img_name = 'epoch '+ str(epoch) + '.png'
            print(plt_save_img_name)
            plt.plot(range(len(val_loss)),val_loss, color='g')
            plt.grid(True)
            plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))
    
            if np.mod(epoch,9) == 0:
                print ("Epoch ", epoch, " is done. Saving the model ...")
                saver.save(sess, os.path.join(self.path_prj, 'trainmodel'), global_step=epoch)
    
        loss_out.close()
        
        
    
    def inference(self,vfile):

        #self.test_data = self.get_test_data(self.test_text_path,self.test_feat_path)
        #test_text_data = pd.read_csv(self.test_text_path, sep=',',engine='python')
        #print(test_text_data.shape)
        #test_videos = self.test_data['video_path'].unique()
        #print(test_videos)
    
        self.idx2word = pd.Series(np.load(self.path_prj / "idx2word.npy",allow_pickle=True).tolist())
    
        self.n_words = len(self.idx2word)
        #print("number of words:",self.n_words)
        video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = self.build_generator()
    
        sess = tf.compat.v1.InteractiveSession()
        #tf.compat.v1.global_variables_initializer().run()
        #print(self.model_path)
        saver = tf.compat.v1.train.Saver()
      

        #print(saver)
        
        mypath=os.path.join(self.path_prj,'trainmodel-99')
        
        
        
        saver.restore(sess,mypath)
        testfeat=vfile+'.'+'npy'
        feat_fold='vgg16_feat'
        feat_fold_path=os.path.join(self.path_prj,feat_fold)
        video_feat_path=os.path.join(feat_fold_path,testfeat)
        #print(video_feat_path)
        idx=0
    
        f = open(f'{self.path_prj}/video_captioning_results.txt', 'w')
        #for idx, video_feat_path in enumerate(test_videos):
        video_feat = np.load(video_feat_path)[None,...]#adding one more dimension to the tensor
        #print(video_feat)
        #print(idx)
        #print(video_feat_path)
        #print(video_feat.shape)
        #print(video_feat.shape[1])
        #print(self.frame_step)
        #if video_feat.shape[1] == self.frame_step:
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        #else:
        #continue
            #print(caption_tf)
            #print(video_tf)
            #print(video_feat)
            #print(video_mask_tf)
        gen_word_idx = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            #print("wordindex:",gen_word_idx)
        gen_words = self.idx2word[gen_word_idx]
            #print(gen_words)
        punct = np.argmax(np.array(gen_words) == '<eos>') + 1
            #print("punct:",punct)
        gen_words = gen_words[:punct]
            #print(gen_words)
            #print("gen_words from list:",gen_words)
        gen_sent = ' '.join(gen_words)
        gen_sent = gen_sent.replace('<bos> ', '')
        gen_sent = gen_sent.replace(' <eos>', '')
            #print(f'Video path {video_feat_path} : Generated Caption {gen_sent}')
        #print(gen_sent,'\n')
        f.write(video_feat_path + '\n')
        f.write(video_feat_path +','+gen_sent)
        f.write(gen_sent + '\n\n')
        #sess.close()
        return gen_sent    
    def process_main(self):
        if self.mode == 'train':
            self.train()
        else:
            self.inference()
        


# In[7]:



#if __name__ == '__main__':
    #with ElapsedTimer('Video Captioning'):
    
 # fire.Fire(VideoCaptioning)
  


# In[8]:


#path_prj='video_captioning_dataset\YouTubeClips\Iteration_vgg16'
#caption_file='video_corpus.csv'
#feat_dir='vgg16_feat'
#feat_path='video_captioning_dataset\YouTubeClips\iteration4\\dest'
#text_path='video_captioning_dataset\YouTubeClips\iteration4\\video_corpus.csv'
#caption_file='test.csv'
#mytest=VideoCaptioning(path_prj,caption_file,feat_dir)
#mydata=mytest.get_data(text_path,feat_path)
#mydata_split=mytest.train_test_split(mydata,test_frac=0.2)
#mytest_testdata=mytest.get_test_data(text_path,feat_path)
#my_train=mytest.train()
#len(mytest_testdata)
#pred_vfile='0hyZ__3YhZc_279_283'
#myinfer=mytest.inference(pred_vfile)
#print(myinfer)


# In[ ]:





# In[ ]:





# In[ ]:




