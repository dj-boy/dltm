# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:18:01 2018

@author: Allen
"""
import collections
import re

class vocabulary:
    '处理数据集，建立词汇表和转换文档到矩阵'
    vocab={}
    reverse={}
    data=[]
    
    def __init__(self, x_text):
        '参数是文本向量的形式'
        self.__build_voca(x_text)
        
    def __build_voca(self,x_text):
        lines=[s for s in (re.split(' ',line) for line in x_text) if s!='']
        max_len = max([len(line) for line in lines])
        words = [item for line in lines for item in line]
        dic =collections.Counter(words)
        self.vocab['UNK']=0
        self.reverse[0]='UNK'
        index = 1
        
        for key in dic:
            if dic[key]>1:
                self.vocab[key]=index        
                self.reverse[index]=key
                index += 1
         
        self.data = self.__convert(lines, max_len)
        
    def __convert(self, lines, max_len):
        m_data = []
        for line in lines:
            m_line = []
            
            for item in line:
                if(item in self.vocab):
                    m_line.append(self.vocab[item])
                else:
                    m_line.append(self.vocab['UNK'])
            m_line += [0]*(max_len-len(line))
            m_data.append(m_line)
        
        return m_data
            
class pretrained():
    'load pretraind embedding of GloVe'
    embeds_size=0
    embeds=[]
    
    def __init__(self,fname, vocab, k):
        self.embeds_size = k
        self.__read__(fname, vocab)
        
    def __read__(self, fname, vocab):
        self.embeds = [[0] * self.embeds_size]*len(vocab)
        
        with open(fname, encoding='utf8') as f:
            for line in f.readlines():
                mlist = line.split(' ')
                index = vocab.get(mlist[0],-1)
                if(index==-1): 
                    continue
                else:
                    elist = [float(item) for idx, item in enumerate(mlist) if idx>0]       
                    self.embeds[index]=elist
                
                
                
                
                
                
                
        