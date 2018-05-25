# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):
    #our inpuct vector has 5 dimension(input neurons),3 signals(output neurons) + oriantation + minus oriantation, self is an object for pointing our values
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)#all the neurons in input layer must be connected to the hidden layer. 30 is the number of the neurons we made the fullconnection between the input layer and the hidden layer
        self.fc2 = nn.Linear(30, nb_action)#second fullconnection is beetween output layer(nb_action which it is equal to 3 because we 3 possible actions) and the hidden layer
    
    def forward(self, state):
        x = F.relu(self.fc1(state))#activating hidden neurons(x is represent the hidden neurons)
        q_values = self.fc2(x)#output neurons corespond to our actions but these are not action directly these are q values. We are building Deep learning model that combine with deep learning model to q learning there for we are using q learning to get our q values for each our actions
        return q_values
    
# Implementing Experience Replay
class ReplayMemory(object): 
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples=zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)),samples)
    
# Implementing Deep Q Learning
        
#bir önceki adımları burada kullanacağız. Derin öğrenmenin mimarisi ve hafızasını yani . Tüm derin öğreme algoritması tek bir sınıfa sığacak
#bu sınıf yapay zekayı mplement ettiğimiz sınıf.
        #ilk init fonksiyonunda değişkenler üretiyoruz ki nesnelere bağlayabilelim
class Dqn():
    def __init__(self, input_size, nb_action, gamma):#bu değişkenler demek oluyorki  artık her dqn sınıfı çağrıldığında bu değişkenler girlerek çağrılmalı
        self.gamma=gamma
        self.reward_window = [] #bu sliding window reward benim son 100 rewardımın ortalaması ve bunu ai'ın evrimini değerlendiriken kullanıcaz. son 100 reward'ın zaman içinde ortalamasının artışını gözlemlemek istiyoruz.
        self.model=Network(input_size, nb_action)        #burası neural networkün oluşturulduğu kısım. Bradaki kod parçası sayesinde we creat one neural network for learning model
        self.memory=ReplayMemory(100000)#one hundred transition into memory, we will sample from this memory to get small number of random transaction and that on which the will ok.#replay meory calss için bir nesne
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001 )#ai'ın optime edildiği kısım learning rate diye bir argument ekledik.bunu geniş bir aralık almalıyız çünkü ai düzgün birşekilde öğrensin. öğrenmesi için ona zaman tanımalıyız. 
        self.last_state = torch.Tensor(input_size).unsqueeze(0)#last state 5 dimesionun vektörü. 5 dimesion = 3 signal - orientation and minus oriantation, but for py torch should be more than a vector it should be torch tensor. Sadece torch tensor olması yeterli değil aynı zamanda bir tane daha dimensiona ihtiyaç var. Yığına karşılık olarak gelen bir Fake dimension diyoruz.
        #ve bu yüzden last state , nn'in girişi olacak fakat  ama genelde pytorch, tensor flow, kares gibi nn'ler ile çalışırken input vector kendisi gibi basit bir vector olamaz. Bir yığının içide olması gerekir. 
        #the network can only accept batch of input observations and there for not only will create a tensor for input state vectors but also we will create this fake dimesion corresponding to the batch. 
        #unsqueeze(0 ) ile fake dimension became first dimension
        #torchçtensor ise tensor dimension ve inputtaki 5 elementti içeriyor
        self.last_action =0 # 0,1,2 olabilir. action2rotation = [0,20,-20] buraya denk geliyor. 1 olsa mesala araba aksiyon halindeyken 2o derece dönmesi demek
        self.last_reward=0
        
#birim zamanda yapılması gereken doğru aksiyonu seçmeyi sağlayan fonsiyon
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        return action.data[0,0]
   # def select_action(self, state):
    #    probs=F.softmax(self.model(Variable(Variable(state, volatile=True))*0)#probabilitys for each actions. q values are the output of nn to get these outputs we need to get a network function. we already did that and we implement it in init func. 54.
        #when we call state yukarıda that state is actually going to be a torch tensor because we are gone be use this self.last_state to put it as the argument of the select action function. Tensorlerin çoğu bir değişken içinde kapalıdır. bu Aynı zamnda eğim içeriyor. Bu eğimi biz şimdi kullanmayacağız.  Buyüzden bu torch sensorü torch değişkenine çeviriyoruz. Sonrasında bu eğimi graf ta hiçbir nn modulünde istemediğimizi belirtiyoruz
        #eğimi  nn'in garfına katmayarak hafıza kazanıyoruz. That way we are improving the performance
        #sıcaklık parametresi sayesinde neural netwprk hangi askiyonu alması gerektiğine karar verecek. pozitif numara alıcak. kum yakınında 0 veya ona yakın bir numara alacak  böylece alacağı aksiyondan daha az emin olucak fakat derece arttıkca alacağı aksiyondan daha çok emin olacak.
        #*7 için T=7 sıcaklık derecemiz bu biz atadık şimdi böcek gibi hızlı hareket edecek . bu değeri zamanla arttırıcaz böylece hareketleri daha çok arabaya benzeycek.Yüksek olması tempature nin kazana q değerine yaklşıldığını gösterir
        #softmax([1,2,3])={0.04,0.11,0.85}  => softmax([1,2,3]*3) = {0,0.2,0.98}  it is about certanly of which direction we should decide to play. in this sample it is 3. action
     #   action = probs.multinomial()#getting random draw
        #action will be a random draw of the probability distribuition that we just created at this line before 
      #  return action.data[0,0]
        
#func. for learning side. We'll compare the output of the target to compute the last error then we are  going to back propagate
        
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):#this series are a transition of the markov decision process that is the base of deep w learning
    #bu yığınları hafızdan alıyoruz ki bunlar bziim geçişlerimiz aynı zamanda, böylece batch state'in her durumu için farklı çıkışlar elde edicez, bunu next batch state içinde yapıcaz çünkü ikisinede loss u hesaplamada ihtiyacımız var.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
    
    
    
    
    
    
    
    
    