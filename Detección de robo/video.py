# -*- coding: utf-8 -*-


import cv2
import time
import argparse
from imutils.video import FPS
import os
import sys
import numpy as np
import torchvision.transforms as transforms
import Jetson.GPIO as GPIO
from queue import Queue
from PIL import Image
import torch
import torch.nn as nn
from threading import Thread
from collections import Counter
from datetime import datetime



class VideoFrame:
  
  def __init__(self,source):
    self.source=source

    self.cont=0
    self.video=cv2.VideoCapture(self.source)
    self.fpsvideo=self.video.get(cv2.CAP_PROP_FPS)
    
    self.new=0
    self.old=0
    self.fps=0
    
    (self.success,self.frame)=self.video.read()
    cv2.imshow("Video",self.frame)
    self.stopped=True
    self.t=Thread(target=self.get,args=())
    self.t.daemon=True


  def start(self):
    self.stopped=False
    self.t.start()
    #self.t2.start()
    
  def get(self):
    global q
    while True:
      if self.stopped:
        break
      (self.success,self.frame)=self.video.read()

      if not self.success:
        self.stop()
      else:
        
        frame_color=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_color=ImageFile(frame_color).__getitem__()
        q=torch.cat((q[1:],frame_color),dim=0)
        

             

  def stop(self):
    self.stopped=True

    print("Video terminado")
  
  def show(self):
    if self.success:
    
    	self.new=time.time()
    	self.fps=1/(self.new-self.old)
      
    	cv2.putText(self.frame,"FPS: {:.2f}".format(self.fps),(220,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv2.LINE_4)
    	if robo==1:
    	  self.frame=cv2.putText(self.frame,'Alerta',(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_4)
    	  GPIO.output(led_pin,GPIO.HIGH)
    	else:
    	  GPIO.output(led_pin,GPIO.LOW)
    	self.old=self.new
    	cv2.imshow("Video",self.frame)

    
class Sonido:
  def __init__(self):
    self.stopped=True
    self.t=Thread(target=self.alerta,args=())
    self.t.daemon=True
  
  
  def start(self):
    self.stopped=False
    self.t.start()

  def alerta(self):
    global flagsonido
    while True:
      if self.stopped:
       break
      if robo==1:
        if flagsonido is True:
          os.system("mpg123 Aviso.mp3")
          flagsonido=False
      else:
        flagsonido=True
  
  def stop(self):
    self.stopped=True
  
class Evaluation:
  
  def __init__(self,modelo):
    self.started=torch.cuda.Event(enable_timing=True)
    self.end=torch.cuda.Event(enable_timing=True)

    self.stopped=True
    self.clasification=Queue(maxsize=2)
    self.t=Thread(target=self.inference,args=())
    self.t.daemon=True
    self.modelo=modelo.eval()

  def inference(self):
    global robo
    l=[]
    c=0
    while True:
        if self.stopped:
          mean=sum(l)/c
          print("Tiempo inferencia promedio (ms)",mean)
          break
        
        
        self.secuencia=torch.unsqueeze(q,dim=0)
        
        self.sequencia=self.secuencia.to(device)
        self.started.record()
        outputs=self.modelo(self.sequencia)
        self.end.record()
        
        torch.cuda.synchronize()
        timer=self.started.elapsed_time(self.end)
        
        l.append(timer)
        c+=1

        pred=(outputs>0.0).float()
        self.clasification.put(pred.item())

        if self.clasification.full() is True:
          if self.clasification.queue[0]==1==self.clasification.queue[1]:
            robo=1
          else:
            robo=0
          self.clasification.get()

  def start(self):
    self.stopped=False
    self.t.start()    
    
  def stop(self):
    self.stopped=True

def getFrame(source):

    getVideo=VideoFrame(source)
    getVideo.start()
    eva=Evaluation(modelo)
    eva.start()
    x=Sonido()
    x.start()
    delay=int(1000/getVideo.fpsvideo)
    fps=FPS().start()

    while True:
      if getVideo.stopped:
        fps.stop()
        x.stop()
        eva.stop()

        break

      getVideo.show()
      cv2.waitKey(delay)
      fps.update()
    print("Tiempo transcurrido del video: {:.2f}".format(fps.elapsed()))    
    print("FPS aproximados: {:.2f}".format(fps.fps()))
    GPIO.output(led_pin,GPIO.LOW)
    getVideo.video.release()
    cv2.destroyAllWindows()


        
class LSTM(nn.Module):
  def __init__(self,hidden_size,num_layers,num_classes):
    super(LSTM, self).__init__()
    
    self.hidden_size=hidden_size
    self.num_layers=num_layers
    self.num_classes=num_classes
    self.backbone = torch.hub.load('', 'custom', source='local', path='yolov5n_backbone.pt', autoshape=False)
    features=256
    for param in self.backbone.parameters():
      param.requires_grad = False
    self.pool=nn.AdaptiveAvgPool2d(1)
    self.flat=nn.Flatten()
    self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)    
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    batch, seq, C,H, W = x.size()
    c_in = x.view(batch * seq, C, H, W)
    bone_out = self.backbone(c_in)
    pool_out=self.pool(bone_out)
    flat_out=torch.flatten(pool_out)
    r_out, (h_n, h_c) = self.lstm(flat_out.view(batch,seq,-1))
    out = r_out[:, -1, :]
    out=self.fc(out)
    return out


class ImageFile:
  def __init__(self, file):
    global valid_transform
    self.imagefiles=file
    self.transform=valid_transform
  def __len__(self):
    return len(self.imagefiles)
  def __getitem__(self):
    img=Image.fromarray(self.imagefiles)
    frame=self.transform(img)

    return torch.unsqueeze(frame,dim=0)
  
def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--source",default=0)
  args=vars(ap.parse_args())
  getFrame(args["source"])

if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    led_pin=7
    GPIO.setup(led_pin,GPIO.OUT,initial=GPIO.LOW)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    modelo = LSTM(51,3,1)
    path='Modelofinal.pth'
    cp=torch.load(path,map_location='cpu')
    modelo.load_state_dict(cp)
    modelo.to(device)
    q=torch.zeros(20,3,320,320)
    robo=0
    flagsonido=True
    valid_transform=transforms.Compose([
      transforms.Resize((320,320)),
      transforms.ToTensor(),  
      transforms.Normalize([0.4335,0.4272,0.4271],[0.2497,0.2502,0.2524]),
                           ])
    main()
