# -*- coding:utf-8 -*-  
import numpy
import math
import matplotlib.pyplot as plt
import codecs

def Main():
    Data=numpy.matrix(numpy.zeros((1000,2),float));
    Size=numpy.random.randint(20,45,1000);
    m_Flie=codecs.open("E:\Python_Project\Spectral_Clustering\Data.txt","r","utf-8");
    Lines=m_Flie.readlines();
    i=0;
    for li in Lines:
        st=li.encode("utf-8");
        list=[st.strip().split("\r\n")]; 
        Data[i,:]=numpy.matrix(list[0][0]);
        i+=1;
    Seeds=numpy.zeros((5,2),float);
    Seeds=K_Means(Data,5);
    Detect_And_Draw(Data,Seeds);

def Detect_And_Draw(Data,Seeds):
    Data_lenth=Data.shape;
    Seed_lenth=Seeds.shape;
    print(Seeds);
    Result=numpy.zeros(Data_lenth[0],int);                  #用于记录每个样本的分类结果
    tem_Data=numpy.zeros((Seed_lenth[0],Data_lenth[1]),float);
    Similarity=numpy.matrix(numpy.zeros(Seed_lenth[0],float));          #用于存储相似度
    for i in range(Data_lenth[0]):                          #对所有样本            
        for j in range(Seed_lenth[0]): tem_Data[j,:]=Data[i,:];
        Difference=numpy.matrix(Seeds-tem_Data);
        for j in range(Seed_lenth[0]): Similarity[0,j]=Difference[j,:]*Difference[j,:].T;   
        Result[i]=Similarity.argmin();               #记录最相似的点
    #然后进行画图
    m_Color=[(0.8,0.0,0.7),(1.0,1.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(0.0,1.0,1.0)];
    Color_Box=[(0.8,0.0,0.7)]*1000;
    m_Maker=[(5,1)]*1000;
    for i in range(Data_lenth[0]):                          #对所有样本         
        if(i<200):
            m_Maker[i]=(5,1);
            Color_Box[i]=m_Color[Result[i]];           
        elif(i<400):
            m_Maker[i]=(4,1);
            Color_Box[i]=m_Color[Result[i]];       
        elif(i<600):
            m_Maker[i]=(3,1);
            Color_Box[i]=m_Color[Result[i]];     
        elif(i<800):
            m_Maker[i]=(4,0);
            Color_Box[i]=m_Color[Result[i]];     
        elif(i<1000):
            m_Maker[i]=(6,0);
            Color_Box[i]=m_Color[Result[i]];     
    for i in range(1000):
        plt.scatter(Data[i,0],Data[i,1],marker=m_Maker[i],color=Color_Box[i],alpha=0.7);
    plt.plot(Seeds[:,0],Seeds[:,1],'x',color=(1.0,0.0,0.0),linestyle=".");
    plt.savefig("E:\Python_Project\K-Means\M_Data1.png",dpi=266);
    plt.show();

def K_Means(Data,k):
    lenth=Data.shape;
    SeedPoint=numpy.random.randint(0,int(lenth[0]),k);      #随机生成k个点作为聚类中心
    Seeds=numpy.zeros((k,lenth[1]),float);
    OldSeeds=numpy.zeros((k,lenth[1]),float);
    for i in range(k): Seeds[i,:]=Data[SeedPoint[i],:];
    #下面就开始训练
    tem_Data=numpy.zeros((k,lenth[1]),float);
    Similarity=numpy.matrix(numpy.zeros(k,float));          #用于存储相似度
    Record=[0]*100;
    t=0;
    while(True):
        OldSeeds=Seeds;
        #首先用当前聚类中心对各个点分类
        Result=numpy.zeros((lenth[0],k),int);#用于记录训练中每个样本的分类结果
        for i in range(lenth[0]):#对所有样本            
            for j in range(k): tem_Data[j,:]=Data[i,:];
            Difference=numpy.matrix(Seeds-tem_Data);
            for j in range(k): Similarity[0,j]=Difference[j,:]*Difference[j,:].T;   
            Result[i,Similarity.argmin()]=1;#记录最相似的点
        #然后重新计算聚类中心
        Seeds=Result.T*Data;
        for j in range(k): Seeds[j,:]/=sum(Result.T[j,:]);
        #然后是停止条件
        Difference=numpy.matrix(Seeds-OldSeeds);
        Step=0;
        for j in range(k): Step+=float(Difference[j,:]*Difference[j,:].T);
        Record[t]=Step;
        t+=1;
        print(Step);
        if Step<0.001: break;
    return Seeds;
    #plt.plot(numpy.linspace(0,t,t),Record[0:t],'x',color=(1.0,0.0,0.0),linestyle="-");
    #plt.show();


##下面这一大段代码用于生成样本并保存在文件中，第一次使用即可，后面使用数据直接从文件中读取
def Organize_Data():
    Sigma = numpy.matrix([[1, 0],
                          [0, 1]]);
    mu1 = numpy.array([1, -1]);
    mu2 = numpy.array([5.5, -4.5]);
    mu3 = numpy.array([1, 4]);
    mu4 = numpy.array([6, 4.5]);
    mu5 = numpy.array([9, 0.0]);
    x1=numpy.random.multivariate_normal(mu1,Sigma,200);
    x2=numpy.random.multivariate_normal(mu2,Sigma,200);
    x3=numpy.random.multivariate_normal(mu3,Sigma,200);
    x4=numpy.random.multivariate_normal(mu4,Sigma,200);
    x5=numpy.random.multivariate_normal(mu5,Sigma,200);
    file=codecs.open("E:\Python_Project\Spectral_Clustering\Data.txt","w","utf-8");
    for i in range(200):
        file.writelines(str(x1[i,0])+","+str(x1[i,1])+"\r\n");
    for i in range(200):
        file.writelines(str(x2[i,0])+","+str(x2[i,1])+"\r\n");
    for i in range(200):
        file.writelines(str(x3[i,0])+","+str(x3[i,1])+"\r\n");
    for i in range(200):
        file.writelines(str(x4[i,0])+","+str(x4[i,1])+"\r\n");
    for i in range(200):
        file.writelines(str(x5[i,0])+","+str(x5[i,1])+"\r\n");
    file.close();

Main();