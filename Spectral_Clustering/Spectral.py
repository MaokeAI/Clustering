# -*- coding:utf-8 -*-  
import numpy
import math
import matplotlib.pyplot as plt
import codecs

def Main():
    Data=numpy.zeros((200,2),float);
    m_Flie=codecs.open("E:\Python_Project\K-Means\Data.txt","r","utf-8");
    Lines=m_Flie.readlines();
    i=0;
    for li in Lines:
        st=li.encode("ascii");
        list=[st.strip().split("\r\n")]; 
        Data[i,:]=numpy.matrix(list[0][0]);
        i+=1;
    X=numpy.linspace(0.01,60,30);
    Precision=[0.0]*X.size;
    for i in range(X.size): 
        Precision[i]=Spectral_Clustering3(Data,2,X[i],10);
        print(i)
    plt.plot(X,Precision,'x',linestyle="-")
    plt.savefig("E:\Python_Project\Spectral_Clustering\M_Data1.png",dpi=266);
    #for i in range(2,21): 
    #    Precision[i]=Spectral_Clustering3(Data,2,10,i);
    #    print(i)
    #plt.plot(numpy.linspace(0,20,20),Precision[:20],'x',linestyle="-")
    #plt.savefig("E:\Python_Project\Spectral_Clustering\M_Data2.png",dpi=266);


def Detect_And_Draw(Data,Seeds,PrimData):
    Data_lenth=Data.shape;
    Seed_lenth=Seeds.shape;
    Result=numpy.zeros(Data_lenth[0],int);                  #用于记录每个样本的分类结果
    tem_Data=numpy.zeros((Seed_lenth[0],Data_lenth[1]),float);
    Similarity=numpy.matrix(numpy.zeros(Seed_lenth[0],float));          #用于存储相似度
    for i in range(Data_lenth[0]):                          #对所有样本            
        for j in range(Seed_lenth[0]): tem_Data[j,:]=Data[i,:];
        Difference=numpy.matrix(Seeds-tem_Data);
        for j in range(Seed_lenth[0]): Similarity[0,j]=Difference[j,:]*Difference[j,:].T;   
        Result[i]=Similarity.argmin();               #记录最相似的点
    #然后进行画图
    #m_Color=[(1.0,0.0,0.0),(0.0,0.0,1.0),(0.0,1.0,0.0)];
    #for i in range(Data_lenth[0]):
    #    plt.scatter(PrimData[i,0],PrimData[i,1],color=m_Color[Result[i]],alpha=0.7);
    #plt.savefig("E:\Python_Project\Spectral_Clustering\M_Data.png",dpi=266);
    #plt.show();
    return numpy.sum(Result[:100]),numpy.sum(Result[100:200])

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
        #Record[t]=Step;
        t+=1;
        #print(Step);
        if Step<0.00001: break;
    return Seeds;

#谱聚类算法主函数
def Spectral_Clustering3(Data,k,Sigma,k_near):
    lenth=Data.shape;
    W=numpy.zeros((lenth[0],lenth[0]),float);          #首先来构造相似度矩阵
    Similarity=numpy.matrix(numpy.zeros((lenth[0],lenth[0]),float));
    for i in range(lenth[0]):
        d1=Data[i,:];
        for j in range(i):
            d2=Data[j,:];
            d=numpy.matrix(d1-d2)
            Similarity[i,j]=math.exp(-float(d*d.T)/(2*Sigma));
            Similarity[j,i]=Similarity[i,j];
    ##下面是一个k近邻的过程
    for i in range(lenth[0]):
        index=Similarity[i,:].argsort();
        for j in range(lenth[0]-k_near,lenth[0]):
            W[i,index[0,j]]=Similarity[i,index[0,j]];
    W=numpy.matrix(W);      
    Ones=numpy.matrix([1.0]*lenth[0]);
    Row_sum=numpy.array(W*Ones.T).T;                  #计算行和
    D=numpy.matrix(numpy.eye(lenth[0])*Row_sum).I;    #这里直接求的D的逆矩阵
    D=numpy.power(D,0.5);
    L_sym=numpy.eye(lenth[0])-D*W*D;
    b,c=numpy.linalg.eig(L_sym);
    B=b.real;
    B=B.argsort();
    U=numpy.matrix(numpy.zeros((lenth[0],k),float));
    for i in range(k):
        U[:,i]=c.real[:,B[i]];
    Seeds=K_Means(U,k);
    n1,n2=Detect_And_Draw(U,Seeds,Data);
    return float(max(n1,n2)*2.0)/lenth[0]

Main();