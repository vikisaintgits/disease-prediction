from django.shortcuts import render
from django.shortcuts import redirect ,HttpResponse
import datetime
from datetime import datetime
from django.core.files.storage import FileSystemStorage
from .models import Report as res, heartReport as res1
from .models import login as log,Category as cats,Subcategory as subcats,Doctor as doc,User as usr,Appointment,Consulting_time as ct
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from django.http import HttpResponse
# Create your views here.
def index(request):
    return render(request,"index.html",{"msg":""})
def AdminHome(request):
    
    return render(request,"adminhome.html",{"msg":""})
def DoctorHome(request):
    
    return render(request,"doctorhome.html",{"msg":""})
def UserHome(request):
    
    return render(request,"userhome.html",{"msg":""})

def Manage_Category(request):
    ids=request.session['id']
    d1=doc.objects.filter(log_id=ids).values('Doc_id').first()
    d2=Appointment.objects.filter(Doc_id=d1['Doc_id'],App_stat='Waiting').all()
    #d3=usr.objects.filter(User_id=d2['User_id'])
    #d4=ct.objects.filter(Doc_id=d1['Doc_id'])
    return render(request,'mang_cat1.html',{'d3':d2,"msg":""})
    

def Manage_Category1(request):
    msg=""
    if request.POST:
        t1=request.POST["t1"]
        cats.objects.create(Category=t1)
        msg="inserted sucessfully"
    datax=cats.objects.all()
    return render(request,"manage_cat.html",{"msg":msg,"datax":datax})


def Manage_Subcat1(request):
    msg=""
    if request.POST:
        t1=request.POST["t1"]
        datax=cats.objects.get(Cat_id=t1)
        t2=request.POST["t2"]
        subcats.objects.create(Cat_id=datax,Subcat=t2)
        msg="inserted sucessfully"
    data1 = cats.objects.all()
    data2= subcats.objects.all()
    return render(request,"manage_subcat.html",{"msg":msg,"data":data1,"data1":data2})


def delete_cat(request):
    cats.objects.filter(Cat_id=request.GET["id"]).delete()
    response = redirect('/Manage_Category1')
    return response
def delete_subcat(request):
    subcats.objects.filter(Subcat_id=request.GET["id"]).delete()
    response = redirect('/Manage_Subcat1')
    return response
def Manage_Subcat(request):
    ids=request.session['id']
    data=doc.objects.filter(log_id=ids).values('Doc_id').first()
    data=Appointment.objects.filter(Doc_id=data['Doc_id'],App_stat='Confirmed')
    return render(request,'mang_subcat1.html',{'data':data,"msg":""})
    # msg=""
    # if request.POST:
    #     t1=request.POST["t1"]
    #     datax=cats.objects.get(Cat_id=t1)
    #     t2=request.POST["t2"]
    #     subcats.objects.create(Cat_id=datax,Subcat=t2)
    #     msg="inserted sucessfully"
    # data1 = cats.objects.all()
    # data2= subcats.objects.all()
    # return render(request,"manage_subcat.html",{"msg":msg,"data":data1,"data1":data2})
def Privacy(request):
    msg=request.GET.get("msg","")
    ids=request.session["id"]
    data=log.objects.filter(log_id=ids)
    return render(request,'privacy.html',{'data':data,"msg":msg})
def cancerpredict(request):
    datai={}
    id=request.GET["id"]
    dataj=Appointment.objects.get(App_id=id)
    msg = ""
    data1 = dataj.User_id
    data2 = doc.objects.get(log_id=request.session["id"])
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        t4 = request.POST["t4"]
        t5 = request.POST["t5"]
        t6 = request.POST["t6"]
        t7 = request.POST["t7"]
        t8 = request.POST["t8"]
        t9 = request.POST["t9"]
        t10 = request.POST["t10"]
        u1 = request.POST["u1"]
        u2 = request.POST["u2"]
        u3 = request.POST["u3"]
        u4 = request.POST["u4"]
        u5 = request.POST["u5"]
        u6 = request.POST["u6"]
        u7 = request.POST["u7"]
        u8 = request.POST["u8"]
        u9 = request.POST["u9"]
        u10 = request.POST["u10"]
        v1 = request.POST["v1"]
        v2 = request.POST["v2"]
        v3 = request.POST["v3"]
        v4 = request.POST["v4"]
        v5 = request.POST["v5"]
        v6 = request.POST["v6"]
        v7 = request.POST["v7"]
        v8 = request.POST["v8"]
        v9 = request.POST["v9"]
        v10 = request.POST["v10"]
        z1=request.POST["z1"]
        dataset = pd.read_csv('dataset/datasets.csv')
        dataset = dataset.drop("id", axis=1)
        dataset = dataset.drop("Unnamed: 32", axis=1)
        data = {'M': 1, 'B': 0}
        dataset.diagnosis = [data[i] for i in dataset.diagnosis.astype(str)]
        with open('training/innovators.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                             'radius_se', 'texture_se', 'perimeter_se','area_se', 'smoothness_se', 'compactness_se', 'concavity_se','concave points_se', 'symmetry_se', 'fractal_dimension_se',
                             'radius_worst', 'texture_worst', 'perimeter_worst','area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])
            writer.writerow([0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10])
        dataset2 = pd.read_csv('training/innovators.csv')

        train_features = dataset.iloc[:, 0:31]

        train_label = dataset.iloc[:, 0]

        test_features = dataset2.iloc[:, 0:31]

        gnb = GaussianNB()

        gnb.fit(train_features, train_label)

        prediction = gnb.predict(test_features)
        r=prediction[0];
        if r==0:
           d="Negative"
        else:
            d="Positive"

        res.objects.create(
            report_Appointment=dataj,
            radius_mean=t1,
                           texture_mean=t2,
                           perimeter_mean=t3,
                           area_mean=t4,
                           smoothness_mean=t5,
                           compactness_mean=t6,
                           concavity_mean=t7,
                           concave_points_mean=t8,
                           symmetry_mean=t9,
                           fractal_dimension_mean=t10,
                           radius_se=u1,
                           texture_se=u2,
                           perimeter_se=u3,
                           area_se=u4,
                           smoothness_se=u5,
                           compactness_se=u6,
                           concavity_se=u7,
                           concave_points_se=u8,
                            symmetry_se=u9,
                           fractal_dimension_se=u10,
                           radius_worst=v1,
                           texture_worst=v2,
                           perimeter_worst=v3,
                           area_worst=v4,
                           smoothness_worst=v5,
                           compactness_worst=v6,
                           concavity_worst=v7,
                           concave_points_worst=v8,
                           symmetry_worst=v9,
                           fractal_dimension_worst=v10,
                           report_user=data1,
                           report_staff=data2,
                           report_date=z1,
                           result=d
                           )
        msg="successfully updated and this patient report is "+d
    outc=res.objects.filter(report_Appointment=dataj).count()
    if outc==1:
        out=res.objects.get(report_Appointment=dataj)
    else :
        out=[]

    return render(request,"cancerpredict.html",{"msg":msg,"data":out,"id":id,"outc":outc})

def heartpredict(request):
    datai={}
    id=request.GET["id"]
    dataj=Appointment.objects.get(App_id=id)
    msg = ""
    data1 = dataj.User_id
    data2 = doc.objects.get(log_id=request.session["id"])
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        t4 = request.POST["t4"]
        t5 = request.POST["t5"]
        t6 = request.POST["t6"]
        t7 = request.POST["t7"]
        t8 = request.POST["t8"]
        t9 = request.POST["t9"]
        t10 = request.POST["t10"]
        t11 = request.POST["t11"]
        
        z1=request.POST["z1"]
        dataset = pd.read_csv('dataset/heart.csv')
        #dataset = dataset.drop("id", axis=1)
        #dataset = dataset.drop("Unnamed: 32", axis=1)
        data = {'M': 1, 'F': 0}
        dataset.Sex = [data[i] for i in dataset.Sex.astype(str)]
        datas1 = {'ASY': 1, 'NAP': 2,'ATA': 3, 'TA': 4}
        dataset.ChestPainType = [datas1[i] for i in dataset.ChestPainType.astype(str)]
        datas2 = {'Normal': 1, 'ST': 2,'LVH': 3}
        dataset.RestingECG = [datas2[i] for i in dataset.RestingECG.astype(str)]     
        data3 = {'Y': 1, 'N': 0}
        dataset.ExerciseAngina = [data3[i] for i in dataset.ExerciseAngina.astype(str)]  
        data4 = {'Up': 1, 'Flat': 2,"Down":3}
        dataset.ST_Slope = [data4[i] for i in dataset.ST_Slope.astype(str)]   
        
        with open('training/heartinv.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Age', 'Sex', 'ChestPainType', 'RestingBP','Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
            'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease' ])
            writer.writerow([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,0])
        dataset2 = pd.read_csv('training/heartinv.csv')
        dataset2.Sex = [data[i] for i in dataset2.Sex.astype(str)]
        dataset2.ChestPainType = [datas1[i] for i in dataset2.ChestPainType.astype(str)]
        dataset2.RestingECG = [datas2[i] for i in dataset2.RestingECG.astype(str)]  
        dataset2.ExerciseAngina = [data3[i] for i in dataset2.ExerciseAngina.astype(str)]
        dataset2.ST_Slope = [data4[i] for i in dataset2.ST_Slope.astype(str)]  

        train_features = dataset.iloc[:, 0:10]

        train_label = dataset.iloc[:, 11]

        test_features = dataset2.iloc[:, 0:10]

        gnb = GaussianNB()

        gnb.fit(train_features, train_label)

        prediction = gnb.predict(test_features)
        r=prediction[0];
        if r==0:
           d="Negative"
        else:
            d="Positive"

        res1.objects.create(
            report_Appointment=dataj,
             age=t1,Gender=t2,Chest_pain=t3,pressure=t4,cholesterol=t5,fasting=t6,electrocardiogram=t7,heart_rate=t8,ExerciseAngina=t9,Oldpeak=t10,slope=t11,
                        
                           report_user=data1,
                           report_staff=data2,
                           report_date=z1,
                           result=d
                           )
        msg="successfully updated and this patient report is "+d
    outc=res1.objects.filter(report_Appointment=dataj).count()
    if outc==1:
        out=res1.objects.get(report_Appointment=dataj)
    else :
        out=[]

    return render(request,"heartpredict.html",{"msg":msg,"data":out,"id":id,"outc":outc})

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
def predict(request):
    prediction=""
    if request.POST:
        images_dir = 'dataset/train/'
        datagen = ImageDataGenerator (
            rescale = 1./255, 
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
            )
        train_generator  =    datagen.flow_from_directory(
                             images_dir,
                             seed=42,
                             target_size = (200,200),
                             batch_size =32 ,               
                             class_mode = 'binary',
                             subset = 'training'
                            )

        Validation_generator = datagen.flow_from_directory(
                             images_dir ,
                             seed=42, 
                             target_size = (200,200),
                             batch_size = 32 ,               
                             class_mode = 'binary',
                             subset = 'validation'
                            )

        model=Sequential()

        model.add(Conv2D(32,(3,3),input_shape=(200,200,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #The first CNN layer followed by Relu and MaxPooling layers

        model.add(Conv2D(64,(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #The second convolution layer followed by Relu and MaxPooling layers

        model.add(Flatten())
        model.add(Dropout(0.5))
        #Flatten layer to stack the output convolutions from second convolution layer
        model.add(Dense(128,activation='relu'))
        #Dense layer of 128 neurons
        model.add(Dense(2,activation='softmax'))
        #The Final layer with two outputs for two categories

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        H = model.fit_generator(
                    train_generator ,
                    epochs = 10,
                    validation_data = Validation_generator)
        t8 = request.FILES["t1"]
        fs = FileSystemStorage()
        fnm=fs.save(t8.name, t8)
        print(fnm)
        test_image = image.load_img('media/'+fnm, target_size = (200,200,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        answer = np.argmax(result,axis=1) 
        train_generator.class_indices
        if result[0][0]==1:
            prediction = 'Confirm'
        else :
            prediction = 'normal'
        print(prediction)
    return render(request, "pnemniapredict.html",{"prediction":prediction})

def privacychange(request):
    msg=""
    if request.POST:
        t0=request.POST["t0"]
        t2=request.POST["t2"]
        t1=request.POST["t1"]
        t3=request.POST["t3"]
        data=log.objects.get(log_id=int(t0))
        if data.password == t1:
            log.objects.filter(log_id=int(t0)).update(password=t3)
            msg="updated successfuly"
        else :
            msg="invalid old password"
        response = redirect('/Privacy'+"?msg="+msg)
        return response

def Privacydoctor(request):
    msg=request.GET.get("msg","")
    ids=request.session["id"]
    data=log.objects.filter(log_id=ids)
    return render(request,'Privacydoctor.html',{'data':data,"msg":msg})

def privacychangedoc(request):
    msg=""
    if request.POST:
        t0=request.POST["t0"]
        t2=request.POST["t2"]
        t1=request.POST["t1"]
        t3=request.POST["t3"]
        data=log.objects.get(log_id=int(t0))
        if data.password == t1:
            log.objects.filter(log_id=int(t0)).update(password=t3)
            msg="updated successfuly"
        else :
            msg="invalid old password"
        response = redirect('/Privacydoctor'+"?msg="+msg)
        return response  

def Privacyuser(request):
    msg=request.GET.get("msg","")
    ids=request.session["id"]
    data=log.objects.filter(log_id=ids)
    return render(request,'Privacyuser.html',{'data':data,"msg":msg})     

def privacychangeuser(request):
    msg=""
    if request.POST:
        t0=request.POST["t0"]
        t2=request.POST["t2"]
        t1=request.POST["t1"]
        t3=request.POST["t3"]
        data=log.objects.get(log_id=int(t0))
        if data.password == t1:
            log.objects.filter(log_id=int(t0)).update(password=t3)
            msg="updated successfuly"
        else :
            msg="invalid old password"
        response = redirect('/Privacyuser'+"?msg="+msg)
        return response  
        

def Logout(request):
    try:
        del request.session['id']
        del request.session['role']
        del request.session['username']
        response = redirect("/index")
        return response
    except:
        response = redirect("/index")
        return response

def Doc_reg(request):
    msg=""
    dataz=subcats.objects.all()
    dataw=cats.objects.all()
    
    if request.POST:
        t1=request.POST["t1"]
        t2=request.POST["t2"]
        t3=request.POST["t3"]
        log.objects.create(username=t2, password=t3, role="Doctor")
        data=log.objects.last()
        t4=request.POST["t4"]
        t5=request.POST["t5"]
        t6=request.POST["t6"]
        t7=request.POST["t7"]
        t8=request.POST["t8"]
        t9=request.POST["t9"]
        t10=request.POST["t10"]
        t11=request.FILES["t11"]
        t12=request.POST["t12"]
        t13=request.POST["t13"]
        fs = FileSystemStorage()
        fs.save(t11.name, t11)
        datax=subcats.objects.get(Subcat_id=t12)
        
        doc.objects.create(Doc_name=t1,
                           Doc_qualif=t4,
                           Doc_special=t5,
                           Doc_hospital_name=t6,
                           Doc_contact=t7,
                           Doc_address=t8,
                           Doc_loc=t9,
                           Doc_stat="Waiting",
                           Doc_mail=t10,
                           Doc_photo=t11,                           
                           log_id=data,
                           Subcat_id=datax)

        msg="Registered successfuly"
    #data1=doc.objects.all()
    return render(request,"Doc_reg.html",{"msg":msg,'dataz':dataz,'dataw':dataw}) 
def getsub(request):
    datac=cats.objects.get(Cat_id=request.GET["id"])
    dataz=subcats.objects.filter(Cat_id=datac)
    msg="<option value=''>Choose SubCategory</option>"
    for y in dataz:
        msg+="<option value='"+str(y.Subcat_id)+"'>"+y.Subcat+"</option>"

    return HttpResponse(msg)
def User_reg(request):
    msg=""
    if request.POST:
        t1=request.POST["t1"]
        t2=request.POST["t2"]
        t3=request.POST["t3"]
        
        t4=request.POST["t4"]
        t5=request.POST["t5"]
        t6=request.FILES["t6"]
        t7=request.POST["t7"]
        t8=request.POST["t8"]
        log.objects.create(username=t7, password=t8, role="User")
        fs = FileSystemStorage()
        fs.save(t6.name, t6)
        data=log.objects.last()
        usr.objects.create(User_name=t1,
                           User_dob=t2,
                           User_address=t3,
                           User_phone=t4,
                           User_gender=t5,
                           User_photo=t6,                                                     
                           log_id=data)
        msg="Registered successfuly"
    #data1=doc.objects.all()
    return render(request,"User_reg.html",{"msg":msg}) 

def login(request):
    if request.POST:
        user=request.POST["t1"]
        password=request.POST["t2"]
        print(user,password)
        try:
            data=log.objects.get(username=user,password=password)
            print(data,data.role)
            if(data.role=="Admin"):
                request.session['username'] = data.username
                request.session['role'] = data.role
                request.session['id'] = data.log_id
                response = redirect('/AdminHome')
                return response
            elif (data.role=="Doctor"):
                print("welcome doctor")
                userdt = doc.objects.get(log_id=data.log_id)
                print('userdt',userdt)
                if userdt.Doc_stat == "Approved":
                      request.session['username'] = data.username
                      request.session['role'] = data.role
                      request.session['id'] = data.log_id
                      response = redirect('/DoctorHome')
                      return response
                else: 
                        response = redirect('/index')
                        return response
            elif (data.role == "User"):
                print('welcome user')
                userdt= usr.objects.get(log_id=data.log_id)  
                print(userdt)            
                request.session['username'] = data.username
                request.session['role'] = data.role
                request.session['id'] = data.log_id
                response = redirect('/UserHome')
                return response
               
               
            
        except:
            response = redirect('/index')
            return response
    else:
        response = redirect('/index')
        return response

def adminlistdoctor(request):
    dic=doc.objects.filter(Doc_stat='Approved')
    return render(request,'doctorlist.html',{'dic':dic})

def doctorupdate(request,dataid):
    data=doc.objects.filter(Doc_id=dataid)
    return render(request,"editdoctor.html",{'data':data})

def adminuserlist(request):
    data=usr.objects.all()
    return render(request,'adminuserlist.html',{'data':data})

def adminuserupdate(request,dataid):
    data=usr.objects.filter(User_id=dataid)
    return render(request,'adminuseredit.html',{'data':data})

def adminusreditsubmit(request):
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        print(t1)
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        print(t2,t3)
        t4 = request.POST["t4"]
        t5 = request.POST["t5"]
        t6 = request.POST["t6"]
        
        if request.FILES:
             pic = request.FILES['imge']
             fs=FileSystemStorage()
             save_path="media/%s"%pic.name
             filename=fs.save(save_path,pic)
             usr.objects.filter(User_id=int(t6)).update(User_name=t1,User_address=t3,User_phone=t4,User_gender=t5,User_dob=t2,User_photo=filename)
             msg="updated successfuly"
             response = redirect('/adminuserlist')
             return response
        
             
        else:
             usr.objects.filter(User_id=int(t6)).update(User_name=t1,User_address=t3,User_phone=t4,User_gender=t5,User_dob=t2)
             msg="updated successfuly"
             response = redirect('/adminuserlist')
             return response


def adminuserdelete(request,dataid):
    datax=usr.objects.filter(User_id=dataid).values('log_id').first()
    log.objects.filter(log_id=datax['log_id']).delete()
    usr.objects.filter(User_id=dataid ).delete()
    response = redirect('/adminuserlist')
    return response

def admindocupdate(request,dataid):
    data=doc.objects.filter(Doc_id=dataid)
    return render(request,'admindocedit.html',{'data':data})

def admindocdelete(request,dataid):
    datax=doc.objects.filter(Doc_id=dataid).values('log_id').first()
    log.objects.filter(log_id=datax['log_id']).delete()
    doc.objects.filter(Doc_id=dataid ).delete()
    response = redirect('/adminlistdoctor')
    return response

def admindoceditsubmit(request):
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        print(t1)
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        t4 = request.POST["t4"]
        t5 = request.POST["t5"]
        t6 = request.POST["t6"]
        t7 = request.POST["t7"]
        t8 = request.POST["t8"]
        t9 = request.POST["t9"]
        t10 = request.POST["t10"]
        
        if request.FILES:
             pic = request.FILES['imge']
             fs=FileSystemStorage()
             save_path="media/%s"%pic.name
             filename=fs.save(save_path,pic)
             doc.objects.filter(Doc_id=int(t8)).update(Doc_name=t1,Doc_address=t3,Doc_contact=t4,Doc_hospital_name=t5,Doc_qualif=t2,Doc_special=t6,Doc_loc=t7,Doc_stat=t9,Doc_mail=t10,Doc_photo=filename)
             msg="updated successfuly"
             response = redirect('/adminlistdoctor')
             return response
        
             
        else:
             doc.objects.filter(Doc_id=int(t8)).update(Doc_name=t1,Doc_address=t3,Doc_contact=t4,Doc_hospital_name=t5,Doc_qualif=t2,Doc_special=t6,Doc_loc=t7,Doc_stat=t9,Doc_mail=t10)
             msg="updated successfuly"
             response = redirect('/adminlistdoctor')
             return response

def adminlistdocwait(request):
    dic=doc.objects.filter(Doc_stat='Waiting')
    return render(request,'doclistwait.html',{'dic':dic})

def admindocwaitaccept(request,dataid):
    doc.objects.filter(Doc_id=dataid).update(Doc_stat='Approved')
    msg="updated successfuly"
    response = redirect('/adminlistdocwait')
    return response

def admindocwaitreject(request,dataid):
    datax=doc.objects.filter(Doc_id=dataid ).values('log_id').first()
    log.objects.filter(log_id=datax['log_id']).delete()
    doc.objects.filter(Doc_id=dataid ).delete()
    response = redirect('/adminlistdocwait')
    return response

def doctorconsult(request):
    ids=request.session['id']
    datax=doc.objects.filter(log_id=ids).values('Doc_id').first()
    data=ct.objects.filter(Doc_id=datax['Doc_id'])
    return render(request,'doctorconsult.html',{'data':data})

def doctconsultsubmit(request):
    ids=request.session['id']
    print(ids)
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        print(t1)
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        t4 = request.POST["t4"]
        data=doc.objects.get(log_id=ids)
        ct.objects.create(Consult_day=t1,Consult_time_from=t2,Consult_time_to=t3,Consult_nob=t4,Doc_id=data)
        response = redirect('/doctorconsult')
        return response
        
def doc_consultdel(request,dataid):
    ct.objects.filter(Consult_id=dataid).delete()
    response = redirect('/doctorconsult')
    return response


def usersearch(request):
    data=cats.objects.all()
    data1=subcats.objects.all()
    data2=list(set(doc.objects.values_list('Doc_loc', flat=True)))
    return render(request,'usersearch.html',{'data':data,'data1':data1,'data2':data2,"msg":""})

def userbooking(request):
    msg=""
    data=cats.objects.all()
    #data2=doc.objects.all()
    data2=list(set(doc.objects.values_list('Doc_loc', flat=True)))
    if request.POST:
        t1 = request.POST["t1"]
        print(t1)
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        print(t2,t3)
        #data=cats.objects.get(Cat_id=t1)
        data1=subcats.objects.get(Subcat_id=t2)
            #print(data1['Subcat_id'],'subcategory filtered')
            #data2=doc.objects.filter(Subcat_id=t2)
        #if doc.objects.filter(Subcat_id=data1,Doc_loc=t3).exists():
           # print("dotor with specified location exists")
            #objt=doc.objects.filter(Subcat_id=data1).values('Doc_id').first()
        dic=doc.objects.filter(Subcat_id=data1,Doc_loc=t3).all()
        
        return render(request,'usersearch.html',{'dic':dic,"msg":"",'data':data,'data2':data2})
        #else:
            #return render(request,'usersearch.html',{"msg":"",'data':data,'data2':data2})
        

        # if cats.objects.filter(Category=t1).exists():
            
        #     print("doctor")
        #     data = subcats.objects.filter(Subcat=t2).values('Subcat_id').first()
        #     obj=doc.objects.filter(Subcat_id=data['Subcat_id'])
        #     print(obj)
        #     objt=doc.objects.filter(Subcat_id=data['Subcat_id']).values('Doc_id').first()
        #     dic=ct.objects.filter(Doc_id=objt['Doc_id'])
        #     return render(request,'usersearch.html',{'dic':dic})
def goback(request):
    response = redirect('/UserHome')
    return response


def userfileupload(request,dataid):
    msg=""
    if request.POST:
        if request.FILES:
            pic = request.FILES['img']
            fs=FileSystemStorage()
            save_path="media/%s"%pic.name
            filename=fs.save(save_path,pic)  
            ids=request.session['id']
            d1=usr.objects.filter(log_id=ids).values('User_id').first()
            d2=Appointment.objects.filter(User_id=d1['User_id']).values('App_id').first()
            Appointment.objects.filter(App_id=dataid).update(App_files=filename)
            msg="Uploaded Successfully"
            response = redirect('/UserHome')
            return response


            

def myappointments(request,dataid,datacon):
    datax=usr.objects.filter(log_id=dataid)
    datay=ct.objects.filter(Consult_id=datacon).values('Doc_id').first()
    
    datadoc=datay['Doc_id']
    print("Userid",dataid)
    print("Consultid",datacon)
    dt=ct.objects.get(Consult_id=datacon)
    doctt=datay['Doc_id']
    print(doctt)
    doct=doc.objects.get(Doc_id=doctt)
     
    dataconsult=ct.objects.filter(Consult_id=datacon)
    datalog=usr.objects.get(log_id=dataid)
    docdic=ct.objects.filter(Doc_id=doctt)
    docdtt=doc.objects.filter(Doc_id=doctt)
   
    
    
    Appointment.objects.create(App_stat="Waiting",App_date=datetime.today().strftime('%m/%d/%Y'),Consult_id=dt,User_id=datalog,Doc_id=doct,conf_date="")
    appdic=Appointment.objects.filter(Doc_id=doctt)
    
    return render(request,'myappointments.html',{'appdic':appdic,"msg":""})


def docbookingapprove(request,uid):
    Appointment.objects.filter(App_id=uid).update(App_stat='Confirmed')
    data=Appointment.objects.filter(App_id=uid).values('Consult_id').first()
    datax=ct.objects.filter(Consult_id=data['Consult_id']).values('Consult_nob').first()
    booking=int(datax['Consult_nob'])
    totalbooking=booking-1
    ct.objects.filter(Consult_id=data['Consult_id']).update(Consult_nob=totalbooking)
    
    response = redirect('/Manage_Category')
    return response


def docbookingreject(request,uid):
    Appointment.objects.filter(App_id=uid).delete()
    response = redirect('/Manage_Category')
    return response

def myprofile(request):
    ids=request.session['id']
    data=doc.objects.filter(log_id=ids)
    return render(request,'myprofile.html',{'data':data,"msg":""})


def confirmeduserapt(request):
    ids=request.session['id']
    print(ids)
    data=usr.objects.filter(log_id=ids).values('User_id').first()
    dic=Appointment.objects.filter(User_id=data['User_id'])
    return render(request,'usermyappointments.html',{'dic':dic,"msg":""})
       

def docprofile_edit(request):
    ids=request.session['id']
    print(ids)
    data=doc.objects.filter(log_id=ids)
    return render(request,'myprofile_edit.html',{'data':data,"msg":""})
    

def updateprofiledoc(request):
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        print(t1)
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        t4 = request.POST["t4"]
        t5 = request.POST["t5"]
        t6 = request.POST["t6"]
        t7 = request.POST["t7"]
        t8 = request.POST["t8"]
        #t9 = request.FILES["t9"]
        t10 = request.POST["t10"]
        
        if request.FILES:
            # pic = request.FILES['t9']
            # fs=FileSystemStorage()
             #save_path="media/%s"%pic.name
             #filename=fs.save(save_path,pic)
             doc.objects.filter(Doc_id=int(t10)).update(Doc_name=t1,Doc_address=t6,Doc_contact=t5,Doc_hospital_name=t4,Doc_qualif=t2,Doc_special=t3,Doc_loc=t7,Doc_mail=t8)
             #,Doc_photo=filename
             msg="updated successfuly"
             response = redirect('/DoctorHome')
             return response
        
             
        else:
             doc.objects.filter(Doc_id=int(t10)).update(Doc_name=t1,Doc_address=t6,Doc_contact=t5,Doc_hospital_name=t4,Doc_qualif=t2,Doc_special=t3,Doc_loc=t7,Doc_mail=t8)
             msg="updated successfuly"
             response = redirect('/DoctorHome')
             return response


def usermyprofile(request):
    ids=request.session['id']
    data=usr.objects.filter(log_id=ids)
    return render(request,'usermyprofile.html',{'data':data,"msg":""})

def userprofile_edit(request):
    ids=request.session['id']
    print(ids)
    data=usr.objects.filter(log_id=ids)
    return render(request,'userprofile_edit.html',{'data':data,"msg":""})


def updateprofileusr(request):
    msg=""
    if request.POST:
        t1 = request.POST["t1"]
        print(t1)
        t2 = request.POST["t2"]
        t3 = request.POST["t3"]
        t4 = request.POST["t4"]
        t5 = request.POST["t5"]
        t6 = request.POST["t6"]
        #t7 = request.FILES["t7"]
       
        
        if request.FILES:
             pic = request.FILES['t7']
             #fs=FileSystemStorage()
             #save_path="media/%s"%pic.name
             #filename=fs.save(save_path,pic)
             usr.objects.filter(User_id=int(t4)).update(User_name=t1,User_address=t6,User_phone=t5,User_dob=t2,User_gender=t3)
             #,User_photo=filename
             msg="updated successfuly"
             response = redirect('/UserHome')
             return response
        
             
        else:
             usr.objects.filter(User_id=int(t4)).update(User_name=t1,User_address=t6,User_phone=t5,User_dob=t2,User_gender=t3)
             msg="updated successfuly"
             response = redirect('/UserHome')
             return response

