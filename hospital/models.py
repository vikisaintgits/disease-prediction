from django.db import models

# Create your models here.

class login(models.Model):
    log_id = models.AutoField(primary_key=True)
    username = models.CharField("username", max_length=100)
    password = models.CharField("password", max_length=100)
    role = models.CharField("role", max_length=100)
#log_id,username,password,role

class Category(models.Model):
    Cat_id = models.AutoField(primary_key=True)
    Category=models.CharField("Category", max_length=50)
#Cat_id,Category    

class Subcategory(models.Model):
    Subcat_id=models.AutoField(primary_key=True)
    Cat_id=models.ForeignKey(Category, on_delete=models.CASCADE, null=True)
    Subcat=models.CharField("Subcat", max_length=50)
# Subcat_id,Cat_id,Subcat

class Doctor(models.Model):
    Doc_id=models.AutoField(primary_key=True)
    Doc_name=models.CharField("Doc_name", max_length=25)
    Doc_qualif=models.CharField("Doc_qualif", max_length=100)
    Doc_special=models.CharField("Doc_special", max_length=100)
    Doc_hospital_name=models.CharField("Doc_hospital_name", max_length=100)
    Doc_contact=models.CharField("Doc_contact", max_length=50)
    Doc_address=models.CharField("Doc_address", max_length=100)
    Doc_loc=models.CharField("Doc_loc", max_length=150)
    Doc_stat=models.CharField("Doc_stat", max_length=300)
    Doc_mail=models.CharField("Doc_mail", max_length=100)
    Doc_photo=models.CharField("Doc_photo", max_length=100)
    Subcat_id=models.ForeignKey(Subcategory, on_delete=models.CASCADE, null=True)
    log_id=models.ForeignKey(login, on_delete=models.CASCADE, null=True)
    @property
    def consult_t(self):
        data=Consulting_time.objects.filter(Doc_id=self).all()
        return data
    @property
    def consult_count(self):
        data=Consulting_time.objects.filter(Doc_id=self).count()
        return data
#Doc_id,Doc_name,Doc_qualif,Doc_special,Doc_hospital_name,Doc_contact,Doc_address,Doc_locDoc_stat,Doc_mail,Doc_photo,Subcat_id,log_id

class Consulting_time(models.Model):
    Consult_id=models.AutoField(primary_key=True)
    Doc_id=models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True)
    Consult_day=models.CharField("Consult_day", max_length=10)
    Consult_time_from=models.CharField("Consult_time_from", max_length=20)
    Consult_time_to=models.CharField("Consult_time_to", max_length=20)
    Consult_nob=models.CharField("Consult_num_of_booking", max_length=100)
#  Consult_id,Doctor_id,Consult_day,Consult_time_from,Consult_time_to,Consult_nob

class User(models.Model):
    User_id=models.AutoField(primary_key=True)
    User_name=models.CharField("User_name", max_length=25)
    User_dob=models.CharField("User_dob", max_length=20)
    User_address=models.CharField("User_address", max_length=100)
    User_phone=models.CharField("User_phone", max_length=20)
    User_gender=models.CharField("User_gender", max_length=10)
    User_photo=models.CharField("User_photo", max_length=100)
    log_id=models.ForeignKey(login, on_delete=models.CASCADE, null=True)
    User_files=models.FileField("User_files", max_length=1000,upload_to='userfiles/')
#User_id,User_name,User_dob,User_address,User_phoneUser_gender,User_photo,log_id

class Appointment(models.Model):
    App_id=models.AutoField(primary_key=True)
    Doc_id=models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True)
    User_id=models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    Consult_id=models.ForeignKey(Consulting_time, on_delete=models.CASCADE, null=True)
    App_date=models.CharField("App_date", max_length=20)
    App_stat=models.CharField("App_stat", max_length=20)
    App_files=models.FileField("app_files", max_length=1000,upload_to='appfiles/')
    conf_date=models.CharField("conf_date", max_length=100)
#App_id,Doc_id,User_id,Consult_id,App_date,App_stat,conf_date

class Report(models.Model):
      report_id = models.AutoField(primary_key=True)
      report_Appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, null=True)
      report_user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
      report_staff = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True)
      report_date = models.CharField("report_date", max_length=100)
      radius_mean=models.CharField("radius_mean", max_length=100)
      texture_mean=models.CharField("texture_mean", max_length=100)
      perimeter_mean=models.CharField("perimeter_mean", max_length=100)
      area_mean=models.CharField("area_mean", max_length=100)
      smoothness_mean=models.CharField("smoothness_mean", max_length=100)
      compactness_mean=models.CharField("compactness_mean", max_length=100)
      concavity_mean=models.CharField("concavity_mean", max_length=100)
      concave_points_mean=models.CharField("concave_points_mean", max_length=100)
      symmetry_mean=models.CharField("symmetry_mean", max_length=100)
      fractal_dimension_mean=models.CharField("fractal_dimension_mean", max_length=100)
      radius_se=models.CharField("radius_se", max_length=100)
      texture_se=models.CharField("texture_se", max_length=100)
      perimeter_se=models.CharField("perimeter_se", max_length=100)
      area_se=models.CharField("area_se", max_length=100)
      smoothness_se=models.CharField("smoothness_se", max_length=100)
      compactness_se=models.CharField("compactness_se", max_length=100)
      concavity_se=models.CharField("concavity_se", max_length=100)
      concave_points_se=models.CharField("concave_points_se", max_length=100)
      symmetry_se=models.CharField("symmetry_se", max_length=100)
      fractal_dimension_se=models.CharField("fractal_dimension_se", max_length=100)
      radius_worst=models.CharField("radius_worst", max_length=100)
      texture_worst=models.CharField("texture_worst", max_length=100)
      perimeter_worst=models.CharField("perimeter_worst", max_length=100)
      area_worst=models.CharField("area_worst", max_length=100)
      smoothness_worst=models.CharField("smoothness_worst", max_length=100)
      compactness_worst=models.CharField("compactness_worst", max_length=100)
      concavity_worst=models.CharField("concavity_worst", max_length=100)
      concave_points_worst=models.CharField("concave_points_worst", max_length=100)
      symmetry_worst=models.CharField("symmetry_worst", max_length=100)
      fractal_dimension_worst=models.CharField("fractal_dimension_worst", max_length=100)
      result=models.CharField("test_result", max_length=100)

class heartReport(models.Model):
      report_id = models.AutoField(primary_key=True)
      report_Appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, null=True)
      report_user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
      report_staff = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True)
      report_date = models.CharField("report_date", max_length=100)
      age=models.CharField("age", max_length=100)
      Gender=models.CharField("Gender", max_length=100)
      Chest_pain=models.CharField("Chest_pain", max_length=100)
      pressure=models.CharField("pressure", max_length=100)
      cholesterol=models.CharField("cholesterol", max_length=100)
      fasting=models.CharField("fasting", max_length=100)
      electrocardiogram=models.CharField("electrocardiogram", max_length=100)
      heart_rate=models.CharField("heart_rate", max_length=100)
      ExerciseAngina=models.CharField("ExerciseAngina", max_length=100)
      Oldpeak=models.CharField("Oldpeak", max_length=100)
      slope=models.CharField("slope", max_length=100)
      result=models.CharField("test_result", max_length=100)
#age,Gender,Chest_pain,pressure,cholesterol,fasting,electrocardiogram,heart_rate,ExerciseAngina,Oldpeak,slope,result,report_date,report_staff,report_user,report_Appointment,
