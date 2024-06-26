from django.urls import path
from . import views

urlpatterns = [
    path('', views.index,name="index"),
    path('index', views.index,name="index"),
    path('UserHome', views.UserHome,name="UserHome"),
    path('DoctorHome', views.DoctorHome,name="DoctorHome"),
    path('AdminHome', views.AdminHome,name="AdminHome"),
    path('Doc_reg', views.Doc_reg,name="Doc_reg"),
    path('User_reg', views.User_reg,name="User_reg"),
    path('Manage_Category', views.Manage_Category,name="Manage_Category"),
    path('Manage_Category1', views.Manage_Category1,name="Manage_Category1"),
    path('delete_cat', views.delete_cat,name="delete_cat"),
    path('Manage_Subcat', views.Manage_Subcat,name="Manage_Subcat"),
    path('Manage_Subcat1', views.Manage_Subcat1,name="Manage_Subcat1"),
    path('delete_subcat', views.delete_subcat,name="delete_subcat"),
    path('Privacy', views.Privacy,name="Privacy"),
    path('privacychange', views.privacychange,name="privacychange"),
    path('Logout', views.Logout,name="Logout"),
    path('login',views.login,name='login'),
    path('adminlistdoctor',views.adminlistdoctor,name='adminlistdoctor'),
    path('doctorupdate/<int:dataid>',views.doctorupdate,name='doctorupdate'),
    path('adminuserlist',views.adminuserlist,name='adminuserlist'),
    path('adminuserupdate/<int:dataid>',views.adminuserupdate,name='adminuserupdate'),
    path('adminusreditsubmit',views.adminusreditsubmit,name='adminusreditsubmit'),
    path('adminuserdelete/<int:dataid>',views.adminuserdelete,name='adminuserdelete'),
    path('admindocupdate/<int:dataid>',views.admindocupdate,name='admindocupdate'),
    path('admindocdelete/<int:dataid>',views.admindocdelete,name='admindocdelete'),
    path('admindoceditsubmit',views.admindoceditsubmit,name='admindoceditsubmit'),
    path('adminlistdocwait',views.adminlistdocwait,name='adminlistdocwait'),
    path('admindocwaitaccept/<int:dataid>',views.admindocwaitaccept,name='admindocwaitaccept'),
    path('admindocwaitreject/<int:dataid>',views.admindocwaitreject,name='admindocwaitreject'),
    path('doctorconsult',views.doctorconsult,name='doctorconsult'),
    path('doctconsultsubmit',views.doctconsultsubmit,name='doctconsultsubmit'),
    path('doc_consultdel/<int:dataid>',views.doc_consultdel,name='doc_consultdel'),
    path('usersearch',views.usersearch,name='usersearch'),
    path('userbooking',views.userbooking,name='userbooking'),
    path('myappointments/<int:dataid>/<int:datacon>',views.myappointments,name='myappointments'),
    path('docbookingapprove/<int:uid>',views.docbookingapprove,name='docbookingapprove'),
    path('docbookingreject/<int:uid>',views.docbookingreject,name='docbookingreject'),
    path('myprofile',views.myprofile,name='myprofile'),
    path('Privacydoctor',views.Privacydoctor,name='Privacydoctor'),
    path('privacychangedoc', views.privacychangedoc,name="privacychangedoc"),
    path('Privacyuser',views.Privacyuser,name='Privacyuser'),
    path('privacychangeuser', views.privacychangeuser,name="privacychangeuser"),
    path('confirmeduserapt',views.confirmeduserapt,name='confirmeduserapt'),
    path('docprofile_edit',views.docprofile_edit,name='docprofile_edit'),
    path('updateprofiledoc',views.updateprofiledoc,name='updateprofiledoc'),
    path('usermyprofile',views.usermyprofile,name='usermyprofile'),
    path('userprofile_edit',views.userprofile_edit,name='userprofile_edit'),
    path('updateprofileusr',views.updateprofileusr,name='updateprofileusr'),
    path('goback',views.goback,name='goback'),
    path('userfileupload/<int:dataid>',views.userfileupload,name='userfileupload'),
    path('getsub',views.getsub,name='getsub'),

path('cancerpredict',views.cancerpredict,name='cancerpredict'),
path('heartpredict',views.heartpredict,name='heartpredict'),
path('predict',views.predict,name='predict'),

    

]
