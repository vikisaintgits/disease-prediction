# Generated by Django 4.0.1 on 2022-03-06 10:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Category',
            fields=[
                ('Cat_id', models.AutoField(primary_key=True, serialize=False)),
                ('Category', models.CharField(max_length=50, verbose_name='Category')),
            ],
        ),
        migrations.CreateModel(
            name='Doctor',
            fields=[
                ('Doc_id', models.AutoField(primary_key=True, serialize=False)),
                ('Doc_name', models.CharField(max_length=25, verbose_name='Doc_name')),
                ('Doc_qualif', models.CharField(max_length=100, verbose_name='Doc_qualif')),
                ('Doc_special', models.CharField(max_length=100, verbose_name='Doc_special')),
                ('Doc_hospital_name', models.CharField(max_length=100, verbose_name='Doc_hospital_name')),
                ('Doc_contact', models.CharField(max_length=50, verbose_name='Doc_contact')),
                ('Doc_address', models.CharField(max_length=100, verbose_name='Doc_address')),
                ('Doc_loc', models.CharField(max_length=150, verbose_name='Doc_loc')),
                ('Doc_stat', models.CharField(max_length=300, verbose_name='Doc_stat')),
                ('Doc_mail', models.CharField(max_length=100, verbose_name='Doc_mail')),
                ('Doc_photo', models.CharField(max_length=100, verbose_name='Doc_photo')),
            ],
        ),
        migrations.CreateModel(
            name='login',
            fields=[
                ('log_id', models.AutoField(primary_key=True, serialize=False)),
                ('username', models.CharField(max_length=100, verbose_name='username')),
                ('password', models.CharField(max_length=100, verbose_name='password')),
                ('role', models.CharField(max_length=100, verbose_name='role')),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('User_id', models.AutoField(primary_key=True, serialize=False)),
                ('User_name', models.CharField(max_length=25, verbose_name='User_name')),
                ('User_dob', models.CharField(max_length=20, verbose_name='User_dob')),
                ('User_address', models.CharField(max_length=100, verbose_name='User_address')),
                ('User_phone', models.CharField(max_length=20, verbose_name='User_phone')),
                ('User_gender', models.CharField(max_length=10, verbose_name='User_gender')),
                ('User_photo', models.CharField(max_length=100, verbose_name='User_photo')),
                ('log_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.login')),
            ],
        ),
        migrations.CreateModel(
            name='Subcategory',
            fields=[
                ('Subcat_id', models.AutoField(primary_key=True, serialize=False)),
                ('Subcat', models.CharField(max_length=50, verbose_name='Subcat')),
                ('Cat_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.category')),
            ],
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('report_id', models.AutoField(primary_key=True, serialize=False)),
                ('report_date', models.DateField(verbose_name='report_date')),
                ('radius_mean', models.CharField(max_length=100, verbose_name='radius_mean')),
                ('texture_mean', models.CharField(max_length=100, verbose_name='texture_mean')),
                ('perimeter_mean', models.CharField(max_length=100, verbose_name='perimeter_mean')),
                ('area_mean', models.CharField(max_length=100, verbose_name='area_mean')),
                ('smoothness_mean', models.CharField(max_length=100, verbose_name='smoothness_mean')),
                ('compactness_mean', models.CharField(max_length=100, verbose_name='compactness_mean')),
                ('concavity_mean', models.CharField(max_length=100, verbose_name='concavity_mean')),
                ('concave_points_mean', models.CharField(max_length=100, verbose_name='concave_points_mean')),
                ('symmetry_mean', models.CharField(max_length=100, verbose_name='symmetry_mean')),
                ('fractal_dimension_mean', models.CharField(max_length=100, verbose_name='fractal_dimension_mean')),
                ('radius_se', models.CharField(max_length=100, verbose_name='radius_se')),
                ('texture_se', models.CharField(max_length=100, verbose_name='texture_se')),
                ('perimeter_se', models.CharField(max_length=100, verbose_name='perimeter_se')),
                ('area_se', models.CharField(max_length=100, verbose_name='area_se')),
                ('smoothness_se', models.CharField(max_length=100, verbose_name='smoothness_se')),
                ('compactness_se', models.CharField(max_length=100, verbose_name='compactness_se')),
                ('concavity_se', models.CharField(max_length=100, verbose_name='concavity_se')),
                ('concave_points_se', models.CharField(max_length=100, verbose_name='concave_points_se')),
                ('symmetry_se', models.CharField(max_length=100, verbose_name='symmetry_se')),
                ('fractal_dimension_se', models.CharField(max_length=100, verbose_name='fractal_dimension_se')),
                ('radius_worst', models.CharField(max_length=100, verbose_name='radius_worst')),
                ('texture_worst', models.CharField(max_length=100, verbose_name='texture_worst')),
                ('perimeter_worst', models.CharField(max_length=100, verbose_name='perimeter_worst')),
                ('area_worst', models.CharField(max_length=100, verbose_name='area_worst')),
                ('smoothness_worst', models.CharField(max_length=100, verbose_name='smoothness_worst')),
                ('compactness_worst', models.CharField(max_length=100, verbose_name='compactness_worst')),
                ('concavity_worst', models.CharField(max_length=100, verbose_name='concavity_worst')),
                ('concave_points_worst', models.CharField(max_length=100, verbose_name='concave_points_worst')),
                ('symmetry_worst', models.CharField(max_length=100, verbose_name='symmetry_worst')),
                ('fractal_dimension_worst', models.CharField(max_length=100, verbose_name='fractal_dimension_worst')),
                ('result', models.CharField(max_length=100, verbose_name='test_result')),
                ('report_staff', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.doctor')),
                ('report_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.user')),
            ],
        ),
        migrations.AddField(
            model_name='doctor',
            name='Subcat_id',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.subcategory'),
        ),
        migrations.AddField(
            model_name='doctor',
            name='log_id',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.login'),
        ),
        migrations.CreateModel(
            name='Consulting_time',
            fields=[
                ('Consult_id', models.AutoField(primary_key=True, serialize=False)),
                ('Consult_day', models.CharField(max_length=10, verbose_name='Consult_day')),
                ('Consult_time_from', models.CharField(max_length=20, verbose_name='Consult_time_from')),
                ('Consult_time_to', models.CharField(max_length=20, verbose_name='Consult_time_to')),
                ('Consult_nob', models.CharField(max_length=100, verbose_name='Consult_num_of_booking')),
                ('Doc_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.doctor')),
            ],
        ),
        migrations.CreateModel(
            name='Appointment',
            fields=[
                ('App_id', models.AutoField(primary_key=True, serialize=False)),
                ('App_date', models.CharField(max_length=20, verbose_name='App_date')),
                ('App_stat', models.CharField(max_length=20, verbose_name='App_stat')),
                ('Consult_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.consulting_time')),
                ('Doc_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.doctor')),
                ('User_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='hospital.user')),
            ],
        ),
    ]