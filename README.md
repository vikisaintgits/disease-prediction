# Disease Prediction System

The Disease Prediction System is a Django-based application that allows users to book appointments with doctors and provides doctors with a dashboard to check for Heart Disease, Breast Cancer, and Pneumonia predictions.

## Features

### Admin Module
- Manage users (patients and doctors).
- View and manage appointments.
- Access to system settings and logs.

### Doctor Module
- Dashboard to view and manage appointments.
- Access to prediction tools for:
  - **Heart Disease**
  - **Breast Cancer**
  - **Pneumonia**
- View patient history and health records.

### User Module
- Register and log in to the system.
- Book appointments with doctors.
- View appointment history and status.

## Installation

To get started with the Disease Prediction System, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/vikisaintgits/disease-prediction.git
    ```

2. **Navigate to the project directory:**
    ```sh
    cd disease-prediction
    ```

3. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

4. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

5. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

6. **Apply migrations:**
    ```sh
    python manage.py migrate
    ```

7. **Create a superuser:**
    ```sh
    python manage.py createsuperuser
    ```

8. **Run the development server:**
    ```sh
    python manage.py runserver
    ```

9. **Access the application:**
    - Open your web browser and navigate to `http://127.0.0.1:8000`.

## Usage

### Admin Module

1. **Log in as Admin:**
    - Use the credentials created during the superuser setup to log in to the admin panel at `http://127.0.0.1:8000/admin`.

2. **Manage Users and Appointments:**
    - Add, edit, or delete users and appointments from the admin panel.

### Doctor Module

1. **Log in as Doctor:**
    - Use your doctor credentials to log in to the dashboard.

2. **View Dashboard:**
    - Access the prediction tools for Heart Disease, Breast Cancer, and Pneumonia.
    - Manage and view your appointments.

### User Module

1. **Register and Log in:**
    - Create a new account or log in with your existing user credentials.

2. **Book Appointments:**
    - Select a doctor and book an appointment.
    - View your appointment history and status.
