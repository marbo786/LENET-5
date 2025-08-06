LeNet-5 Handwritten Digit Recognizer
A simple Streamlit web app that uses the LeNet-5 CNN architecture to recognize handwritten digits drawn by the user in real time.

Demo:
Try it live (after deployment): [https://lenet-5-zedlezgpachnhzcn4po5ld.streamlit.app/]

Features:

Interactive canvas to draw digits (0–9)

Real-time prediction using a trained LeNet-5 PyTorch model

Clean and responsive Streamlit interface

Easy to modify and retrain with your own dataset

Tech Stack:

Frontend: Streamlit, streamlit-drawable-canvas

Backend: PyTorch, NumPy, PIL

Model: LeNet-5 architecture trained on MNIST dataset

How to Run Locally:

Clone the repository:
git clone https://github.com/your-username/lenet5-digit-recognizer.git
cd lenet5-digit-recognizer

Install dependencies (use a virtual environment if needed):
pip install -r requirements.txt

Run the app:
streamlit run app.py

Project Structure:

app.py → Streamlit app interface

model.py → LeNet5 model definition

lenet5_mnist.pth → Trained model weights

requirements.txt → Python dependencies

README.md → Project overview

Credits:

Model inspired by LeCun et al.'s LeNet-5

Dataset: MNIST Handwritten Digits

Built with love using Streamlit and PyTorch

Contact:

Developer: [MOHSIN SAEED]
Email: marboo786@gmail.com

