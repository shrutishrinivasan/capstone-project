# AI Assisted Personal Finance Management System

<p align="justify">
A lightweight, privacy-focused financial management tool with intelligent features to help you make better financial decisions without requiring external account integration.
</p>

<div align="center">
  <img src="output/about.PNG" width="800"/>
</div>

## Key Features
- **Butterfly Effect Simulator:** LSTM model showing the compounding impact of small financial changes over time, reflecting the butterfly effect concept of chaos theory in personal finance.
- **Financial Scenario Tester:** XGBoost model for stress testing various financial situations (market crash, medical emergency, job loss) and assessing likelihood of reaching financial milestones (education, home purchase, investments).
- **AI-Powered Assistant:** Custom chatbot using Mistral Saba 24B to answer both transactional and financial queries.
- **Privacy-First Design:** No external account integration required, eliminating data security concerns.

## Tech Stack
### Frontend  
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/) [![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML) [![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)

### Backend 
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![ChromaDB](https://img.shields.io/badge/ChromaDB-303030?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB...)](https://www.trychroma.com/) [![Groq](https://img.shields.io/badge/Groq_API-FF6B6B?style=for-the-badge&logoColor=white)](https://groq.com/)

### Database  
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

### Libraries  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) [![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/) [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/) [![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge)](https://www.langchain.com/)  

### Evaluation
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![NLTK](https://img.shields.io/badge/NLTK-4B8BBE?style=for-the-badge&logoColor=white)](https://www.nltk.org/) [![ROUGE](https://img.shields.io/badge/ROUGE-E34F26?style=for-the-badge&logoColor=white)](https://pypi.org/project/rouge-score/) [![SacreBLEU](https://img.shields.io/badge/SacreBLEU-00BFFF?style=for-the-badge&logoColor=white)](https://github.com/mjpost/sacrebleu) [![Ragas](https://img.shields.io/badge/Ragas-111111?style=for-the-badge&logoColor=white)](https://github.com/explodinggradients/ragas) [![asyncio](https://img.shields.io/badge/asyncio-6E57E0?style=for-the-badge)](https://docs.python.org/3/library/asyncio.html)

## Installation

### Prerequisites
- Windows 10/11
- Minimum 8GB RAM
- MySQL
- Groq API access

### Set up
1. Clone the repository.
   
   `git clone https://github.com/shrutishrinivasan/capstone-project.git`

2. Install dependencies.
   
   `pip install -r requirements.txt`

3. Configure API Access.
   - Open the `.env` file in the root directory.  
   - Add your Groq API key:  
     ```env
     GROQ_API_KEY=your_api_key_here
   - Update MySQL credentials in `chatbot.py`

4. Launch the application.
   
   `python -m streamlit run app.py`

## Usage Guide
### Application Layout
- **Landing Page:** About, Features, Tools, Bot, Learn, Login sections
- **Personal Dashboard:** Getting Started, Upload Data, Overview, Income/Expense, Financial Foresight, Custom Bot, Resources, Logout sections 

### Frontend Screenshots
### Overview Section
<div align="center">
  <img src="output/overview1.PNG" width="800"/>
  <img src="output/overview4.PNG" width="800"/>
</div>

### Chatbot Section
<div align="center">
  <img src="output/data_digger.PNG" width="800"/>
  <img src="output/fin_mentor.PNG" width="800"/>
</div>

### Butterfly Effect Simulator
<div align="center">
  <img src="output/butterfly_effect1.PNG" width="800"/>
</div>

### Financial Scenario Tester
<div align="center">
  <img src="output/scenario_tester1.PNG" width="800"/>
</div>

To see more frontend screenshots, please refer the output folder.
