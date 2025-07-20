## ⚖️ AlgoJury: Algorithmic Fairness and Bias Auditing Tool
- A fairness auditing web application that evaluates ML models for bias using statistical fairness metrics.

Live Demo  
- [Coming Soon – Deploying via Render/Netlify]

---

## Features

- Upload and evaluate ML models for fairness
- Accepts:
  - Trained `.pkl` model file
  - CSV dataset file
- Lets user select:
  - Target column
  - Sensitive attribute (e.g., Gender, Caste)
- Calculates:
  - Demographic Parity
  - Equal Opportunity
- Visualizes results with:
  - Metric scores
  - Bar plots for fairness comparison
- Offers fairness suggestions
- Load sample model and dataset for quick testing

---

## Fairness Metrics

- **Demographic Parity:** Compares positive outcome rates across sensitive groups
- **Equal Opportunity:** Compares true positive rates across sensitive groups

---

## Project Structure
algojury/
├── app.py  
├── model/  
│   └── sample_model.pkl  
├── static/  
│   ├── css/style.css  
│   └── js/main.js  
├── templates/  
│   └── index.html  
├── utils/  
│   ├── audit.py  
│   └── helpers.py  
├── uploads/  
│   ├── uploaded_model.pkl  
│   └── uploaded_data.csv  
├── requirements.txt  
└── README.md  

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **ML Libraries:** scikit-learn, pandas, matplotlib, seaborn

---

## Author
- Sanya Gautam  
- GitHub: [@sanyagautam12](https://github.com/sanyagautam12)  
- Portfolio: Coming Soon

---

## Show Your Support
- If you like this project, please ⭐ the repo and share it!
