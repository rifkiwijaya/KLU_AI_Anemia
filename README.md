# KLU AI Anemia - Maternal Anemia Risk Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

An AI-powered dashboard system for analyzing and predicting maternal anemia risk in North Lombok Regency, Indonesia. This research project combines spatial analysis, natural language processing, and machine learning to support data-driven healthcare interventions for pregnant women. You can access in http://206.189.81.131:8501/ (if server up)

## 🏗️ Repository Structure

```
KLU_AI_Anemia/
├── anemia-dashboard/          # Main Streamlit application
│   ├── app.py                 # Application entry point
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   ├── modules/               # Analytical modules
│   │   ├── modul1_hotspot_mapping.py
│   │   ├── modul2_ttd_barriers.py
│   │   └── modul3_risk_segmentation.py
│   ├── utils/                 # Utility functions
│   ├── data/                  # Data directory (excluded from Git)
│   ├── models/                # Trained ML models
│   └── assets/                # Static assets (CSS, images)
└── README.md                  # This file
```

## ✨ Key Features

### 📍 Module 1: Spatial Hotspot Mapping
- K-Means clustering for anemia prevalence
- Interactive Folium maps with risk visualization
- Geographic pattern analysis by district and village
- Export capabilities for GIS integration

### 📋 Module 2: IFA Barriers Analysis
- Indonesian text mining and sentiment analysis
- Sastrawi stemming and tokenization
- Word cloud visualization of consumption barriers
- Automated recommendation generation based on text patterns

### 🎯 Module 3: Risk Segmentation
- Random Forest classifier with hyperparameter optimization
- Multi-class risk prediction (Low/Medium/High)
- What-if scenario analysis
- Similar case identification using k-NN
- Batch prediction capabilities
- Evidence-based intervention recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rifkiwijaya/KLU_AI_Anemia.git
   cd KLU_AI_Anemia
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   cd anemia-dashboard
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place your survey Excel file in `anemia-dashboard/data/raw/`
   - Update `config.py` if your filename differs from the default

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   - Open your browser to http://localhost:8501

## 📊 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit, Python 3.9+ |
| **Machine Learning** | scikit-learn, Random Forest, k-NN |
| **NLP** | Sastrawi, NLTK, WordCloud |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Plotly, Folium, Matplotlib, Seaborn |
| **Geospatial** | Folium, GeoPandas-compatible |

## 📈 Model Performance

The risk classification model achieves:
- **Accuracy**: ~85-90% on test set
- **F1-Score**: Weighted average across risk classes
- **Cross-validation**: Stratified k-fold with hyperparameter tuning
- **Class balancing**: Handles imbalanced risk distributions

## 🔒 Data Privacy & Security

⚠️ **Important Security Notice:**
- This repository does **NOT** contain any personally identifiable information (PII)
- All survey data files are excluded via `.gitignore`
- No sensitive patient information is committed to version control
- Data files must be obtained separately and placed locally

## 🌐 Deployment

For production deployment, see the detailed guide in `anemia-dashboard/docs/deployment_guide.md`.

### Quick Deployment Options:
- **Streamlit Cloud**: Direct deployment from GitHub
- **Docker**: Container-based deployment (Dockerfile included)
- **VPS/Cloud**: Traditional server deployment with Nginx reverse proxy

## 📖 Documentation

- [User Manual](anemia-dashboard/docs/user_manual.md) - End-user guide
- [Deployment Guide](anemia-dashboard/docs/deployment_guide.md) - Production setup
- [API Documentation](anemia-dashboard/docs/api_docs.md) - Technical reference

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update documentation as needed

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{klu_anemia_dashboard_2026,
  title = {KLU AI Anemia: Maternal Anemia Risk Intelligence System},
  author = {KLU Research Team},
  year = {2026},
  url = {https://github.com/rifkiwijaya/KLU_AI_Anemia}
}
```


## 🙏 Acknowledgments

- North Lombok Regency Health Department for survey data
- Healthcare workers and pregnant women participants
- KLU University research team
- Open-source community for excellent tools and libraries

---

**Version**: 0.1.0  
**Last Updated**: December 2025  
**Status**: Active Development

Made with ❤️ by KLU Research Team for improving maternal health outcomes
