# Maternal Anemia Risk Intelligence Dashboard

## Project Overview
This Streamlit dashboard is designed to support anemia research in pregnant women in North Lombok Regency. The application combines spatial hotspot mapping, text mining analytics for Iron-Folic Acid (IFA) supplement consumption barriers, and machine learning-based risk segmentation to develop actionable intervention recommendations for healthcare workers.

## Key Features
- **Main Dashboard** with key metrics, summary visualizations, and quick access to analytical modules.
- **Module 1 – Anemia Hotspot Mapping**: Spatial K-Means clustering with interactive Folium maps, cluster statistics, and result export.
- **Module 2 – IFA Barriers Analysis**: Indonesian NLP (tokenization, Sastrawi stemming, sentiment), word clouds, geographic heatmaps, automated recommendations.
- **Module 3 – Risk Segmentation**: Random Forest classifier with hyperparameter tuning, comprehensive evaluation, new case prediction forms, batch prediction, what-if analysis, similar case search, intervention recommendations.
- **Professional theme** with custom styling, production-ready deployment.

## Project Structure
```
anemia-dashboard/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── custom.css
│   └── images/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── deployment_guide.md
│   └── user_manual.md
├── models/
├── modules/
│   ├── modul1_hotspot_mapping.py
│   ├── modul2_ttd_barriers.py
│   └── modul3_risk_segmentation.py
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── export_manager.py
└── assets/
    └── logo.png (optional)
```

## System Requirements
- Python 3.9 or newer
- Pip / virtual environment (recommended)
- Streamlit 1.28 or newer

## Installation
1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the survey Excel file in the `data/raw/` folder according to the name in `config.py`.

## Running the Application
```bash
streamlit run app.py
```
The application will open in your local browser (default: http://localhost:8501).

## Usage Guide
1. **Main Dashboard**: Monitor summary metrics and initial visualizations. Use CTA buttons to navigate between modules.
2. **Module 1**: Configure cluster count and additional features in the sidebar, explore the map, download results.
3. **Module 2**: Use demographic filters, review word clouds per category, download analysis summary.
4. **Module 3**: View model performance, perform new case or batch predictions, simulate scenarios, download high-risk list.
5. **About**: Read methodology, data sources, and project contact information.

## Screenshot (Placeholder)
- ![Dashboard Overview](docs/images/dashboard_overview.png)
- ![Hotspot Map](docs/images/hotspot_map.png)
- ![TTD Barriers](docs/images/ttd_barriers.png)
- ![Risk Segmentation](docs/images/risk_segmentation.png)

## Prepare a VPS (Ubuntu) with Python 3.9+ and pip.
2. Copy the project code to the server (git clone or rsync).
3. Install dependencies using `pip install -r requirements.txt`.
4. Use `pm2`, `systemd`, or `supervisor` to keep the Streamlit service running.
5. Configure reverse proxy (Nginx) for custom domain and HTTPS.
6. Update `.streamlit/config.toml` for production server configuration.
7. Ensure data files are placed in `data/raw/` and update `config.py` if the name/location differs.

Detailed guide available in `docs/deployment_guide.md`.

## Contributing
- Fork the repository
- Create a feature branch: `git checkout -b your-feature`
- Commit changes: `git commit -m "Add feature X"`
- Push branch: `git push origin your-feature`
- Submit a pull request with detailed description

## License
This project is licensed under the MIT License. See the LICENSE file if available.

## Contact
- Email: prayaadhiganaglobal@gmail.com
- WhatsApp: +62 812-3456-7890
- Website: https://kluanemia.example.com

## Data Privacy & Security
⚠️ **Important:** This repository does NOT include any personally identifiable information (PII) or sensitive survey data. All data files are excluded via `.gitignore` for privacy protection.

---
Developed by the KLU research team to improve anemia interventions in pregnant women through modern analytics
Dikembangkan oleh tim penelitian KLU untuk meningkatkan intervensi anemia pada ibu hamil melalui pemanfaatan analitik modern.
