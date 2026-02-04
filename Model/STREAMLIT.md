# Streamlit Frontend

Run the app from the project root:

```powershell
streamlit run Model/streamlit_app.py
```

The app uses:
- `Model/best_nn_model.keras` (trained neural network)
- `Model/cleaned_telco_churn.csv` (reference preprocessing schema)

For batch prediction CSV upload, include these columns:
- `gender`
- `seniorcitizen`
- `partner`
- `dependents`
- `tenure`
- `phoneservice`
- `multiplelines`
- `internetservice`
- `onlinesecurity`
- `onlinebackup`
- `deviceprotection`
- `techsupport`
- `streamingtv`
- `streamingmovies`
- `contract`
- `paperlessbilling`
- `paymentmethod`
- `monthlycharges`
- `totalcharges`
