
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from pathlib import Path

def run_project():
    csv_file=list(Path("data").glob("*.csv"))[0]
    df=pd.read_csv(csv_file)

    cols=df.columns.str.lower()

    amount_col=[c for c in df.columns if "amount" in c.lower()]
    age_col=[c for c in df.columns if "age" in c.lower()]
    gender_col=[c for c in df.columns if "gender" in c.lower()]
    category_col=[c for c in df.columns if "category" in c.lower()]

    if amount_col:
        amt=amount_col[0]

        plt.figure(figsize=(6,4))
        df[amt].hist()
        plt.title("Sales Distribution")
        plt.savefig("outputs/sales_trend.png")
        plt.close()

        X=np.arange(len(df)).reshape(-1,1)
        y=df[amt].values

        model=LinearRegression()
        model.fit(X,y)

        pred=model.predict(X)

        plt.figure(figsize=(6,4))
        plt.plot(y,label="Actual")
        plt.plot(pred,label="Predicted")
        plt.legend()
        plt.title("Revenue Forecast")
        plt.savefig("outputs/revenue_forecast.png")
        plt.close()

    if category_col:
        plt.figure(figsize=(6,4))
        df[category_col[0]].value_counts().head(10).plot(kind="bar")
        plt.title("Category Analysis")
        plt.savefig("outputs/category_analysis.png")
        plt.close()

    if gender_col and amount_col:
        plt.figure(figsize=(6,4))
        df.groupby(gender_col[0])[amount_col[0]].sum().plot(kind="bar")
        plt.title("Gender Analysis")
        plt.savefig("outputs/gender_analysis.png")
        plt.close()

    if age_col:
        plt.figure(figsize=(6,4))
        df[age_col[0]].hist()
        plt.title("Age Distribution")
        plt.savefig("outputs/age_distribution.png")
        plt.close()

    with open("outputs/business_report.txt","w") as f:
        f.write("Retail Sales Forecasting Report\\n")
        f.write(f"Total Records: {len(df)}\\n")

    print("Project Completed")
