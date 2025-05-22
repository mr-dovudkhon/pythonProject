from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from get_formatted_output import get_laptop_requirements_from_user_input

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load datasets
df_laptops = pd.read_csv("data/laptops_1000.csv")
df_user_needs = pd.read_csv("data/user_needs.csv")

# Load sklearn model and preprocessors
model = joblib.load("models/classifier.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

# Label encoders for categorical features
le_gpu = LabelEncoder().fit(df_laptops["GPU"])
le_cpu = LabelEncoder().fit(df_laptops["CPU"])

# Store answers across steps
session_answers = {}


@app.get("/")
def welcome(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})


# @app.post("/use_case")
# def ask_budget(request: Request, use_case: str = Form(...)):
#     session_answers["use_case"] = use_case
#     return templates.TemplateResponse("ask_budget.html", {"request": request})
#
#
# @app.post("/budget")
# def ask_condition(request: Request, budget: int = Form(...)):
#     session_answers["budget"] = budget
#     return templates.TemplateResponse("ask_condition.html", {"request": request})
#
#
# @app.post("/condition")
# def show_results(request: Request, condition: str = Form(...)):
#     session_answers["condition"] = condition
#     return recommend_from_session(request)
#
# def recommend_from_session(request):
#     spec = session_answers
#     df = df_laptops[df_laptops["Price_USD"] <= spec.get("Max_Budget_USD", 9999)].copy()
#     if spec.get("condition") and spec["condition"] != "Any":
#         df = df[df["Condition"] == spec["condition"]]
#
#     # Build model input
#     X_input = pd.DataFrame({
#         "RAM_GB": df["RAM_GB"],
#         "Storage_GB": df["Storage_GB"],
#         "Price_USD": df["Price_USD"],
#         "GPU": df["GPU"],
#         "CPU": df["CPU"],
#         "Use_Case": spec.get("use_case", "Unknown")
#     })
#
#     cat_features = encoder.transform(X_input[["GPU", "CPU", "Use_Case"]])
#     num_features = scaler.transform(X_input[["RAM_GB", "Storage_GB", "Price_USD"]])
#     X_final = np.hstack([num_features, cat_features])
#
#     df["Score"] = model.predict(X_final)
#     top_laptops = df.sort_values(by=["Score", "Price_USD"], ascending=[False, True]).head(5)
#
#     return templates.TemplateResponse("results.html", {
#         "request": request,
#         "laptops": top_laptops.to_dict(orient="records"),
#         "reasoning": f"Based on your needs as a {spec.get('use_case', 'user')}, here are the top laptops recommended."
#     })

# @app.post("/nlp_input")
# def handle_nlp_input(request: Request, description: str = Form(...)):
#     user_input: str = description
#     print(user_input, "user's input")
#     try:
#         structured = get_laptop_requirements_from_user_input(user_input)
#         session_answers.update(structured)
#         # Redirect to budget form like in Option 1
#         return templates.TemplateResponse("ask_budget.html", {"request": request})
#     except Exception as e:
#         return templates.TemplateResponse("welcome.html", {"request": request, "error": str(e)})


@app.post("/nlp_input")
def handle_nlp_input(request: Request, description: str = Form(...)):
    user_input: str = description
    try:
        structured = get_laptop_requirements_from_user_input(user_input)
        print(structured, "==========structured list ============")
        session_answers.update(structured)
        # Flow continues to ask budget like option 1
        return templates.TemplateResponse("ask_budget.html", {"request": request})
    except Exception as e:
        return templates.TemplateResponse("welcome.html", {"request": request, "error": str(e)})


@app.post("/use_case")
def ask_budget(request: Request, use_case: str = Form(...)):
    session_answers["use_case"] = use_case
    return templates.TemplateResponse("ask_budget.html", {"request": request})


@app.post("/budget")
def ask_condition(request: Request, budget: int = Form(...)):
    session_answers["Max_Budget_USD"] = budget
    return templates.TemplateResponse("ask_condition.html", {"request": request})


@app.post("/condition")
def show_results(request: Request, condition: str = Form(...)):
    session_answers["condition"] = condition
    return recommend_from_session(request)

#
# def recommend_from_session(request):
#     spec = session_answers
#     df = df_laptops.copy()
#
#     # Load default specs if coming from Option 1 (dropdown only)
#     if spec.get("use_case") and "Min_RAM_GB" not in spec:
#         match = df_user_needs[df_user_needs["Use_Case"] == spec["use_case"]]
#         if not match.empty:
#             row = match.iloc[0]
#             spec.update({
#                 "Min_RAM_GB": row["Min_RAM_GB"],
#                 "Min_Storage_GB": row["Min_Storage_GB"],
#                 "Required_GPU": row["Required_GPU"],
#                 "Preferred_CPU": row["Preferred_CPU"]
#             })
#
#     # Primary filters from Option 1 or Option 2
#     if spec.get("Max_Budget_USD"):
#         df = df[df["Price_USD"] <= spec["Max_Budget_USD"]]
#     if spec.get("condition") and spec["condition"] != "Any":
#         df = df[df["Condition"] == spec["condition"]]
#     if spec.get("Min_RAM_GB"):
#         df = df[df["RAM_GB"] >= spec["Min_RAM_GB"]]
#     if spec.get("Min_Storage_GB"):
#         df = df[df["Storage_GB"] >= spec["Min_Storage_GB"]]
#     if spec.get("Required_GPU"):
#         df = df[df["GPU"] == spec["Required_GPU"]]
#     if spec.get("Preferred_CPU"):
#         df = df[df["CPU"].str.contains(spec["Preferred_CPU"], case=False, na=False)]
#
#     # Fallback if too strict
#     fallback_mode = False
#     if df.empty:
#         fallback_mode = True
#         df = df_laptops.copy()
#         if spec.get("Max_Budget_USD"):
#             df = df[df["Price_USD"] <= spec["Max_Budget_USD"]]
#         if spec.get("condition") and spec["condition"] != "Any":
#             df = df[df["Condition"] == spec["condition"]]
#         if spec.get("Min_RAM_GB"):
#             df = df[df["RAM_GB"] >= spec["Min_RAM_GB"]]
#         if spec.get("Min_Storage_GB"):
#             df = df[df["Storage_GB"] >= spec["Min_Storage_GB"]]
#
#     if df.empty:
#         return templates.TemplateResponse("results.html", {
#             "request": request,
#             "laptops": [],
#             "reasoning": "No laptops matched your criteria. Please try again."
#         })
#
#     # Prepare input for scoring
#     X_input = pd.DataFrame({
#         "RAM_GB": df["RAM_GB"],
#         "Storage_GB": df["Storage_GB"],
#         "Price_USD": df["Price_USD"],
#         "GPU": df["GPU"],
#         "CPU": df["CPU"],
#         "Use_Case": spec.get("use_case", "Unknown")
#     })
#
#     try:
#         cat_features = encoder.transform(X_input[["GPU", "CPU", "Use_Case"]])
#         num_features = scaler.transform(X_input[["RAM_GB", "Storage_GB", "Price_USD"]])
#         X_final = np.hstack([num_features, cat_features])
#         df["Score"] = model.predict(X_final)
#         top_laptops = df.sort_values(by=["Score", "Price_USD"], ascending=[False, True]).head(5)
#     except Exception as e:
#         return templates.TemplateResponse("results.html", {
#             "request": request,
#             "laptops": [],
#             "reasoning": f"Error scoring laptops: {str(e)}"
#         })
#
#     reasoning = ("We couldn’t find exact matches, but here are the closest laptops based on your RAM, storage, and budget."
#                  if fallback_mode else
#                  f"Based on your needs as a {spec.get('use_case', 'user')}, here are the top laptops recommended.")
#
#     return templates.TemplateResponse("results.html", {
#         "request": request,
#         "laptops": top_laptops.to_dict(orient="records"),
#         "reasoning": reasoning
#     })

def normalize_spec(spec):
    result = {
        "use_case": spec.get("use_case", "General User"),
        "Min_RAM_GB": int(spec.get("Min_RAM_GB") or 8),
        "Min_Storage_GB": int(spec.get("Min_Storage_GB") or 256),
        "Required_GPU": "Dedicated" if "dedicated" in str(spec.get("Required_GPU", "")).lower() else "Integrated",
        "Preferred_CPU": spec.get("Preferred_CPU", "i5"),
        "Max_Budget_USD": int(spec.get("Max_Budget_USD") or spec.get("budget", 1000)),
        "condition": spec.get("condition", "Any")
    }
    print("Result:--", result)
    return result

def recommend_from_session(request):
    raw_spec = session_answers
    spec = normalize_spec(raw_spec)
    df = df_laptops.copy()

    # Apply filters from combined input (manual + Gemini)
    if spec.get("Max_Budget_USD"):
        df = df[df["Price_USD"] <= spec["Max_Budget_USD"]]
    if spec.get("condition") and spec["condition"] != "Any":
        df = df[df["Condition"] == spec["condition"]]
    if spec.get("Min_RAM_GB"):
        df = df[df["RAM_GB"] >= spec["Min_RAM_GB"]]
    if spec.get("Min_Storage_GB"):
        df = df[df["Storage_GB"] >= spec["Min_Storage_GB"]]
    if spec.get("Required_GPU"):
        df = df[df["GPU"] == spec["Required_GPU"]]
    if spec.get("Preferred_CPU"):
        df = df[df["CPU"].str.contains(spec["Preferred_CPU"], case=False, na=False)]

    print("--------", df.head())
    # Fallback mode
    fallback_mode = False
    if df.empty:
        fallback_mode = True
        df = df_laptops.copy()
        df = df[df["Price_USD"] <= spec.get("Max_Budget_USD", 9999)]
        if spec.get("condition") and spec["condition"] != "Any":
            df = df[df["Condition"] == spec["condition"]]
        df = df[df["RAM_GB"] >= spec.get("Min_RAM_GB", 8)]
        df = df[df["Storage_GB"] >= spec.get("Min_Storage_GB", 256)]

    if df.empty:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "laptops": [],
            "reasoning": "No laptops matched your criteria. Please try again."
        })

    X_input = pd.DataFrame({
        "RAM_GB": df["RAM_GB"],
        "Storage_GB": df["Storage_GB"],
        "Price_USD": df["Price_USD"],
        "GPU": df["GPU"],
        "CPU": df["CPU"],
        "Use_Case": spec.get("use_case", "Unknown")
    })
    print("after X_input.................")

    try:
        cat_features = encoder.transform(X_input[["GPU", "CPU", "Use_Case"]])
        print(cat_features, cat_features.shape, "---------------------------------")
        num_features = scaler.transform(X_input[["RAM_GB", "Storage_GB", "Price_USD"]])
        X_final = np.hstack([num_features, cat_features])
        df["Score"] = model.predict(X_final)
        top_laptops = df.sort_values(by=["Score", "Price_USD"], ascending=[False, True]).head(5)
        print("Top laptops:-------------", top_laptops)
    except Exception as e:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "laptops": [],
            "reasoning": f"Error scoring laptops: {str(e)}"
        })

    reasoning = (
        "We couldn’t find exact matches, but here are the closest laptops based on your RAM, storage, and budget."
        if fallback_mode else
        f"Based on your needs as a {spec.get('use_case', 'user')}, here are the top laptops recommended."
    )

    return templates.TemplateResponse("results.html", {
        "request": request,
        "laptops": top_laptops.to_dict(orient="records"),
        "reasoning": reasoning
    })