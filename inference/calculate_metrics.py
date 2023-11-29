import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr, mannwhitneyu


def prc():
    y_df = pd.read_csv(
        "/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/inference/Adam_best_models/resnet50_CADLab_recipient_predictions.csv"
    )
    y_test = y_df["label"]
    y_pred = y_df[["pred_prob(class=0)", "pred_prob(class=1)", "pred_prob(class=2)"]]

    binarize = LabelBinarizer()
    binarize.fit(y_test)
    y_test_binarized = binarize.transform(y_test)

    print("Metric: PRC")
    print(
        "Average over classes: ",
        average_precision_score(y_test_binarized, y_pred, average="macro"),
    )
    print(
        "Per class: ", average_precision_score(y_test_binarized, y_pred, average=None)
    )
    print(
        "Baseline Class 0 PRC: ",
        y_test_binarized[:, 0].sum() / y_test_binarized.shape[0],
    )
    print(
        "Baseline Class 1 PRC: ",
        y_test_binarized[:, 1].sum() / y_test_binarized.shape[0],
    )
    print(
        "Baseline Class 2 PRC: ",
        y_test_binarized[:, 2].sum() / y_test_binarized.shape[0],
    )


def pc9_correlations():
    df_pc9 = pd.read_csv(
        "/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/inference/resnet50_CADLab_recipient_pca-features.csv",
        index_col="id",
        usecols=["id", "feature_pca_8"],
        skiprows=range(39, 131),
    )
    df_region = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/EVLP Outcome.xlsx",
        sheet_name="Region, dPO2, STEEN Lost",
        index_col="EVLP ID",
        usecols=[0, *range(5, 17)],
    )
    df_image_abnorm_1h = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Image (First 170).xlsx",
        sheet_name="first_hour",
        index_col=0,
        skiprows=range(1, 625),
    )
    df_image_abnorm_3h = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Image (First 170).xlsx",
        sheet_name="third_hour",
        index_col=0,
        skiprows=range(1, 625),
    )
    assert df_image_abnorm_1h.index.all() == df_image_abnorm_3h.index.all()

    df_delta_region = pd.DataFrame()
    df_delta_image = pd.DataFrame()

    df_delta_region["dRUL"] = df_region["RUL (3rd hour)"] - df_region["RUL (1st hour)"]
    df_delta_region["dRML"] = df_region["RML (3rd hour)"] - df_region["RML (1st hour)"]
    df_delta_region["dRLL"] = df_region["RLL (3rd hour)"] - df_region["RLL (1st hour)"]
    df_delta_region["dLUL"] = df_region["LUL (3rd hour)"] - df_region["LUL (1st hour)"]
    df_delta_region["dLingula"] = (
        df_region["Lingula (3rd hour)"] - df_region["Lingula (1st hour)"]
    )
    df_delta_region["dLLL"] = df_region["LLL (3rd hour)"] - df_region["LLL (1st hour)"]

    df_delta_image["dConsolidation"] = (
        df_image_abnorm_3h["Consolidation"] - df_image_abnorm_1h["Consolidation"]
    )
    df_delta_image["dInfiltrate"] = (
        df_image_abnorm_3h["Infiltrate"] - df_image_abnorm_1h["Infiltrate"]
    )
    df_delta_image["dAtelectasis"] = (
        df_image_abnorm_3h["Atelectasis"] - df_image_abnorm_1h["Atelectasis"]
    )
    df_delta_image["dNodules"] = (
        df_image_abnorm_3h["Nodules"] - df_image_abnorm_1h["Nodules"]
    )
    df_delta_image["dInterstitial Lines"] = (
        df_image_abnorm_3h["Interstitial Lines"]
        - df_image_abnorm_1h["Interstitial Lines"]
    )
    df_delta_image["dAll"] = df_delta_image.sum(axis=1)

    df_all = df_pc9.merge(df_delta_image, left_index=True, right_on="EVLP_ID")
    df_all = df_all.merge(df_delta_region, left_index=True, right_on="EVLP ID")

    for col in df_all.columns:
        print("\n", col)
        print(spearmanr(df_all["feature_pca_8"], df_all[col]))

    df_cons_region = pd.DataFrame()
    df_RUL = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Region (for Regression).xlsx",
        sheet_name="RUL",
        index_col="EVLP ID",
        usecols=[1, 2, 4, 5, 6, 7, 8],
    )
    df_RML = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Region (for Regression).xlsx",
        sheet_name="RML",
        index_col="EVLP ID",
        usecols=[1, 2, 4, 5, 6, 7, 8],
    )
    df_RLL = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Region (for Regression).xlsx",
        sheet_name="RLL",
        index_col="EVLP ID",
        usecols=[1, 2, 4, 5, 6, 7, 8],
    )
    df_LUL = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Region (for Regression).xlsx",
        sheet_name="LUL",
        index_col="EVLP ID",
        usecols=[1, 2, 4, 5, 6, 7, 8],
    )
    df_Lingula = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Region (for Regression).xlsx",
        sheet_name="Lingula",
        index_col="EVLP ID",
        usecols=[1, 2, 4, 5, 6, 7, 8],
    )
    df_LLL = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Region (for Regression).xlsx",
        sheet_name="LLL",
        index_col="EVLP ID",
        usecols=[1, 2, 4, 5, 6, 7, 8],
    )
    df_cons_region["RUL, 1hr"] = df_RUL["Cons (0 to 3)"][df_RUL["Time Point (hr)"] == 1]
    df_cons_region["RUL, 3hr"] = df_RUL["Cons (0 to 3)"][df_RUL["Time Point (hr)"] == 3]
    df_cons_region["RML, 1hr"] = df_RML["Cons (0 to 3)"][df_RML["Time Point (hr)"] == 1]
    df_cons_region["RML, 3hr"] = df_RML["Cons (0 to 3)"][df_RML["Time Point (hr)"] == 3]
    df_cons_region["RLL, 1hr"] = df_RLL["Cons (0 to 3)"][df_RLL["Time Point (hr)"] == 1]
    df_cons_region["RLL, 3hr"] = df_RLL["Cons (0 to 3)"][df_RLL["Time Point (hr)"] == 3]
    df_cons_region["LUL, 1hr"] = df_LUL["Cons (0 to 3)"][df_LUL["Time Point (hr)"] == 1]
    df_cons_region["LUL, 3hr"] = df_LUL["Cons (0 to 3)"][df_LUL["Time Point (hr)"] == 3]
    df_cons_region["Lingula, 1hr"] = df_Lingula["Cons (0 to 3)"][
        df_Lingula["Time Point (hr)"] == 1
    ]
    df_cons_region["Lingula, 3hr"] = df_Lingula["Cons (0 to 3)"][
        df_Lingula["Time Point (hr)"] == 3
    ]
    df_cons_region["LLL, 1hr"] = df_LLL["Cons (0 to 3)"][df_LLL["Time Point (hr)"] == 1]
    df_cons_region["LLL, 3hr"] = df_LLL["Cons (0 to 3)"][df_LLL["Time Point (hr)"] == 3]

    df_cons_region["dRUL"] = df_cons_region["RUL, 3hr"] - df_cons_region["RUL, 1hr"]
    df_cons_region["dRML"] = df_cons_region["RML, 3hr"] - df_cons_region["RML, 1hr"]
    df_cons_region["dRLL"] = df_cons_region["RLL, 3hr"] - df_cons_region["RLL, 1hr"]
    df_cons_region["dLUL"] = df_cons_region["LUL, 3hr"] - df_cons_region["LUL, 1hr"]
    df_cons_region["dLingula"] = (
        df_cons_region["Lingula, 3hr"] - df_cons_region["Lingula, 1hr"]
    )
    df_cons_region["dLLL"] = df_cons_region["LLL, 3hr"] - df_cons_region["LLL, 1hr"]

    df_cons_pc9 = df_pc9.merge(df_cons_region, left_index=True, right_on="EVLP ID")
    print("\n-----------------Consolidation Only-----------------")

    for col in df_cons_pc9.columns:
        print("\n", col)
        print(spearmanr(df_cons_pc9["feature_pca_8"], df_cons_pc9[col]))


def donortype_RLS():
    df_donortype = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/EVLP Outcome.xlsx",
        sheet_name="Donor Info",
        index_col="EVLP ID No",
        usecols=[0, 1],
    )
    df_donortype = df_donortype.replace({"U": np.nan})
    df_rls = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Score By Image (First 170).xlsx",
        sheet_name="features_outcome",
        index_col=0,
    )
    df_rls = df_rls.drop(columns="Vent Day Outcome")
    df_rls["sum_1h"] = df_rls.iloc[:, 0:5].sum(axis=1)
    df_rls["sum_3h"] = df_rls.iloc[:, 5:10].sum(axis=1)
    df_rls["sum_all"] = df_rls.iloc[:, 0:10].sum(axis=1)
    df_donortype_rls = df_donortype.merge(df_rls, left_index=True, right_on="EVLP ID")

    for sum in ["sum_1h", "sum_3h", "sum_all"]:
        print(df_donortype_rls.groupby("Donor Type")[sum].mean())
        print(
            mannwhitneyu(
                df_donortype_rls[sum][df_donortype_rls["Donor Type"] == "DBD"],
                df_donortype_rls[sum][df_donortype_rls["Donor Type"] == "DCD"],
            )
        )


def radiology_impression_pc9():
    df_pc9 = pd.read_csv(
        "/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/inference/resnet50_CADLab_recipient_pca-features.csv",
        index_col="id",
        usecols=["id", "feature_pca_8"],
        skiprows=range(39, 131),
    )
    df_impression = pd.read_excel(
        "/home/bonnie/Documents/OneDrive_UofT/BME_PhD/EVLP Radiology/EVLP_ImageLabel_ScoringSheet/Scoring Sheet (First 170 Cases Labelled).xlsx",
        sheet_name="Edited Scoring Sheet",
        usecols=[0, 14],
    )
    df_impression = df_impression.dropna()
    df_impression["Notes from Labelling"] = df_impression[
        "Notes from Labelling"
    ].str.lower()
    df_impression["Notes from Labelling"] = df_impression[
        "Notes from Labelling"
    ].replace({"better": 1, "stable": 0, "worse (slight)": -0.5, "worse": -1})
    df_impression = df_impression.drop_duplicates(
        subset=["EVLP ID"], keep="last"
    ).set_index("EVLP ID")
    df_impression["Notes from Labelling"] = df_impression[
        "Notes from Labelling"
    ].astype(float)

    df_pc9_impression = df_pc9.merge(df_impression, left_index=True, right_on="EVLP ID")
    print(df_pc9_impression.dtypes)
    print(
        spearmanr(
            df_pc9_impression["feature_pca_8"],
            df_pc9_impression["Notes from Labelling"],
        )
    )


radiology_impression_pc9()
