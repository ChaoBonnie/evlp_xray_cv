import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score


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
print("Per class: ", average_precision_score(y_test_binarized, y_pred, average=None))
print(
    "Baseline Class 0 PRC: ", y_test_binarized[:, 0].sum() / y_test_binarized.shape[0]
)
print(
    "Baseline Class 1 PRC: ", y_test_binarized[:, 1].sum() / y_test_binarized.shape[0]
)
print(
    "Baseline Class 2 PRC: ", y_test_binarized[:, 2].sum() / y_test_binarized.shape[0]
)
