import torch
import numpy as np
import pandas as pd
from conf import *
from model import MyModel
import time

def analysis():
    start_time = time.time()

    data = pd.read_csv(DATA_PATH, header=None)

    # Load model
    model_path = './outputs/checkpoints/best_model.pth'
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract features and labels
    data.columns = [
    "Energy","Shell","MFP",
    "MAC_Total","MAC_Incoherent","MAC_Coherent","MAC_Photoelectric","MAC_Pair_production",
    "Inf_Flu_BUF","Fin_Flu_BUF","Inf_Exp_BUF","Fin_Exp_BUF","Inf_Eff_BUF","Fin_Eff_BUF"]
    features = data.drop(columns=["MAC_Coherent"]).iloc[:, :7].values
    targets = data.iloc[:, 13:14].values  # True values
    
    X = data.drop(columns=["MAC_Coherent"]).iloc[:, :7].copy()
    X["Energy"] = np.log(X["Energy"])
    X = X.values
    X = torch.tensor(X, dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for row in X:
            row = row.unsqueeze(0).to(device)   # shape [1, 7]
            out = model(row)                     # forward
            preds.append(out.cpu().numpy().squeeze())
            torch.cuda.empty_cache()             # 及时释放

    predictions = np.exp(np.array(preds)).reshape(-1, 1)
    rel_errors = np.abs((predictions - targets) / targets) * 100

    results = np.hstack((features,predictions, targets, rel_errors))

    results_columns = [
        "Energy","Shell","MFP",
        "MAC_Total","MAC_Incoherent","MAC_Photoelectric","MAC_Pair_production",
        "Predictions","Target","Rel_Error(%)"
    ]

    df_results = pd.DataFrame(results, columns=results_columns)
    df_results = df_results.sort_values(by="Rel_Error(%)", ascending=False)
    df_results.to_csv("./outputs/analysis_results.csv", index=False, encoding="utf-8-sig")
    print("Analysis results saved to ./outputs/analysis_results.csv")

    end_time = time.time()
    total_run_time = end_time - start_time
    hours, remainder = divmod(total_run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'Total program execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s')
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    analysis()
