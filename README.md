# NASA C-MAPSS Predictive Maintenance
<p align="center"><img src="nasa-logo.png" width="150"></p>

**[Full Code & Analysis](./NASA_C-MAPSS_Predictive_Modeling_Project.ipynb)**


## Remaining Useful Life Prediction and Failure-Risk Modeling

This project builds an end-to-end predictive maintenance workflow using **NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) and Turbofan Engine Degradation Simulation Dataset (FD001)**. The analysis uses multivariate engine sensor data to estimate **Remaining Useful Life (RUL)** and identify engines at risk of failure within the next **30 operational cycles**.

This project seeks to find the best machine-learning model by combining various tools and metrics (validations, time-series feature engineering, regression/classifcation, hyperparamter tuning, SHAP explainability, cost sensitive threshold optimization)

---

## Project Objectives

The project addresses two related predictive-maintenance questions based on the given sensor data for a number of engines:

1. **Remaining Useful Life:** How many operational cycles remain before an engine fails?
2. **Near-term failure risk:** Is an engine likely to fail within the next 30 cycles?

The regression model seeks to support maintenance planning and prioritization, while the classification model provides a more actionable warning signal for engines approaching failure.

---

## Dataset

**NASA C-MAPSS — FD001**

FD001 contains simulated run-to-failure trajectories for turbofan engines operating under one operating condition.

- **100 training engines**
- **100 test engines**
- **20,631 training observations**
- **13,096 test observations**
- **3 operational settings**
- **21 sensor measurements**

Each row represents one engine at one operational cycle. Training trajectories continue until failure, while test trajectories stop before failure. NASA provides the actual remaining life for each test engine.

Dataset sources:

- [NASA C-MAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## Methodology

### 1. Data Preparation and Exploratory Analysis

The notebook:

- creates the RUL target from each engine's failure cycle
- checks missing values and duplicate engine-cycle records
- identifies constant sensor variables
- explores engine lifetime distributions
- Analyzes sensor relationships with degradation
- Visualizes sensor trajectories across engine lifecycles

### 2. Leakage-Aware Validation

Because multiple rows belong to the same engine, randomly splitting individual observations could place cycles from the same engine in both training and validation data.

Instead, the project splits **entire engine IDs** into training and validation sets. Hyperparameter tuning uses **5-fold GroupKFold cross-validation**, ensuring that an engine never appears in both training and validation portions of the same fold.

### 3. RUL Regression Modeling

The following models were compared:

- Median baseline
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Model performance was evaluated using **MAE, RMSE, and R²**.

### 4. Time-Series Feature Engineering

To capture degradation trends rather than relying only on current sensor values, the project creates historical features within each engine:

- lagged sensor values: 1, 5, and 10 cycles
- rolling means: 5 and 20 cycles
- rolling standard deviations
- one-cycle sensor changes
- rolling sensor slopes

This produced **182 engineered model features** after removing constant variables.

### 5. Hyperparameter Tuning

XGBoost regression and classification models were tuned using **RandomizedSearchCV** with grouped cross-validation.


### 6. Failure-Within-30-Cycles Classification

The project reframes predictive maintenance as a binary classification problem:

```text
failure_within_30 = 1 if RUL <= 30 cycles
failure_within_30 = 0 otherwise
```

The following classifiers were compared:

- Logistic Regression
- Random Forest
- XGBoost
- Tuned XGBoost

Because the positive class is less common, evaluation emphasizes **precision, recall, F1, ROC-AUC, and PR-AUC** rather than accuracy alone.

### 7. Cost-Sensitive Threshold Optimization

A default probability threshold of 0.50 is not automatically the best business decision rule.

For demonstration purposes, the notebook assumes:

- **$5,000** cost for an unnecessary inspection / false positive
- **$100,000** cost for a missed imminent failure / false negative

These values are illustrative only and are not NASA cost estimates.

The model threshold is then selected by minimizing estimated validation cost.

---





## Findings/Conclusions

### A.) The project produced model capable of predicting remaining engine life much better than a simple baseline
- The first major step of the project was create a baseline model that simply predicted the median RUL for every engine.  It had a validation MAE of 55.6 cycles and RMSE of 66.4 cycles.
- Next, a couple of machine learning models were trained (linear regression, decision tree, random forest, XGboost) and the best model was Random Forest with RMSE of 33.9.
- After selecting the model, tuning, testing it with cross validation, and retraining using all available training data, the final RUL model achived the following results on the official test data:
| Model / Stage | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median Baseline — Validation | 55.45 | 66.44 | -0.002 |
| Random Forest — Raw Features | 25.44 | 33.90 | 0.739 |
| XGBoost — Raw Features | 25.81 | 34.20 | 0.735 |
| XGBoost — Historical Features | 24.06 | 32.84 | 0.755 |
| Tuned XGBoost — Validation | 24.13 | 32.21 | 0.765 |
| **Final XGBoost — NASA Test Set** | **19.25** | **25.75** | **0.616** |

Adding lag, rolling, change, and slope features improved XGBoost validation RMSE by approximately **4.0%** compared with the raw-feature XGBoost model.

- The final model's RUL predictions on average were of by 19 cycles.  Larger prediction errors increased RMSE to about 26 cycles, suggesting that some engines were more difficult to predict than others.  The model explained about 62% of the variation in RUL across the test engines.

### B.) Reframing the problem/question as a classifcation project provides useful maintence warnings
- Another model was built to answer a more "business focused" question: **How likely is this engine to fail within the next 30 cycles**
- The strongest model was XGBoost with validation PR-AUC of 0.980.
- This assumes a normal probability cutoff of 0.50
| Metric | Value |
| --- | --- |
| Precision | 0.903 |
| Recall | 0.942 |


### C.) Using business/economic assumptions, we can tune the model to better meet the business needs
- In reality, missing an engine that is close to failure could be much more expensive than performing an unnecssary inspection.
- In other words, the cost of a false negative is much greater than a false positive.
- By sacrificing the model's precision for recall, the model will better predict actual business/economic outcomes
- This notebook assumes/makes up the example cost of 100,000 dollars  for each missed failure (false negative) and 5000 for each unnecessary inspection (false postiive)
- Recalibrating the model to these parameters, the best validation threshold would be 0.06 instead of 0.50 and the following metrics:

| Metric | Value |
| --- | --- |
| Precision | 0.757 |
| Recall | 0.982 |

- Using these assumptions and test data, the estimated savings versus the baseline 0.50 model saves about $2 million (47% reduction)
- This lower threshold makes the model more likely to issue maintenence warnings which creates more false alarms, but greatly reduces the chance of missing an engine that is actually close to failure


## Other Findings:
### 1.) Sensor history improves predictions
- Additional time series variables were created to describe how sensor readings changed over time:
    - Lag: previous sensor reading
    - Rolling mean
    - Rolling std
    - Recent sensor trends (slope, m)
- SHAP analysis gives more evidence to these findings as many of the top important features were time-series features
    1. cycle
    2. sensor_4_roll_mean_5
    3. sensor_21_roll_mean_20
    4. sensor14_roll_mean_20
    5. sensor_15_roll_mean_5
- This only explains how the model is making predictions and not inherently suggesting causation.
### 2.) Errors are smaller and more important closer to engine failure
- The model does not perform equally well during every stage of an engine's life
- For example, MAE was 30.3 cycles for engines with more than 100 actual remaining cycles
- The MAE dropped to 5.2 cycles when the engine had only less than 30 cycles remaining actual
- This suggests predictions close to failure are more important for maintenence decisions
- Because of this, future models should bge judged not only by their overal RMSE, but also by how accurately it predicts engines that are closer to failure
- Prediction error varied across the engine lifecycle:

| Actual RUL Band | MAE |
|---|---:|
| 0–30 cycles | **5.17** |
| 31–60 cycles | 17.52 |
| 61–100 cycles | 27.71 |
| 100+ cycles | 30.31 |

## Business Recommendations:
1. **Use predicted RUL to help prioritize maintenence.**
    - Scheduled maintenance should incorporate RUL metric to better determine machines to prioritize
2. **Classifer model (30/x cycles) should be used as an additional warning system.**
    - Using both models gives more dimensions to the situation
    - "How much life is left" vs "Does this engine need attention soon"
3. **Choose an alert threshold based on real business costs**
    - Other subject matter experts should be consulted to calculate the actual costs for machine breakdowns versus inspection to come up with a threshold that achives maximum business savings
4. **Subject matter experts should pay attention to the most important sensors and their trends**
    - The SHAP analysis can help the engineering team identify which sensors may deserve closer monitoring.  These sensors readings should be more readily available in say a dashboard
5. **Keep engineering/maintenence professionals/other subject matter experts involved in the final decision**
    - The model should be used as a decision support tool rather than an autopilot system.  Predictions should be considered together with other things such as regular inspections, maintenance history, etc.
  
## Project Limitations
- C-MAPSS is simulated data.  Real data has more noise
- RUL model only gives one estimate.  For example, the model can predict a machine has 25 remaining cycles but it does not currently say how confident it is in that estimate
- 30 cycle warning was a arbitrary number.  Real business would analyze when the actual warning should occur
- Mainenance costs (100k, 5k) were arbitrary numbers used to illustrate the best probability threshold.  Real business would analyze the actual costs
- SHAP does not prove cause and effect.
- Some multicolinearity with engine data as measurements collected over time from one engine are still naturally connected.  Selected models are decently robust to multicolinearity however

## Recommended Next Steps
1. Create additional historical sensor features
    - no reason to assume the metrics selected here are the best. other metric examples include rolling min/max values, historical delta, changing the lagperiods/window sizes
2. Add confidence/uncertainty to the RUL predictions
    - Instread of only predicting a static RUL value, a more useful metric for business purposes would be x cycles remaining plus or minus y. 
3. Predict several maintenance windows
   - Label cycle warnings such as RUL: 15= urgent, 30 = needs maintenence, 60 =order parts, etc
4. If the model is accepted, plan how the model would be used/monitored in production
   - A real predictive maintenence model would need to be monitored after deployment
   - The model should be retrained when there is evidence that its performance has changed or real world factors (for example new types of machines, cost change, etc)

## Overall Conclusion

This notebook/project demonstrates more than simply fitting an XGBoost model.  It aims to build builds a predictive maintenance workflow that connects multivariate sensor data, time-series feature engineering, grouped validation, regression, classification, explainability, anomaly detection, and cost-sensitive decision-making.

The most important practical lesson is that predictive maintenance is not only a question of maximizing a model metric.  A useful system must identify deterioration early enough to support action, control the cost of false alarms and missed failures, explain which signals are influencing predictions, and continue to perform on completely unseen equipment.

## Results/Output File

- **`rul_regression_model.joblib`** — final fitted RUL regression pipeline
- **`failure_classifier.joblib`** — final fitted failure-risk classifier
- **`model_config.json`** — feature list, failure window, selected threshold, cost assumptions, and random seed
- **`nasa_test_rul_predictions.csv`** — actual versus predicted RUL for NASA test engines
- **`nasa_test_failure_predictions.csv`** — final failure probabilities and classifications
- **`final_metrics_summary.json`** — compact summary of final regression and classification performance

---