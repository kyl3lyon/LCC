# Launch Commit Criteria Prediction Model
Launch Commit Criteria for Space Flight

### Overview
This model predicts the probabilities of both weather-related Launch Commit Criteria (LCC) violations and launch attempts based on historical launch statistics and weather data. It aims to support launch scheduling decisions by providing insights into potential weather risks, aiding in the anticipation of LCC violations, and ultimately estimating the likelihood of a launch attempt.

### Model Details
Authors: Kyle Lyon, `kyle@algobyte.io`

### Basic Information
* **Model date**: March, 2024
* **Model version**: 1.0
* **License**: Apache 2.0

### Workflow
The workflow begins by loading and preprocessing historical launch statistics, launch forecasts, and hourly weather data. Feature engineering involves calculating average weather conditions during launch windows and joining this data with launch statistics and forecasts. The preprocessed data is then split into training and test sets. Various models, such as Logistic Regression or Random Forest, are trained and evaluated using metrics like accuracy, balanced accuracy, ROC AUC, and F1 score. The evaluation results are summarized in a Markdown table for comparison and analysis.

## Intended Use
* **Primary intended uses**: This model is designed to assist the United States Space Force (USSF) in proactively identifying potential LCC violations and making informed launch scheduling decisions.
* **Primary intended users**: Personnel affiliated with AlgoByte LLC or the USSF, including launch weather officers, mission planners, and decision-makers.
* **Out of scope use-cases**: The models presented in this repository are strictly for research and demonstration purposes. They should not be solely relied upon to determine real-world space launch weather viability or LCC violation likelihood. Operational launch decisions should always be based on comprehensive weather assessments and established protocols.

## Training Data
* **Data Dictionary**:

| Name                  | Modeling Role | Measurement Level | Description                                                                                                                                                   |
|-----------------------|---------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| launched              | Target        | Binary            | Indicates whether a launch has occurred.                                                                                                                      |
| visibility            | Predictor     | Interval          | Average visibility in meters. Maximum value is 10km.                                                                                                          |
| dew_point             | Predictor     | Ratio             | Atmospheric temperature below which water droplets begin to condense and dew can form. Units: Kelvin.                                                         |
| feels_like            | Predictor     | Interval          | This temperature parameter accounts for the human perception of weather. Units: Kelvin.                                                                       |
| temp_min              | Predictor     | Interval          | Minimum temperature at the moment. This can deviate for large cities and megalopolises geographically expanded. Units: Kelvin.                                |
| temp_max              | Predictor     | Interval          | Maximum temperature at the moment. This can deviate for large cities and megalopolises geographically expanded. Units: Kelvin.                                |
| pressure              | Predictor     | Ratio             | Atmospheric pressure at sea level. Units: hPa.                                                                                                                |
| humidity              | Predictor     | Ratio             | Humidity percentage.                                                                                                                                          |
| wind_speed            | Predictor     | Ratio             | Wind speed. Units: meter/sec.                                                                                                                                 |
| wind_deg              | Predictor     | Ordinal           | Wind direction in degrees (meteorological).                                                                                                                   |
| clouds_all            | Predictor     | Ratio             | Cloudiness percentage.                                                                                                                                        |
| rain_1h               | Predictor     | Ratio             | Rain volume for the last hour. Units: mm.                                                                                                                     |
| rain_3h               | Predictor     | Ratio             | Rain volume for the last 3 hours. Units: mm.                                                                                                                  |
| weather_main          | Predictor     | Nominal           | Group of weather parameters (e.g., Rain, Snow, Extreme).                                                                                                      |
| weather_description   | Predictor     | Nominal           | Detailed weather condition within the group.                                                                                                                  |
| weather_icon          | Predictor     | Nominal           | Identifier for the icon representing the weather condition.                                                                                                   |

* **Source of training data**: Eastern Range 5 SLS Launch Stats [[Source]](https://drive.google.com/drive/folders/1IZolgMb5Rgst-68dKOf-PRnZCZpjMI1j?usp=sharing)
* **How training data was divided into training and validation data**: 80% Training, 20% Validation
* **Number of rows in training and validation data**:
    * Training rows: 186, columns=16
    * Validation rows: 47, columns=16

## Test Data
* **Source of test data**: Eastern Range 5 SLS Launch Stats [[Source]](https://drive.google.com/drive/folders/1IZolgMb5Rgst-68dKOf-PRnZCZpjMI1j?usp=sharing)
* **Number of rows in test data**: 233
* **State any differences in columns between training and test data**: All the columns are as the same as the training & validation data.

## Model Details
* **Columns used as inputs in the final model**:
`visibility`, `dew_point`, `feels_like`, `temp_min`, `temp_max`,
`pressure`, `humidity`, `wind_speed`, `wind_deg`, `clouds_all`,
`rain_1h`, `rain_3h`, `weather_main`, `weather_description`,
`weather_icon`

* **Column(s) used as target(s) in the final model**: `launched`
* **Type of models**:
    * Logistic Regression
    * Gradient Boosting
    * SVC
* **Software used to implement the model**: Python 3.10.13, [Scikit-learn](https://github.com/scikit-learn/scikit-learn) v1.4.1.post1.

## Quantitative Analysis
| Model                          | Accuracy  | Balanced Accuracy | ROC AUC   | F1 Score  | Time Taken |
|:-------------------------------|----------:|------------------:|----------:|----------:|-----------:|
| Logistic Regression            |  0.702128 |          0.541126 |  0.619048 |  0.815789 |   0.104826 |
| Random Forest                  |  0.765957 |          0.648268 |  0.619048 |  0.849315 |   0.161093 |
| Gradient Boosting              |  0.680851 |          0.587662 |  0.632035 |  0.782609 |   0.153263 |
| SVM                            |  0.680851 |          0.484848 |  0.629870 |  0.810127 |   0.035027 |

## Ethical Considerations
* Coming later