# California Housing: Regression, From-Scratch Gradient Descent and Error Maps

A small project that predicts California median house value and maps where the model gets it wrong. It uses Python (scikit-learn, NumPy, Folium) and MATLAB.

## What this project shows

| Skill | What I did |
|---|---|
| Machine learning | Compared Linear Regression with Random Forest on the same train/test split |
| Algorithm development | Wrote gradient descent from scratch in NumPy and checked it against scikit-learn |
| MATLAB | Re-implemented the regression with least squares, tested outlier filtering, made an error map |
| GIS (basic level) | Folium maps of house locations and per-house prediction errors |

## Data

- Source: scikit-learn `fetch_california_housing` (1990 US Census, 20,640 block groups).
- Target: `MedHouseVal`, median house value in units of $100,000.
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.
- For MATLAB, a random 5,000-row sample was exported to `cali.csv` (`random_state=42`).

## Results

| Model | Test MSE | Notes |
|---|---|---|
| Linear Regression (scikit-learn) | 0.53 | 80/20 split |
| Gradient descent from scratch (NumPy) | 0.53 | 1000 steps, learning rate 0.1, standardized features |
| Random Forest (100 trees) | 0.26 | `random_state=42` |
| Least squares in MATLAB, outliers removed | 0.4605 | 5,000-row sample, see caveat below |

The from-scratch result matches scikit-learn's, which checks that my implementation is correct. Random Forest roughly halves the error compared with linear regression.

### Caveat: the numbers are not all comparable

The MATLAB result (0.4605) comes from a different sample (5,000 rows), a different split, and data with outliers removed (`AveOccup < 10` and `AveRooms < 20`). Do not read it as "MATLAB beats Python". It only shows that a few extreme rows affect a linear model.

## Error maps

**Python (Folium).** Each test house is a circle coloured by absolute error: green below 0.3 (about $30k), orange 0.3 to 0.7, red above 0.7. Red points are concentrated in the dense Bay Area and Los Angeles regions, which is also where most of the data is, so this does not by itself show that the model fails there.

**MATLAB.** A scatter of longitude vs latitude coloured by absolute error, with outliers removed:

![MATLAB error map](images/error_map_filtered.png)

## Repository layout

```
.
├── README.md
├── notebooks/
│   └── california_housing.ipynb
├── matlab/
│   └── error_map.m
├── data/
│   └── cali.csv
└── images/
    ├── error_map_filtered.png
    └── folium_error_map.png
```

## How to run

**Python (Google Colab or local)**
```
pip install pandas numpy scikit-learn folium matplotlib seaborn
```
Open the notebook and run all cells. Note: the linear regression split has no fixed seed, so the MSE may differ slightly between runs.

**MATLAB (MATLAB Online works)**
1. Upload `cali.csv` and `error_map.m` to the same folder.
2. Run `error_map.m`. It prints the test MSE and saves `error_map_filtered.png`.

## Limitations

- Linear regression is a weak baseline for this data, and I did not tune the Random Forest.
- The Folium error map uses only 500 randomly sampled test houses.
- The data is from 1990 and values are capped, so it is not a model of today's housing market.
- My GIS work is basic: plotting and visual error analysis, with no spatial statistics or projections yet.

## Possible next steps

- Add spatial features such as distance to the coast or to major cities.
- Test spatial cross-validation, because neighbouring houses leak information between train and test.
- Compare gradient boosting with Random Forest.
- Quantify spatial autocorrelation of the errors (for example, Moran's I).
