# Time-Series-Analysis
## Abstract
Air pollution in Delhi is a critical environmental issue with significant health and economic impacts. A report by the Air Quality Life Index (AQLI) indicates that air pollution in Delhi is shortening lives by nearly 10 years, underscoring the severe health impacts of sustained high pollution levels. In recent years, CO levels have shown periodic spikes, particularly during winter months when atmospheric conditions prevent dispersion. NO2 concentrations from vehicles, power plants, and industrial activity consistently exceed permissible limits, contributing to respiratory problems. Although SO2 levels are relatively lower, they remain a concern due to industrial emissions and fossil fuel combustion. PM2.5, the most dangerous pollutant due to its ability to penetrate deep into the lungs, frequently exceeds WHO guidelines, especially during winter months due to stubble burning in neighbouring states, vehicular pollution, and construction activities. A 2023 study revealed that one in three children in Delhi suffers from asthma due to high pollution levels. Continuous monitoring and effective pollution control strategies are essential to mitigate these impacts. This highlights the need for comprehensive, long-term measures to improve air quality and protect public health in Delhi. Accurate forecasting models, such as ARIMA, ARIMAX, ARMA, and LSTM, are essential for effective monitoring and mitigation efforts.

## Keywords: 
Air Pollution, Time Series Analysis, Respiratory Diseases, NO₂, CO₂, Predictive Modelling, Environmental Forecasting

## Introduction:
Air pollution in India is a critical environmental issue, with the country experiencing some of the highest pollution levels globally. According to the 'World Air Quality Report, 2023' by the Swiss organization IQAir, twenty-six of the world's 30 most polluted cities are in India, with Delhi ranked as the third most polluted city globally. The situation in Delhi is particularly alarming; in December 2023, the city recorded its worst air quality in the last five years, as reported by the Ministry of Environment, Forest and Climate Change. This severe air pollution is a result of various factors including vehicular emissions, industrial activities, and seasonal stubble burning, which cause dangerous spikes in pollution levels, especially during winter.

Several key aspects contribute to air pollution in India. Vehicular emissions are a major source of pollutants such as nitrogen oxides (NOx), carbon monoxide (CO), and particulate matter (PM), which degrade air quality, particularly in urban areas. Industrial activities also significantly contribute to pollution, with emissions of sulphur dioxide (SO2), nitrogen oxides (NOx), and other harmful pollutants from power plants and manufacturing industries. The burning of biomass and crop residue, especially in states like Punjab and Haryana, adds to severe seasonal air pollution, and natural causes such as volcanic eruptions, natural radioactivity, and wind erosion further exacerbate the issue. The extensive use of fossil fuels for energy production and transportation is another major contributor to poor air quality.
The impacts of this persistent pollution crisis on society are profound. Public health is severely affected, with air pollution contributing to respiratory diseases, cardiovascular conditions, and reduced life expectancy. The World Health Organization (WHO) estimates that air pollution causes approximately 4.2 million deaths per year globally, with a significant portion of these occurring in India. The economic consequences are also substantial, as poor air quality hampers productivity and increases healthcare costs. Additionally, environmental impacts include damage to ecosystems, reduced agricultural yields, and the deterioration of natural and built environments.

Many prediction models forecast air pollutants, but more accurate models are needed for future estimates. Time Series analysis predicts future values using past data, with the Auto Regressive Integrated Moving Average (ARIMA) model being widely used. ARIMA relies on historical data, making it efficient and robust for short-run forecasting. Other models include Auto Regressive Moving Average (ARMA), ARIMA with Explanatory Variable (ARIMAX), Seasonal ARIMA (SARIMA), and Long Short-Term Memory (LSTM). ARMA combines autoregression and moving averages, ARIMAX includes external variables, SARIMA accounts for seasonality, and LSTM, a recurrent neural network, handles complex, non-linear time series forecasting. These models vary in complexity and accuracy for predicting future pollutant levels.




## 2 Model Design and Problem Evaluation
### 2.1 DATA SET
This dataset contains air quality data from the national capital of Delhi, India. It includes information on air pollution levels, including Nitrogen Oxide (NO), nitrogen dioxide (NO2), sulphur dioxide (SO2) and carbon monoxide (CO). The data was collected from monitoring stations located in various areas of Delhi between January 31, 2018, and August 24, 2023. 
### 2.2 Data Cleaning
Data cleaning involves multiple steps to prepare a dataset for analysis or modelling. Initially, missing values are dealt by either eliminating or inserting them using methods like mean, median, mode. Duplicates are identified and removed to prevent skewed results, while inconsistencies in data formats, such as date formats or categorical labels, are corrected. Outliers are detected and managed through methods like IQR or Z-score, ensuring they don't distort the analysis. Numerical data may be normalized or scaled, particularly for algorithms sensitive to magnitude, and categorical variables are encoded into numerical formats using techniques like one-hot encoding or label encoding.

### ARMA:
The Auto Regressive Moving Average (ARMA) model is a mathematical model designed to predict future values of a time series based on its past data, specifically for data that is already stationary. Stationary time series is a crucial assumption for the ARMA model, meaning the time series should have a constant mean and variance, and its autocorrelation should not depend on time. To ensure a time series is stationary, two primary tests are commonly used: Rolling Statistics (RS) and the Augmented Dickey-Fuller Test (ADFT). The RS test involves plotting the moving average or moving variance of the time series and observing if these metrics vary over time; if they do, the model is non-stationary. Otherwise, it is stationary. The ADFT provides a more formal statistical assessment of stationery, where the null hypothesis is that the time series is non-stationary. A low p-value and critical values greater than the test statistic suggest the model is stationary.
Once stationary time series is confirmed, the ARMA model can be applied. The ARMA model combines two components: the Auto Regressive (AR) and the Moving Average (MA) components. The AR component assumes the current value of the time series is linearly dependent on its previous values. It is represented by AR(p), where p denotes the number of lagged observations (lags) used. The equation for the AR component is:
Yt=ϕ1Yt−1+ϕ2Yt−2+⋯+ ϕpYt−p+ϵt

Here, Yt is the current value of the time series, ϕi  are the coefficients, Yt-i are the past values (lags), and ϵt  is the white noise error term. The MA component models the relationship between the current value of the time series and past error terms (residuals), assuming the current value is influenced by past prediction errors. It is represented by MA(q), where q denotes the number of lagged forecast errors in the prediction equation. The equation for the MA component is:
 ϵt=θ1ϵt−1+θ2ϵt−2+…+ θqϵt−q+ ηt
 Here ϵt is the current error term, θi are the coefficients, ϵt−i  are past error terms (lags), and ηt is the white noise
When combined, the ARMA model is expressed as:
Yt=ϕ1Yt−1+ϕ2Yt−2+⋯+ ϕpYt−p+ ηt  - (θ1ηt-1 + θ2ηt-2 +…+ θqηt-q)
where the first part (ϕ1Yt−1+ϕ2Yt−2+⋯+ ϕpYt−p) represents the autoregressive component and 
the second part (- (θ1ηt-1 + θ2ηt-2 +…+ θqηt-q)) represents the moving average component.	

#### Algorithm for prediction using ARMA :
Output: Forecasted values of the time series
Input: Historical time series data
while true do
    Clean the data set;
    Divide the data set randomly for training (70%) and testing (30%);
    Check stationarity of the training data set using Rolling Statistics (RS) or Augmented Dickey-Fuller Test (ADFT);
    
    if training model is stationary then
        Select parameters p, q;
        Train the ARMA model on the training set using the selected p, q parameters;
        Predict values for the training set;
        
        Evaluate the model using training and testing data sets;
        Calculate errors (e.g., Mean Absolute Error, Root Mean Squared Error);
        
        if model performance is satisfactory then
            Generate forecasted values for the future time points;
            Generate graphs and accuracy reports;
            break;
        else
            Adjust parameters p, q and retrain the model;
        end if
    else
        Apply transformations to make the data stationary;
    end if
end while

The p and q values are determined from the Partial Correlation Function (PACF) and Auto Correlation Function (ACF) graphs, respectively. The PACF graph's value of p and ACF graph's value of q are determined by the point when the graph first meets the zero line.

### ARIMA:
The ARIMA model is an extension of the ARMA model that includes an additional step of differencing to handle non-stationary time series data. ARIMA is particularly useful for forecasting time series that exhibit trends or other forms of non-stationarity. The model is characterized by three parameters: p, d, and q , where p represents the autoregressive component, d represents the number of differencing operations required to make the time series stationary, and q represents the moving average component. It is also applied to stationary time series. The ARIMA model is a combination of three components: the Auto Regressive (AR) component, the Integrated (I) component, and the Moving Average (MA) component. The AR and MA components are the same as described in the ARMA model above.
The Integrated (I) component involves differencing the time series to make it stationary, which removes trends or seasonality that are not constant over time. Differencing is calculated as:
 Yt = Xt− Xt-1
The ARIMA model combines these three components and is typically expressed as ARIMA(p, d, q):
ϕ(B) (1−B)d  Xt= θ(B)ϵt
Here ϕ(B) is the polynomial function representing the AR component, θ(B) represents the MA component, and (1−B)d   represents the differencing operator.
The assumptions of the ARIMA model are that the time series data is (or can be made) stationary through differencing, and the residuals of the model are uncorrelated (white noise) and normally distributed.
#### Algorithm: Predicting Values Using ARIMA Model
Output: Forecasted values of the time series 
Input: Historical time series data
while true do
    Clean the data set;
    Divide the data set randomly for training (70%) and testing (30%);
    Check stationarity of the training data set using Rolling Statistics (RS) or Augmented Dickey-Fuller Test (ADFT);
    
    while training model is not stationary do
        Apply differencing to make the data stationary;
    end while
    
    Select parameters p, d, q;
    Train the ARIMA model on the training set using the selected parameters;
    
    Predict values for the training set;
    Evaluate the model using training and testing data sets;
    Calculate errors (e.g., Mean Absolute Error, Root Mean Squared Error);
    
    if model performance is satisfactory then
        Generate forecasted values for the future time points;
        Generate graphs and accuracy reports;
        break;
    else
        Adjust parameters p, d, q and retrain the model;
    end if
end while

### ARIMAX:
The ARIMAX model is an extension of the ARIMA model that incorporates exogenous variables (external predictors) to improve forecasting accuracy. The exogenous variables provide additional information that can help explain the variability in the time series data. The ARIMAX model is characterized by four parameters: p, d, q, and X, where X represents the exogenous variables.
The ARIMAX model is typically expressed as:
Yt = ϕ(B)(1−B)dXt + βXt + θ(B)ϵt

Here, ϕ(B) is the polynomial function representing the AR component, (1−B)d represents the differencing operator, βXt represents the exogenous variables, and θ(B) represents the MA component.
The assumptions of the ARIMAX model are similar to those of the ARIMA model, with the additional assumption that the exogenous variables are not correlated with the error terms.
#### Algorithm: Predicting Values Using ARIMAX Model
Output: Forecasted values of the time series
Input: Historical time series data and exogenous variables
while true do
    Clean the data set;
    Divide the data set randomly for training (70%) and testing (30%);
    Check stationarity of the training data set using Rolling Statistics (RS) or Augmented Dickey-Fuller Test (ADFT);
    
    while training model is not stationary do
        Apply differencing to make the data stationary;
    end while
    
    Select parameters p, d, q, and X;
    Train the ARIMAX model on the training set using the selected parameters;
    
    Predict values for the training set;
    Evaluate the model using training and testing data sets;
    Calculate errors (e.g., Mean Absolute Error, Root Mean Squared Error);
    
    if model performance is satisfactory then
        Generate forecasted values for the future time points;
        Generate graphs and accuracy reports;
        break;
    else
        Adjust parameters p, d, q, and  X and retrain the model;
    end if
end while

### SARIMA:
The SARIMA (Seasonal ARIMA) model is an extension of the ARIMA model that explicitly handles seasonality in time series data. It incorporates both non-seasonal and seasonal components to better capture the patterns in the data. The SARIMA model is characterized by six parameters: p, d, q, P, D, and Q, where P, D, and Q represent the seasonal autoregressive, differencing, and moving average components, respectively. Additionally, s represents the length of the seasonal cycle.
The SARIMA model is typically expressed as:
ϕ(B) Φ(Bs) (1 − B)d (1 − Bs)D Xt= θ(B) Θ(Bs) ϵt

Here, ϕ(B) and Φ(Bs) are the polynomial functions representing the non-seasonal and seasonal AR components, respectively, (1−B)d  and (1−Bs)D represent the non-seasonal and seasonal differencing operators, and θ(B) and Θ(Bs) represent the non-seasonal and seasonal MA components.
The assumptions of the SARIMA model are similar to those of the ARIMA model, with the additional assumption that the seasonal components are appropriately modeled.

#### Algorithm: Predicting Values Using SARIMA Model
Output: Forecasted values of the time series
Input: Historical time series data

while true do
    Clean the data set;
    Divide the data set randomly for training (70%) and testing (30%);
    Check stationarity of the training data set using Rolling Statistics (RS) or Augmented Dickey-Fuller Test (ADFT);
    
    while training model is not stationary do
        Apply differencing to make the data stationary;
    end while
    
    Select parameters p, d, q, P, D, Q, and s;
    Train the ARIMAX model on the training set using the selected parameters;
    
    Predict values for the training set;
    Evaluate the model using training and testing data sets;
    Calculate errors (e.g., Mean Absolute Error, Root Mean Squared Error);
    
    if model performance is satisfactory then
        Generate forecasted values for the future time points;
        Generate graphs and accuracy reports;
        break;
    else
        Adjust parameters p, d, q, P, D, Q, and s and retrain the model;
    end if
end while

### LSTM:
The Long Short-Term Memory (LSTM) model is a type of recurrent neural network (RNN) designed to capture long-term dependencies and patterns in time series data. Unlike traditional RNNs, LSTMs are equipped with a memory cell that helps retain information over long periods, making them particularly useful for time series forecasting where temporal dependencies are crucial.
The LSTM model consists of three main gates: the input gate, the forget gate, and the output gate. These gates regulate the flow of information into and out of the memory cell. The input gate determines how much of the new information should be added to the memory cell, the forget gate controls how much of the existing information should be discarded, and the output gate decides how much of the information in the memory cell should be output to the next layer.

#### Algorithm: Predicting Values Using LSTM Model
Output: Forecasted values of the time series
Input: Historical time series data

while true do
    Clean the data set;
    Normalize or standardize the data if necessary;
    Divide the data set randomly for training (70%) and testing (30%);
    
    Prepare the data for LSTM input:
          Reshape the data into a suitable format for LSTM (e.g., sequences of   fixed length); 
    Define the LSTM model architecture:
     - Input layer 
     - LSTM layers (with chosen number of units and activation functions) 
     - Dense output layer
   Train the LSTM model on the training set using the chosen architecture and hyper parameters;
   Predict values for the training set;
   Evaluate the model using training and testing data sets;
   Calculate errors (e.g., Mean Absolute Error, Root Mean Squared Error);
   if model performance is satisfactory then
        Generate forecasted values for the future time points;
        Generate graphs and accuracy reports;
        break;
   else
        Adjust model architecture and hyper parameters and retrain the model;
   end if
end while




