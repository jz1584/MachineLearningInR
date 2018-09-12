################################################
################################################
#####[11] Time Series Analysis Lecture Code#####
################################################
################################################
library(fpp) #Forecasting principles and practice library.

#Example of a trend in a time series.
plot(ustreas, xlab = "Day", ylab = "US Treasury Bill Contracts")

#Example of a treand and seasonality in a time series.
plot(elec, xlab = "Year", ylab = "Australian Monthly Electricity Production")

#Example of a seasonal and cyclical nature in a time series.
plot(hsales, xlab = "Year", ylab = "Monthly Housing Sales (Millions)")

#Example of an irregular nature in a time series.
plot(diff(dj), xlab = "Day", ylab = "Daily Change in Dow Jones Index")

#Loading the forecast library for time series analysis.
library(forecast)

#Observing the effects of centered moving averages on the Nile dataset.
ylim = c(min(Nile), max(Nile))
plot(Nile, main = "Raw Time Series", ylim = ylim)
plot(ma(Nile, 3), main = "Centered Moving Averages (k = 3)", ylim = ylim)
plot(ma(Nile, 7), main = "Centered Moving Averages (k = 7)", ylim = ylim)
plot(ma(Nile, 15), main = "Centered Moving Averages (k = 15)", ylim = ylim)

plot(Nile, main = "Centered Moving Averages\nNile Data", ylim = ylim)
lines(ma(Nile, 3), col = "red", lwd = 2)
lines(ma(Nile, 7), col = "green", lwd = 2)
lines(ma(Nile, 15), col = "blue", lwd = 2)
legend("bottomleft",
       c("Raw Data", "k = 3", "k = 7", "k = 15"),
       col = c("black", "red", "green", "blue"),
       lwd = c(1, 2, 2, 2))

#Investivating the multiplicative nature of the Airline Passengers dataset.
plot(AirPassengers, main = "Monthly International Airline Passengers")

#Log transforming in order to achieve an additive nature.
log.AirPassengers = log(AirPassengers)
plot(log.AirPassengers, ylab = "Log of Air Passengers",
     main = "Monthly International Airline Passengers\nLog Transformed")

#Applying seasonal decomposition; setting s.window to "period" ensures that
#the seasonal effects will be the same across years.
seasonal.decomposition = stl(log.AirPassengers, s.window = "period")
plot(seasonal.decomposition, main = "Seasonal Decomposition of Airline Data")

#The output provides the overall data, the additive seasonal component, the
#additive trend component, and the irregular (remainder) component. The scales
#of each component are depicted by the gray bars.

#We can manually extract the components of the time series.
seasonal.decomposition$time.series

#Back-transforming the decomposition to be on the original scale of the data.
exp(seasonal.decomposition$time.series)

#Examining the month plot of the Airline Data; seeing the effects of month over
#time, and the average placement for each month.
monthplot(AirPassengers, main = "Month Plot of Airline Data")

#Examining the season plot of the Airline Data; seeing the effects of year over
#time.
seasonplot(AirPassengers, year.labels = TRUE, main = "Season Plot of Airline Data")



################################
#####Fitting an ARIMA Model#####
################################
library(tseries)
plot(Nile, main = "Annual Flow of the Nile River")

#When we smoothed the Nile data before we saw that there might be a trend;
#however, in order to fit a valid ARIMA model, we need to have a stationary
#model. Let's inspect the appropriate number of differences for this series:

ndiffs(Nile) #Estimates the number of differences required to make a given
             #time series stationary; returns d = 1.

dNile = diff(Nile, differences = 1) #Returns lagged and iterated differences;
                                    #default lag and differences are both 1.

#Inspecting the differenced time series for stationality.
plot(dNile, main = "Annual Flow of the Nile River\n1 Difference")

#Conducting the Audmented Dickey-Fuller Test:
adf.test(dNile)

#The p-value for this test is < 0.05, indicating that we would reject the null
#hypothesis that the time series is not stationary; we conclude that the Nile
#River time series, when differenced by 1, is stationary.

#Investigating the ACF and PACF of the detrended Nile data in order to determine
#likely AR and MA levels.
par(mfrow=c(2, 1))
Acf(dNile)
Pacf(dNile)
par(mfrow=c(1,1))

#We see that both the autocorrelation function and the partial autocorrelation
#function seem to drift off. Additionally, there seems to be a significant spike
#at one lag for the autocorrelation, and a significant spike at both one and two
#lags for the partial autocorrelation.

#Try initially fitting an AR(p = 2) and an MA(q = 1). Don't forget that we
#determined that d = 1 earlier.
initial.fit = Arima(Nile, order = c(2, 1, 1))
initial.fit

#We can print out some extra detail with the summary() function that provides
#some predictive accuracy measures; this is a bit superfluous and there are many
#options, but not all of the accuracy measures are useful (e.g., the mean error
#ME and the mean percentage error MPE may not be useful because positive and
#negative errors can cancel out). The root mean squared error RMSE indicates
#the absolute fit of the model to the data and displays how close the observed
#observations are to the model's predictive values; it represents the standard
#deviation of the residuals.
summary(initial.fit)

#For this model, the observations vary around the predicted trend by about 139.7
#units of flow.

#Since the ARIMA model uses maximum likelihood for estimation, the coefficients
#are asymptoticaly normal; thus, we can divide coefficients by their standard
#errors to get the z-statistics and then calculate the associated p-values.
(1 - pnorm(abs(initial.fit$coef)/sqrt(diag(initial.fit$var.coef))))*2

#The AR(2) coefficient appears to not be significant. What happens if we drop
#this extra term?
new.fit = Arima(Nile, order = c(1, 1, 1))
new.fit

summary(new.fit) #The RMSE is similar to the model with the extra term.

#A loose interpretation of the coefficient estimates:
#-The AR(1) term is about .2544 meaning that the series tends to return to the
# mean relatively quickly; the restoring force is quite strong because the
# magnitude of the coefficient is quite small.
#-The MA(1) term is about -.8741 meaning that the series feels some shock across
# consecutive terms in the model; this shock is only felt in neighboring time
# periods.

#The p-values for the parameters in the new model are significant.
(1 - pnorm(abs(new.fit$coef)/sqrt(diag(new.fit$var.coef))))*2

#Both the AIC and BIC for the model with the dropped AR(2) term are lower.
AIC(initial.fit, new.fit)
BIC(initial.fit, new.fit)

#Assessing residual diagnostics:
plot(as.vector(fitted(new.fit)), new.fit$residuals, #Constant variance seems ok.
     main = "Residual Plot")
abline(h = 0, lty = 2)

qqnorm(new.fit$residuals) #Normality of the errors appears to be fine.
qqline(new.fit$residuals)

Acf(new.fit$residuals) #No significant autocorrelations.
Pacf(new.fit$residuals) #No significant partial autocorrelations.

#The Ljung-Box test returns an insignificant p-value, further suggesting that
#the autocorrelations do not differ from 0; this helps us determine that the
#independent errors assumption is not violated.
Box.test(new.fit$residuals, type = "Ljung-Box")

#We can forecast new values by using the forecast() function:
future.values = forecast(new.fit, 3, level = c(60, 80, 95))

#We're given both point estimates and confidence bands represented by the levels
#we specify; this function is nice because it automatically transforms the data
#back onto the original scale!
future.values
plot(future.values)

#What happens if we forecast out for a very long distance?
future.values = forecast(new.fit, 30, level = c(60, 80, 95))
future.values
plot(future.values)