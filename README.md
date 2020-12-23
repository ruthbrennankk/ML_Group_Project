# ML_Group_Project
##About
This is the repo for our 4th Year Machine Learning group project.  We tried to train regression models to predict the price of second hand cars being sold on Carzone.ie.
##What is in the repo
__Data Gathering__ has the scraping file that was used to scrape the data that was then used to train and test the models.  It also has a _Data_ folder containing the main three csv files used for this project with the final one being _g_cars_final.csv_.

__Models__ contains the files used for each of the models to cross validate the hyperparameters. The assist folder then contains a file _inputPlots.py_ that we used to initially visulaise our data. _reading.py_ read the csv into a DataFrame and formatted some of the data.  _testingModels.py_ performed the final tests on all the final models and produced metrics and plots.  

__Plots__ contains the final plots produced by our program.  _InputPlots_ has the initial visualisation plots of our data. _PredictionPlots_ has the final prediction plots for each of our models. _crossValPlots_ has the final plots used for the cross validation for each of our models.

 