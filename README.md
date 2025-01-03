# Healthcare Cost Prediction
This project is focused on predicting healthcare costs using a regression algorithm. The goal is to build a machine learning model that generalizes well and predicts healthcare expenses within an acceptable error range.
## Dataset
The dataset includes information about various individuals along with their healthcare expenses. The data contains both numerical and categorical features. The target variable is expenses, which represents the healthcare costs.
## Preprocessing Steps
1. **Convert Categorical Data:**                                                                                                               
   All categorical features are encoded into numerical representations.                                                                         
2. **Split the Data:**                                                                                                                  
   80% of the dataset is used as the training set (train_dataset).                                                                                                        
   20% of the dataset is used as the testing set (test_dataset).
3. **Extract Labels:**                                                                                                                                    
   The expenses column is removed from the datasets to create labels: train_labels , test_labels
## Model
A regression model is created and trained using the train_dataset and train_labels. The model is evaluated on the test_dataset to measure its generalization performance.                   
### Evaluation Criteria
The model must achieve a Mean Absolute Error (MAE) of under $3500 when evaluated using the model.evaluate() method. This ensures that the model predicts healthcare costs accurately within a $3500 range.
### Visualization
The final step includes predicting healthcare expenses using the test_dataset and visualizing the results to assess the model's performance.
!(img_alt)[]
# Requirements
Python 3.8+

TensorFlow 2.x

Pandas

NumPy

Matplotlib

Scikit-learn

### Results
!(img_alt)[]                                                                                                                                                    
Training Accuracy: 5215.78                                                                                                                                       
Test Accuracy: 4979.72                                                                                                                                             

Visualization demonstrates the accuracy and generalization capability of the model.

### Contacts
[Linkedin](www.linkedin.com/in/g-kamaleashwar-28a2802ba)                                                                                                                                
[Portfolio](https://kamalesh-portfolio.streamlit.app/)
