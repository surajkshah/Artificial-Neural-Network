# ********************************* Artificial Neural Network_Coding Assignment_1 *******************************
import random as random
import pandas as pd
import numpy as np


# User Input=====================================================================================================
L = 8 #int(input("Enter the number of input neurons: "))
M = 8 #int(input("Enter the number of hidden neurons: "))
N = 1 #int(input("Enter the number of output neurons: "))

# Given Data=====================================================================================================
a_1 = 1  # Transfer Function constant of Hidden Layer
a_2 = 1  # Transfer Function constant of Output Layer
eta = 0.7  # Learning Rate
alpha = 0.3 # Momentum Term
tolerance = 0.0001009 # Error Tolerance

# Training Data (Reading from excel file)========================================================================
Pattern = np.asarray(pd.read_excel('Training_Pattern.xlsx', usecols=[*range(0, L + N)]))
Pattern = Pattern.astype(float)

Order_P = np.shape(Pattern)  # Finding order of Pattern Matrix

# Randomly generating Connection weight==========================================================================
# Between Input and Hidden Layer [V]
V = []
for i in range(L + 1):
    temp = []
    for j in range(M):
        temp.append(random.uniform(-1, 1))
    V.append(temp)

# Between Hidden and Output Layer [W]
W = []
for i in range(M + 1):
    temp = []
    for j in range(N):
       temp.append(random.uniform(-1, 1))
    W.append(temp)

# Normalization of Data=========================================================================================
# Initialization of maximum and minimum value
Xp_max = [0] * Order_P[0]
Xp_min = [0] * Order_P[0]

# Finding maximum and minimum value in each pattern
for i in range(Order_P[0]):
    Xp_max[i] = np.max(Pattern[i, :])
    Xp_min[i] = np.min(Pattern[i, :])

# Normalization of Pattern
for i in range(Order_P[0]):
    for j in range(Order_P[1]):
        Pattern[i][j] = 0.1 + 0.8 * ((Pattern[i][j] - Xp_min[i]) / (Xp_max[i] - Xp_min[i]))

# Input and Target Output Matrix
I = Pattern[:, 0:L]  # Input Pattern Matrix
Order_I = np.shape(I)
T = Pattern[:, L:L + N]  # Target Output Value
Order_T = np.shape(T)

# Initialization of all Parameters===============================================================================
Input_H = [0] * M    # Input of Hidden Neurons
Output_H = [0] * M   # Output of Hidden Neurons
Input_O = [0] * N    # Input of Output Neurons
Output_O = [0] * N   # Output of Output Neurons

delta_W = [[0] * N for i in range(M + 1)]
delta_V = [[0] * M for i in range(L + 1)]

delta_W_old = [[0] * N for i in range(M + 1)]
delta_V_old = [[0] * M for i in range(L + 1)]

MSE = 1     # For while loop

# Transfer Function===============================================================================================
def log_H(x):
    return 1 / (1 + np.exp(-x * a_1))


def log_O(x):
    return 1 / (1 + np.exp(-x * a_2))


def tan_H(x):
    return (np.exp(x * a_1) - np.exp(-x * a_1)) / (np.exp(x * a_1) + np.exp(-x * a_1))


def tan_O(x):
    return (np.exp(x * a_2) - np.exp(-x * a_2)) / (np.exp(x * a_2) + np.exp(-x * a_2))

# Main Program=====================================================================================================
iteration = 0

while MSE > tolerance:

    # Storing previous iteration delta value in other variable
    for i in range(N):
        for j in range(M+1):
            delta_W_old[j][i] = delta_W[j][i]

    for i in range(M):
        for j in range(L+1):
            delta_V_old[j][i] = delta_V[j][i]

    # Initialization to zero for next iteration
    delta_W = [[0] * N for i in range(M + 1)]
    delta_V = [[0] * M for i in range(L + 1)]
    MSE = 0

    for p in range(Order_P[0]):

        Input = I[p].tolist()
        Target = T[p].tolist()
        Input.insert(0, 1)       # Inserting bias value 1 in first column



        # Forward Pass Calculations================================
        # Calculation of output of hidden neuron
        for i in range(M):
            temp = 0
            for j in range(L+1):
                temp += V[j][i] * Input[j]
            Input_H[i] = temp
            Output_H[i] = log_H(Input_H[i])

        Output_H.insert(0, 1)     # Inserting bias value 1 in first column

        # Calculation of output of output neuron
        for i in range(N):
            temp = 0
            for j in range(M+1):
                temp += Output_H[j] * W[j][i]
            Input_O[i] = temp
            Output_O[i] = log_O(Input_O[i])

        # Calculation of Error
        error = 0
        for i in range(N):
            error += 0.5*pow((Target[i] - Output_O[i]), 2)
        MSE += error/N

        # Calculation of delta_W====================================
        for i in range(M+1):
            for j in range(N):
                delta_W[i][j] += (Target[j] - Output_O[j]) * a_2 * Output_O[j] * (1-Output_O[j]) * Output_H[i]    #log sigmoid
                #delta_W[i][j] += (Target[j] - Output_O[j]) * a_2 * (1 - pow(Output_O[j], 2)) * Output_H[i]       #tan sigmoid

        Output_H.pop(0)             # Removing first column belonging to bias

        # Calculation of delta_V====================================
        for i in range(L+1):
            for j in range(M):
                temp = 0
                for k in range(N):
                    temp += (Target[k] - Output_O[k]) * a_2 * Output_O[k] * (1-Output_O[k]) * W[j+1][k]         #log sigmoid
                    #temp += (Target[k] - Output_O[k]) * a_2 * (1 - pow(Output_O[k], 2)) * W[j + 1][k]          #tan sigmoid
                temp = temp/N
                delta_V[i][j] += temp * a_1 * Output_H[j] * (1-Output_H[j]) * Input[i]

        Input.pop(0)                      # Removing first column belonging to bias

    # Mean Square Error Calculation=================================
    MSE = MSE/(p+1)

    # Updating Connection Weights===================================
    # Updating W

    for i in range(N):
        for j in range(M+1):
            delta_W[j][i] = delta_W[j][i]/(p+1)

    for i in range(M):
        for j in range(L+1):
            delta_V[j][i] = delta_V[j][i]/(p+1)


    for i in range(N):
        for j in range(M+1):
            W[j][i] += eta * delta_W[j][i] + alpha * delta_W_old[j][i]

    # Updating V
    for i in range(M):
        for j in range(L+1):
            V[j][i] += eta * delta_V[j][i] + alpha * delta_V_old[j][i]

    iteration += 1
    print(str(iteration) + "\t" + str(MSE))
# End of Main Program============================================================================================
print("\nValue of W is: \n")
print(W)
print("\nValue of V is: \n")
print(V)
print("\n\n")

# Testing of Model===============================================================================================
a = int(input("Enter pattern number from which you want to start your testing: "))
b = a + 50
I = Pattern[a:b, 0:L]  # Testing Pattern Matrix
Order_I = np.shape(I)
T = Pattern[a:b, L:L + N]  # Target Output Value of Testing Pattern
Order_T = np.shape(T)

Input_H = [0] * M    # Input of Hidden Neurons
Output_H = [0] * M   # Output of Hidden Neurons
Input_O = [0] * N    # Input of Output Neurons
Output_O = [0] * N   # Output of Output Neurons
Result_O = [0] * N   # Result of Output Neurons after de-normalization
Mean_Error = 0
# Forward Pass Calculation========================
for p in range(Order_I[0]):

    Input = I[p].tolist()
    Input.insert(0, 1)          # Inserting bias value 1 in first column

    # Calculation of output of hidden neuron
    for i in range(M):
        temp = 0
        for j in range(L + 1):
           temp += Input[j] * V[j][i]
        Input_H[i] = temp

        Output_H[i] = log_H(Input_H[i])

    Output_H.insert(0, 1)           # Inserting bias value 1 in first column
    Error = 0
    # Calculation of output of output neuron
    for i in range(N):
        temp = 0
        for j in range(M + 1):
           temp += Output_H[j] * W[j][i]
        Input_O[i] = temp
        Output_O[i] = log_O(Input_O[i])

        # De-normalization of Output
        Result_O[i] = ((Output_O[i] - 0.1) * (Xp_max[a+p] - Xp_min[a+p]) / 0.8) + Xp_min[a+p]
        T[p] = ((T[p] - 0.1) * (Xp_max[a+p] - Xp_min[a+p]) / 0.8) + Xp_min[a+p]
        Error = abs(T[p] - Result_O[i])
        Mean_Error += Error
        print("Output of Network for Pattern " + str(a+p) + " is: " + str(Result_O) + "  Error in prediction: " + str(Error))

Mean_Error = Mean_Error/(p+1)
print("\n Mean Prediction Error: " + str(Mean_Error))