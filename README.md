# IRS-Delta-Hedging

In this project I'm using Reinforcement Learning to solve the following problem:

Context:
Our goal is to optimize an Interest-Rate Swap (IRS) desk delta-neutral book management. To do so, let’s represent an IRS book via its delta curve, i.e. its value
sensitivity with regard to each maturity of the market interest rates curve. To simplify, we limit ourselves 10 maturities, the yearly maturities from 1 year to 10 years.


A market-maker delta-neutral management general objective is to keep all its deltas as close to 0 as possible. Nevertheless, as continuously hedging his deltas all day 
long would generate unbearable transaction costs, the market-maker is only asked to keep his deltas within a series a narrow ranges, called delta limits.

Hence the market maker may take positions, even tiny ones, and generate this way a little bit of profit along the year, if he’s able to correctly anticipate market movements.
This model to build has this objective: Profit & Loss (P&L) optimization during a given period, for example one month or one year.

Data:
To start with, data are daily historical data of Euro IRS rates at closing time during the past X years. If necessary, we will later concatenate similar data, synthetic or historical.
Each of the 10 deltas may take positive or negative values, expressed in Euros by basis points. A basis point is 0,01 %. Let’s assume that all 10 deltas are initially set 
to 0.


Delta limits are defines by bucket, I,e, by group of consecutive maturities. Each bucket delta must be constantly kept, in absolute value, below its limit. A bucket delta 
is the sum of its maturities deltas. It is an arithmetic sum, meaning, without taking absolute values. Therefore a certain maturity delta may be balanced by another maturity delta within the same bucket.


Here are the limits we are going to use:
-	1- to 2-year bucket: 10 000 euros
-	3- to 5-year bucket: 10 000 euros
-	6- to 7- year bucket: 15 000 euros
-	8- to 10-year bucket: 20 000 euros
-	And total, i.e. 1- to 10-year bucket: 25 000 euros

To simplify, let’s represent delta hedge transaction just as a modification of this delta value. To simplify even more, let’s assume that the transaction cost associated with this delta modification can be expressed as follows:
 (0,05 bps) x (delta variation) x (delta maturity, in years)

Fonction to optimize
Initial P&L is set to 0. It is incremented every day by the sum of 2 elements:
-	Mark-T-Market daily change, i.e. the sum, along each maturity, of the multiplication of this maturity rate daily variation by this maturity delta.
-	Transaction costs due to delta adjustments at the beginning of that day, as induced by our Reinforcement Learning model.


To build this model, I will work with these Libraries:
    OpenAi Gym, Numpy, Pandas and Stable Baselines.
