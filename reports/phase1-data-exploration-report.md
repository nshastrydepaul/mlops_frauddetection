# Fraud Anomaly Classification - Data Exploration Report

## Overview 
 
The dataset contains all the credit card transactions. The overall fraud is very low around 1%, this means the vast majority of transactions are legitimate. This imbalance to keep in mind when building the models.

---

## Spending Patterns

Fraudulent transactions tends to involve higher amounts than legitimate ones. While most legitimate transactions are small. Fraud transactions evenly spread across wider range of amounts including a notable presence in mid to high value transactions.

Merchant category plays a very important role. Online shopping and travel related categories show the highest fraud rates, while in-person categories like gas station show much lower rates. This suggests fraudster prefer card-not-present environment where physical checks are absent.

Looking at customer's average spend over the last 1,7, and 30 days, fraud transactions consistently comes from customer with higher recent average spend. The 1-day and 7-day window shows that fraudsters tend to spend quickly before the card is blocked.

---

## Transaction Timing

Transaction activity remains relatively stable throughout the day for both legitimate and fraud transaction. However, fraudulent transactions appear slightly more concerted during the late night and early morning hours (12am - 6am), which may indicate that fraudsters prefer time when cardholders are less likely to monitor their account activities,

Day of the week has very little impact on fraud rate it stays content across Monday to Sunday with only minor variation on weekends.

---

## Transaction Frequency 

Customers linked to fraud tend to spend more frequently in a shorter window. Looking at the transaction counts over the past 1 and 7 days, fraud cases shows higher count. This pattern of many transactions in a short period is a classis fraud behavior. The 30 days window is less revealing because the recent spike gets diluted over time.

---

## Location 

Fraud is spread across the entire country with no strong geographical concentration. Larger states likes California, Texas and New York show the highest fraud counts, this is beacuse of the large population and higher transaction volume.

The most informative location signal is the distance between the merchant and customer at the time of transaction, Fraud transaction trend to happen at merchants that are farther away from the customers home address than legitimate ones this is a string indication that something is off. 

---

## Customer Demographics

Most fraud victims fall in the 45-65+ age range. Older customers face a higher fraud rate compared to younger ones. Customers under 25 have the lowest fraud rates across all the ages groups. This likely reflects higher account balances in older customers and potentially less vigilance around monitoring statements.

## Conclusion 

Across all the analysis few features stand out as the most useful signals for detecting fraud:
- The distance between the customer and merchant
- The customer's average spend in past 1 and 7 days
- The number of transaction in past 1 and 7 days
- The transaction amount 
- The merchant category 
- The hour of the transaction