# Investigating Causes of Wildfire in the United States
CS 760 Machine Learning (Fall 2020) Project

Authors: Jingqi Duan, Huanran Li, and Olivia Zhao

Wildfire is a major environmental concern that endangering human life, by affecting both physical
and economical environments. Research has investigated the contributing factors that lead to wildfire
occurrence. These factors include climate change, past forest fire behaviors, human activities, and weather
information. In this project, we obtained data from each of the above modality from official sources,
and explored machine learning models including K-nearest neighbor, decision tree, naive bayes and
multinomial logistic regression, to predict wildfire’s cause. We found that K-nearest neighbor was the
best model overall. Our models had good accuracy with the fire caused by lightning, debris burning; but
poor results for other causes, such as arson, fireworks, equipment use. Further investigation implied that
we might not have enough data for those poorly performed causes.

11-Class (lightning, equipment use, smoking, campfire, debris burning, railroad, arson, children, fireworks, powerline, structure):
| Model | Train Accuracy | Test Accuracy |
|---|---|---|
Decision Tree(Best) |  52.6% | 46.0% |
Weighted-Distance KNN (Best) |  51.5% | **47.2%**  |
Top-1 Multinomial Logistic Regression |  32.6% | 32.6% |
Top-2 Multinomial Logistic Regression | 42.2% | 42.3% |

4-Class (lightning, equipment use, debris burning, and arson):
| Model | Train Accuracy | Test Accuracy |
|---|---|---|
Decision Tree(Best) |  49.1% | 45.4% |
Weighted-Distance KNN (Best) | 53.8% | 48.5% |
Top-1 Multinomial Logistic Regression | 32.2% | 32.0% |
Top-2 Multinomial Logistic Regression | 59.4% | **59.2%** |
