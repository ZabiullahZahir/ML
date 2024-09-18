import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import streamlit as st
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'MLdataset.csv'  # Adjust the file path as necessary
df = pd.read_csv(file_path)

# Select the first 80000 rows and relevant columns
df = df.iloc[0:80000]
keep_columns = ['Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc']
df = df[keep_columns]

# Replace null values in selected columns
for col in keep_columns:
    if col in df.columns:
        df[col] = df[col].fillna('N/A')

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=keep_columns)

# Streamlit Sidebar for Adjusting Support and Confidence
st.sidebar.header('Association Rule Parameters')
min_support = st.sidebar.slider('Minimum Support', 0.01, 0.5, 0.1, 0.01)
min_confidence = st.sidebar.slider('Minimum Confidence', 0.1, 1.0, 0.7, 0.05)

# Apply FP-Growth algorithm with the user-selected support threshold
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Extract association rules with the user-selected confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules = rules[rules['lift'] > 1.2]  # Only keep rules with lift > 1.2

# Display the relevant rule columns
limitedCol = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
st.write('Association Rules:', limitedCol)

# Visualize the support vs confidence for the generated rules
if not rules.empty:
    st.write('Support vs Confidence')
    fig, ax = plt.subplots()
    ax.scatter(rules['support'], rules['confidence'], alpha=0.5)
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title('Support vs Confidence')
    st.pyplot(fig)
else:
    st.write('No rules generated. Please adjust the parameters.')

# Sort rules by confidence and display top 10
sorted_rules = limitedCol.sort_values(by='confidence', ascending=False).head(10)

# Create a bar plot for the top 10 rules based on confidence
if not sorted_rules.empty:
    st.write('Top 10 Association Rules by Confidence')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_rules)), sorted_rules['confidence'], color='skyblue')
    ax.set_yticks(range(len(sorted_rules)))
    ax.set_yticklabels([f"{', '.join(list(ante))} → {', '.join(list(cons))}" for ante, cons in zip(sorted_rules['antecedents'], sorted_rules['consequents'])])
    ax.set_xlabel('Confidence')
    ax.set_title('Top 10 Association Rules by Confidence')
    ax.invert_yaxis()
    st.pyplot(fig)
else:
    st.write('No association rules to display.')
