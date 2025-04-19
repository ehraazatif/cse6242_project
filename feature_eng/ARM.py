import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df_products   = pd.read_csv("../data_versions/clusters.csv")        # Description  | Cluster
df_labels     = pd.read_csv("../data_versions/cluster_labels.csv")  # Cluster | Label

# Attach human‑readable labels to each product row
df_products   = df_products.merge(df_labels, on="Cluster", how="left")  

# 2. Build “basket” (one row = one InvoiceNo, items = cluster labels)
basket = (df_products
          .groupby("InvoiceNo")["Label"]
          .apply(lambda items: sorted(set(items)))       
          .reset_index())

print(basket.head())


# 3. Convert to one‑hot encoded matrix for Apriori
# explode so each row = one (invoice, label) pair
explode = basket.explode("Label")

# one‑hot encode
basket_ohe = (explode
              .assign(value=1)
              .pivot_table(index="InvoiceNo",
                           columns="Label",
                           values="value",
                           fill_value=0)
             )


# 4. Frequent itemsets & association rules

freq_sets = apriori(basket_ohe,
                    min_support=0.01,
                    use_colnames=True)

# generate rules with minimum confidence threshold
rules = association_rules(freq_sets,
                          metric="confidence",
                          min_threshold=0.2)   # adjust as desired

# Sort by lift (or confidence) for readability
rules_sorted = rules.sort_values("lift", ascending=False)

# 5. Inspect / export
print("Top 10 rules by lift:")
print(rules_sorted[["antecedents", "consequents",
                    "support", "confidence", "lift"]].head(10))

# Save to CSV for later use / visualization
rules_sorted.to_csv("../outputs/cluster_rules.csv", index=False)