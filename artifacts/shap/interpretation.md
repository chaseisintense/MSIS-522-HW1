# SHAP Interpretation (RF)

This section explains which business characteristics most influenced the model's high-rating predictions.

- Note: Used RF because SHAP failed for XGB: could not convert string to float: '[5E-1]'

Top drivers and what they mean:
- **Days Open per Week**: Higher values generally push predictions away from the high-rating class.
- **Average Open Hours per Day**: Higher values generally push predictions away from the high-rating class.
- **Fast Food Category**: Higher values generally push predictions away from the high-rating class.
- **Business Currently Open**: Higher values generally push predictions toward the high-rating class.
- **Restaurants Category**: Higher values generally push predictions away from the high-rating class.
- **Food Category**: Higher values generally push predictions toward the high-rating class.
- **Active Life Category**: Higher values generally push predictions toward the high-rating class.
- **Bikeparking**: Higher values generally push predictions toward the high-rating class.

Decision-maker takeaway:
- The strongest signals come from operating intensity, customer engagement, and business type. A stakeholder can use these drivers to understand whether a business profile looks structurally strong or whether it is missing signals commonly associated with highly rated businesses.