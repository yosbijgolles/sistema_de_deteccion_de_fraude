from src.predict import predict

client={'over_draft': 'no checking',
  'credit_history': 'critical/other existing credit',
  'purpose': 'education',
  'Average_Credit_Balance': 'no known savings',
  'employment': '1<=X<4',
  'personal_status': 'female div/dep/mar',
  'property_magnitude': 'car',
  'other_payment_plans': 'none',
  'housing': 'own',
  'credit_usage': 24.0,
  'current_balance': 7.564238475170491,
  'location': 3.0,
  'cc_age': 33.0,
  'existing_credits': 2.0}


response=predict(client)

print(response)
