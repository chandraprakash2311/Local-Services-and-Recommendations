from flask import Flask, render_template, request
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("FinalSAPDATASET.csv")
model_path = "./results"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer_path = "./results"
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

def predict_sentiment(reviews_list, model, tokenizer, max_length=128):
    all_sentiments = []
    batch_size = 32  
    for i in range(0, len(reviews_list), batch_size):
        batch_reviews = reviews_list[i:i+batch_size]
        inputs = tokenizer(batch_reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        outputs = model(**inputs)
        predicted_classes = torch.argmax(outputs.logits, dim=1).tolist()
        all_sentiments.extend(predicted_classes)
    return all_sentiments


@app.route('/')
def index():
    return render_template('newui.html')

@app.route('/directions')
def directions():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    return render_template('directions.html', lat=lat, lon=lon)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_location = request.form["location"]
    user_cuisine = request.form["cuisine"]
    street, city = [s.lower().strip() for s in user_location.split(",")]

    matching_hotels = data[
        (data['city'].str.lower().str.strip() == city) &
        (data['locality'].str.lower().str.strip() == street)
    ]

    resultant_hotels = []
    resultant_reviews = []
    hotel_prices = []
    hotel_address = []
    hotel_latitudes = []
    hotel_longitudes = []
    predicted_sentiment = []    
    hotel_timings=[]
    print("Filtering matching hotels...")
    print(matching_hotels)
    for index, row in matching_hotels.iterrows():
        cuisines = [cuisine.lower().strip() for cuisine in row['cuisine'].split(',')]
        if user_cuisine.lower().strip() in cuisines:
            resultant_hotels.append(row['name'])
            resultant_reviews.extend(row['Review'].split("||"))
            hotel_prices.append(row["cost"])
            hotel_address.append(row["address"])
            hotel_latitudes.append(row["latitude"])
            hotel_longitudes.append(row["longitude"])
            hotel_timings.append(row["timings"])

    print("Resultant hotels: ", resultant_hotels)
    if resultant_reviews:
        print("Predicting sentiment...")
        predicted = predict_sentiment(resultant_reviews, model, tokenizer)
        print("Predicted: ",predicted)
        print("Predicted Sentiments:", predicted)  
        
        hotel_sentiments = []
        current_index = 0
        for reviews in resultant_reviews:
            review_count = len(reviews.split("||"))
            hotel_predicted = predicted[current_index:current_index + review_count]
            current_index += review_count
            hotel_sentiments.append(hotel_predicted)

        print("Predicted Sentiment List:", hotel_sentiments)  

        for hotel, sentiments in zip(resultant_hotels, hotel_sentiments):
            if sentiments.count(1) > sentiments.count(0):
                predicted_sentiment.append(1)
            else:
                predicted_sentiment.append(0)

        recommended_hotels = [
            {
                "name": resultant_hotels[i],
                "price": hotel_prices[i],
                "address": hotel_address[i],
                "timings":hotel_timings[i],
                "latitude": hotel_latitudes[i],
                "longitude": hotel_longitudes[i],
            }
            for i in range(len(predicted_sentiment)) if predicted_sentiment[i] == 1
        ]
        print("Recommended Hotels:", recommended_hotels)  
    else:
        recommended_hotels = []
    recommended_hotels = sorted(recommended_hotels, key=lambda x: x["price"])

    hotel_names = [hotel['name'] for hotel in recommended_hotels]
    addresses = [hotel["address"] for hotel in recommended_hotels]
    safety_reviews = [data.loc[(data['name'] == name) & (data['address'] == addr), 'safety'].values[0] 
                    for name, addr in zip(hotel_names, addresses)]
    print(safety_reviews)

    def predictsafety_sentiment(res_safetyreview, model, tokenizer, max_length=128):
        resPred_safety = []
        for user_input in res_safetyreview:
            inputs = tokenizer(user_input, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            resPred_safety.append(predicted_class)
        return resPred_safety

    # Load the trained model
    modelsafety_path = "./safetyresults"  
    model_safety = DistilBertForSequenceClassification.from_pretrained(modelsafety_path)
    tokenizersafety_path = "./safetyresults"  
    tokenizer_safety = DistilBertTokenizer.from_pretrained(tokenizersafety_path)

    # Predict sentiment
    resPred_safety = predictsafety_sentiment(safety_reviews, model_safety, tokenizer_safety)
    print(resPred_safety)

    result_hotels = []
    for hotel, safety in zip(recommended_hotels, resPred_safety):
        if safety == 1:
            result_hotels.append(hotel)
    for i in result_hotels:
        print(i["name"])
    print(len(result_hotels))
    return render_template('display.html', recommendations=result_hotels)

if __name__ == '__main__':
    app.run(debug=True)
