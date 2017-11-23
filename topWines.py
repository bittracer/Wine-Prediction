import pandas as pd
import json
import numpy
from flask import Flask, app, render_template, request, Response
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

df = pd.read_csv("/Users/tushar/Desktop/DAL/VisualAnalytics/Project/V_bharat/wine.csv")
df['variety'] = df['variety'].str.replace(r'[^\x00-\x7f]', r'')
df.drop(df.columns[[0]], axis=1, inplace=True)
dedupped_df = df.drop_duplicates(subset='description')
varieties = dedupped_df['variety'].value_counts()


################# getMLBarChart#######################

top_wines_df = dedupped_df.loc[dedupped_df['variety'].isin(varieties.axes[0][:30])]
# our labels, as numbers.
le = LabelEncoder()
y = le.fit_transform(top_wines_df['variety'])
wine_stop_words = []
for variety in top_wines_df['variety'].unique():
    for word in variety.split(' '):
        wine_stop_words.append(word.lower())
wine_stop_words = pd.Series(data=wine_stop_words).unique()
stop_words = text.ENGLISH_STOP_WORDS.union(wine_stop_words)
vect = CountVectorizer(stop_words=stop_words)
x2 = vect.fit_transform(top_wines_df['description'])

x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0, random_state=10)

nb = MultinomialNB()
nb.fit(x_train, y_train)
################# getMLBarChart#######################


# http://127.0.0.1:8003/getTopWines/30
@app.route("/getTopWines/<int:Count>")
def getTopWines(Count):
    # ditch that unnamed row numbers column
    top_wines_df = dedupped_df.loc[dedupped_df['variety'].isin(varieties.axes[0][:(Count)])]
    topwinesCount = top_wines_df['variety'].value_counts().index.tolist()
    # this will create a json
    response = ""
    response = {'winenames': topwinesCount}
    json_str = json.dumps(response)
    return json_str


# http://127.0.0.1:8003/getWineCountries/point noir
@app.route("/getWineCountries/<wine>")
def getWineCountries(wine):

    pinotdf = dedupped_df.loc[dedupped_df['variety'] == wine]
    countries = pinotdf['country'].unique()

    list = []

    for country in countries:
        df2_with_dup = pinotdf.loc[pinotdf['country'] == country].drop_duplicates(subset='price')
        max = df2_with_dup.groupby('country')['price'].max()
        min = df2_with_dup.groupby('country')['price'].min()
        median = df2_with_dup.groupby('country')['price'].median()
        if str(country) != "nan":
            list.append({"State": country, "freq": {"low": min[country], "mid": median[country], "high": max[country]}})

    finaldf = pd.DataFrame(list)
    finaljson = finaldf.to_json(orient='records', date_format='iso')
    response = Response(response=finaljson, status=200, mimetype="application/json")
    return response


# http://127.0.0.1:8003/getCollapsibleTree/US
@app.route("/getCollapsibleTree/<Country>/test/<wineName>")
def getCollapsibleTree(Country, wineName):
    d = {"name": Country, "children": []}
    dedupped_df1 = dedupped_df.loc[df['variety'] == wineName]
    df2_with_dup = dedupped_df1.loc[df['country'] == Country]
    df2 = df2_with_dup.drop_duplicates(subset='price')

    max = df2.groupby('country')['price'].max()
    min = df2.groupby('country')['price'].min()
    median = df2.groupby('country')['price'].median()

    lowestPriceRow = df2_with_dup.loc[df2_with_dup['price'] == min[Country]]
    deduppedLowPrice = lowestPriceRow.drop_duplicates(subset='description')

    highestPriceRow = df2_with_dup.loc[df2_with_dup['price'] == max[Country]]
    deduppedHighPrice = highestPriceRow.drop_duplicates(subset='description')

    medianPriceRow = df2_with_dup.loc[df2_with_dup['price'] == median[Country]]
    deduppedMedianPrice = medianPriceRow.drop_duplicates(subset='description')
    deduppedMedianPrice = deduppedMedianPrice.drop_duplicates(subset='winery')

    winesPath = pd.DataFrame(columns=["Country", "Province", "Region1", "Winery", "type"])
    i = 0
    for index, row in deduppedLowPrice.iterrows():
        winesPath.loc[i] = [row["country"], row["province"], row["region_1"], row["winery"], "min"]
        i = i + 1

    for index, row in deduppedHighPrice.iterrows():
        winesPath.loc[i] = [row["country"], row["province"], row["region_1"], row["winery"], "max"]
        i = i + 1

    for index, row in deduppedMedianPrice.iterrows():
        winesPath.loc[i] = [row["country"], row["province"], row["region_1"], row["winery"], "median"]
        i = i + 1

    province = pd.DataFrame(df2_with_dup.province.value_counts().reset_index())
    provinceList = province["index"].tolist()
    for row in provinceList:
        index = winesPath.index[(winesPath["Province"] == row) == True].tolist()
        if (len(index) > 1):
            test1 = winesPath.loc[winesPath['Province'] == row]
            winesPath_unique = test1.drop_duplicates(subset='type')
            if (len(winesPath_unique) > 1):
                type = "many"
            else:
                type = winesPath.iloc[index[0], -1]
        elif (len(index) == 0):
            type = "none"
        else:
            type = winesPath.iloc[index[0], -1]
            # Add new element
        d["children"].append({"name": row, "type": type, "children": []})
        # Get added element
        element = d["children"][-1]
        dfprovince = df2_with_dup.loc[df2_with_dup['province'] == row]
        regions = pd.DataFrame(dfprovince.region_1.value_counts().reset_index())
        for index1, i in regions.iterrows():
            index = winesPath.index[
                ((winesPath["Province"] == row) & (winesPath["Region1"] == i['index'])) == True].tolist()
            if (len(index) > 1):
                test1 = winesPath.loc[((winesPath["Province"] == row) & (winesPath["Region1"] == i['index']))]
                winesPath_unique = test1.drop_duplicates(subset='type')
                if (len(winesPath_unique) > 1):
                    type = "many"
                else:
                    type = winesPath.iloc[index[0], -1]
            elif (len(index) == 0):
                type = "none"
            else:
                type = winesPath.iloc[index[0], -1]
            element = d["children"][-1]
            element["children"].append({"name": i['index'], "type": type, "children": []})
            element = element["children"][-1]
            dfregion = dfprovince.loc[dfprovince['region_1'] == i['index']]
            dfwinery = dfregion.winery.value_counts().reset_index()
            for index1, winery in dfwinery.iterrows():
                index = winesPath.index[((winesPath["Province"] == row) & (winesPath["Region1"] == i['index']) & (
                winesPath["Winery"] == winery[0])) == True].tolist()
                if (len(index) > 1):
                    test1 = winesPath.loc[((winesPath["Province"] == row) & (winesPath["Region1"] == i['index']) & (
                winesPath["Winery"] == winery[0]))]
                    winesPath_unique = test1.drop_duplicates(subset='type')
                    if (len(winesPath_unique) > 1):
                        type = "many"
                    else:
                        type = winesPath.iloc[index[0], -1]
                elif (len(index) == 0):
                    type = "none"
                else:
                    type = winesPath.iloc[index[0], -1]
                dwinery = dfregion.loc[dfregion['winery'] == winery[0]]
                avg_price_list = dwinery["price"]
                if (type == "min"):
                    avg_price = min[0]
                elif (type == "max"):
                    avg_price = max[0]
                elif (type == "median"):
                    avg_price = median[0]
                else:
                    avg_price = avg_price_list.mean()
                if avg_price > 0:
                    element["children"].append(
                        {"name": winery['index'], "type": type, "children": [{"name": avg_price, "value": winery[0]}]})
                else:
                    element["children"].append({"name": winery['index'], "type": type, "children": []})

    return json.dumps(d, sort_keys=False, indent=2)

# http://127.0.0.1:8003/getMLBarChart/red blueberry
@app.route("/getMLBarChart/<textData>")
def getMLBarChart(textData):

    pred = vect.transform([textData])
    prob = nb.predict_proba(pred)

    dct = {}
    for (x, y), value in numpy.ndenumerate(prob):
        wine = list(le.inverse_transform([y]))
        dct[wine[0]] = str("{:.2f}".format(value*100))

    dfBarChart = pd.DataFrame({"wine": list(dct.keys()), "probability": list(dct.values())}).sort_values(by=['probability'], ascending=False).head(5)

    json = dfBarChart.to_json(orient='records', date_format='iso')
    response = Response(response=json, status=200, mimetype="application/json")

    return response

@app.route("/")
def main():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8004)