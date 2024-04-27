from flask import Flask, request, jsonify
import boto3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from flask_cors import CORS
from io import StringIO
import os

app = Flask(__name__)
CORS(app)

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

@app.route('/process-csv', methods=['POST'])
def process_csv():
    print('yo')
    request_data = request.json

    key = request_data['filename']
    bucket_name = 'kmeans-bucket-mbhasin'

    # Get the CSV file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8')

    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(StringIO(content))
    df = df.dropna()
    if len(df.columns) > 2:
        # Normalize the data
        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(df)

        # PCA only if more than 2 columns
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(df_normalized)

        print(principalComponents)

        # Elbow method to find the optimal number of clusters
        sum_of_squared_distances = []
        K = range(1, 15)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(principalComponents)
            sum_of_squared_distances.append(km.inertia_)
        optimal_k = sum_of_squared_distances.index(min(sum_of_squared_distances)) + 1
        kmeans = KMeans(n_clusters=optimal_k)
        df['cluster'] = kmeans.fit_predict(principalComponents)

        pca_df = pd.DataFrame(principalComponents, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = df['cluster'].to_numpy()

        print(pca_df.head(5))

        result_json = pca_df.to_json(orient='records')

        result_key = f'results/{key}-result.json'
        s3_client.put_object(Body=result_json, Bucket=bucket_name, Key=result_key)

        return result_json
    else:
        # Elbow method to find the optimal number of clusters
        sum_of_squared_distances = []
        K = range(1, 15)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df)
            sum_of_squared_distances.append(km.inertia_)
        optimal_k = sum_of_squared_distances.index(min(sum_of_squared_distances)) + 1


        kmeans = KMeans(n_clusters=optimal_k)
        df['Cluster'] = kmeans.fit_predict(df)

        df.rename(columns={col: f'PCA{idx + 1}' for idx, col in enumerate(df.columns[:2])}, inplace=True)

        result_json = df.to_json(orient='records')

        result_key = f'results/{key}-result.json'
        s3_client.put_object(Body=result_json, Bucket=bucket_name, Key=result_key)

        return result_json


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
