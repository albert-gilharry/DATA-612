# -*- coding: utf-8 -*-
"""
DATA 612 Assignment Final Project: Instacart Recommender
Authors: 
    Albert Gilharry
"""

import findspark
findspark.init()

from pyspark.sql import  SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from flask import Flask, render_template, request
import json


app = Flask(__name__)

class Recommender:
    
    def __init__(self):
        self.department_dist = {"departments":[], "orders":[]}
        #initialize Spark connection
        # initialize a spark session
        self.spark = SparkSession.builder.getOrCreate()
        conf = self.spark.sparkContext._conf.setAll([('spark.executor.memory', '20G'),
                                                ('spark.app.name', 'Spark Updated Conf'), 
                                                ('spark.executor.cores', '4'), 
                                                ('spark.cores.max', '4'), 
                                                ('spark.driver.memory','20G'),
                                                ('spark.executor.memoryOverhead', '20G')])
        self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
        self.sc = self.spark.sparkContext
        self.loadData()
    
    def loadData(self):
        # check if data is already loaded into Spark
        try:
            self.aisles_df = self.spark.read.parquet("aisles.parquet")
            self.departments_df = self.spark.read.parquet("departments.parquet")
            self.order_products_df = self.spark.read.parquet("order_products.parquet")
            self.orders_df = self.spark.read.parquet("orders.parquet")
            self.products_df = self.spark.read.parquet("products.parquet")
    
        except:
            # if not, load data files
            self.aisles_df = self.spark.read.format('csv').options(header='true', inferSchema='true').load('data/aisles.csv')
            self.departments_df = self.spark.read.format('csv').options(header='true', inferSchema='true').load('data/departments.csv')
            order_products = self.spark.read.format('csv').options(header='true', inferSchema='true').load('data/order_products__prior.csv')
            self.order_products_df = order_products.union(self.spark.read.format('csv').options(header='true', inferSchema='true').load('data/order_products__train.csv'))
            self.orders_df = self.spark.read.format('csv').options(header='true', inferSchema='true').load('data/orders.csv')
            self.products_df = self.spark.read.format('csv').options(header='true', inferSchema='true').load('data/products.csv')
            # persist data to Spark for future access
            self.aisles_df.write.parquet("aisles.parquet", mode = "overwrite" )
            self.departments_df.write.parquet("departments.parquet")
            self.order_products_df.write.parquet("order_products.parquet")
            self.orders_df.write.parquet("orders.parquet")
            self.products_df.write.parquet("products.parquet")
        self.implicitRatings()
            
    def implicitRatings(self):
        # create users table
        self.users = self.orders_df.groupBy("user_id").count()
        self.users = self.users.drop("count")
        self.num_users = self.users.count()
        # create ratings table
        self.implicit_ratings = self.order_products_df.join(self.orders_df, self.order_products_df.order_id == self.orders_df.order_id).groupBy("user_id", "product_id").count()
        
        # train ALS model if it is not available
    def trainALS(self):
        try:
            self.model = MatrixFactorizationModel.load(self.sc, "als_final")
        except (RuntimeError, TypeError, NameError) as e:
            rank = 4
            numIterations = 20
            self.model = ALS.trainImplicit(self.implicit_ratings, rank, numIterations)
            self.model.save(self.sc, "als_final")
         
         
    # Get the recommended products to user    
    def getRecommendations( self, user_id, n):
        recommendedValue = self.item_similarity_top_k[self.item_similarity_top_k['user_id'] == user_id ]
        return list(recommendedValue["item_id"].astype(str))
	
    # Create dashboard visuals
    def getVisuals(self):
        
        # get popular aisles
        aisles_dist = {"success":True,"data":[]}
        
        try:
            self.aisles_distribution = self.spark.read.parquet("aisles_distribution.parquet")
        except:
            aisles_distribution = self.implicit_ratings.join(self.products_df, self.implicit_ratings.product_id == self.products_df.product_id)
            aisles_distribution=aisles_distribution.join(self.aisles_df,aisles_distribution.aisle_id == self.aisles_df.aisle_id).groupBy("aisle").count()
            aisles_distribution=aisles_distribution.sort("count", ascending=False)
            self.aisles_distribution=aisles_distribution.limit(10)
            self.aisles_distribution.write.parquet("aisles_distribution.parquet", mode = "overwrite")
        aisles_dist['data'] = self.aisles_distribution.toPandas().values.tolist()
        
        # get orders by hour of day
        hourly_orders = self.orders_df.groupBy("order_hour_of_day").count()
        hourly_orders = hourly_orders.sort("count")
        hour_dist = {"success":True,"hour":hourly_orders.select("order_hour_of_day").toPandas().values.tolist(), 
                     "orders":hourly_orders.select("count").toPandas().values.tolist()}
        
        # get orders by day of week
        weekly_orders = self.orders_df.groupBy("order_dow").count()
        weekly_orders = weekly_orders.sort("count")
        weekly_orders_pd = weekly_orders.toPandas()
        weekly_orders_pd = weekly_orders_pd.replace({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday", 6:"Saturday"})
        dow_dist = {"success":True, "data":weekly_orders_pd.values.tolist()}
        
        # other stats
        num_orders = self.orders_df.count()
        num_users = self.users.count()
        num_products = self.products_df.count()
        top_products = self.order_products_df.groupBy("product_id").count()
        top_products = top_products.sort("count", ascending=False)
        top_products = top_products.limit(1).join(self.products_df, top_products.product_id == self.products_df.product_id)
        top_product=top_products.collect()[0]['product_name']
        
        return {"aisles":aisles_dist,"doweek":dow_dist,"hour_of_day":hour_dist,
                "num_orders":num_orders, 
                "num_users":num_users,
                "num_products":num_products,
                "top_product":top_product}
        
    # Get a sample of users to reduce load on the interface
    def sampleUsers(self):
        self.users.sample(False, 0.1).limit(200).toPandas().values.tolist()
        return {"success":True, "data": self.users.sample(False, 0.1).limit(200).toPandas().values.tolist()}
    
    # Send recommendations along with additonal product information to the browswer 
    def recommend(self, user_id):
        self.trainALS()
        product_ids=[]
        recommendations = self.model.recommendProducts(int(user_id), 20)
        print(recommendations)
        for rec in recommendations:
            product_ids.append(rec.product)
            print(int(rec.product))
        print(product_ids)
        products = self.products_df.filter(self.products_df["product_id"].isin(product_ids))
        products = products.join(self.aisles_df,self.products_df.aisle_id == self.aisles_df.aisle_id)
        products = products.join(self.departments_df, self.products_df.department_id == self.departments_df.department_id)
        products = products.drop("aisle_id","department_id")
        
        return {"success":True, "data":products.toPandas().values.tolist()}

recommender = Recommender()

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/recommendations")
def recommendations():
    return render_template('recommendations.html')

@app.route("/getGraphics",methods=['GET'])
def getGraphics():
    graphics = recommender.getVisuals()
    return json.dumps(graphics) 

@app.route("/sampleUsers",methods=['GET'])
def sampleUsers():
    sample = recommender.sampleUsers()
    return json.dumps(sample) 

@app.route("/getRecommendations",methods=['POST'])
def getRecommendations():
    user_id = request.form['user']
    recommendations = recommender.recommend(user_id)
    return json.dumps(recommendations) 


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
