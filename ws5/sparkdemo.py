import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: sparkdemo.py <gs://bucket/path/to/tips.csv>")
        sys.exit(1)
        
    input_path = sys.argv[1]

    # A1. Create a SparkSession named ws5-regression
    spark = SparkSession.builder \
        .appName("ws5-regression") \
        .getOrCreate()

    # A2. Read the dataset with header row and inferred column types, then .show() it
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    print("--- Initial DataFrame ---")
    df.show(5)

    # A3. Combine predictor columns 'total_bill' and 'size' into 'features'
    assembler = VectorAssembler(
        inputCols=["total_bill", "size"], 
        outputCol="features"
    )

    # A4. Split the data into 80% train / 20% test with a fixed seed
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # A5. Define LinearRegression, chain into a Pipeline, and fit on training set
    lr = LinearRegression(featuresCol="features", labelCol="tip")
    pipeline = Pipeline(stages=[assembler, lr])
    pipelineModel = pipeline.fit(train_data)

    # A6. Apply the fitted pipeline to the test set to produce predictions
    predictions = pipelineModel.transform(test_data)

    # A7. Evaluate the predictions on RMSE and R^2
    evaluator = RegressionEvaluator(labelCol="tip", predictionCol="prediction")
    
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    # A8. Pull out the fitted model and print coefficients, intercept, RMSE, and R^2
    lr_model = pipelineModel.stages[-1]
    
    print("\n" + "="*40)
    print("MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Coefficients: {lr_model.coefficients}")
    print(f"Intercept:  {lr_model.intercept}")
    print(f"RMSE:       {rmse}")
    print(f"R^2:         {r2}")
    print("="*40 + "\n")

    spark.stop()

if __name__ == "__main__":
    main()
