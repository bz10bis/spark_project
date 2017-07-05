import org.apache.spark.ml.{Pipeline,PipelineModel}  
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,IndexToString}  
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier}  
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}

// ###################### Functions ######################

// On commence par transformer notre fichier CSV en parquet pour améliorer les performances de traitement.
// De plus comme notre jeux de données est deja propore cela permettra d'avoir les points la dessus
def csv_to_parquet(sqlContext: SQLContext, schema: StructType, tablename: String, filename: String) {
	//on charge notre fichier dans un dataFrame puis on l 'écrit sous format parquet'
	val dataFrame = sqlContext.read
		.format("com.databricks.spark.csv")
		// je pense que peux aussi mettre inferschema vu qu'on a les noms de colonnes a voir
		.schema(schema)
		// les colonnes sont séparées par des virgules
		.option("delimiter", ",")		
		.option("nullValues", "")
		.option("treatEmptyValuesAsNull", "true")
		.load(filename)
	dataFrame.write("dataset/parquet/" + tablename)
}

// ###################### Process ######################

//add time metrics to compare with tensorflow
val startTime: Long = System.currentTimeMillis
val schema = StructType(Array(
		StructField("satisfaction_level", DecimalType, false),
		StructField("last_evaluation", DecimalType, false),
		StructField("number_project", IntergerType, false),
		StructField("average_montly_hours", IntergerType, false),
		StructField("time_spend_company", IntergerType, false),
		StructField("Work_accident", IntergerType, false),
		StructField("left", IntergerType, false),
		StructField("promotion_last_5years", IntergerType, false),
		StructField("sales", StringType, false),
		StructField("salary", StringType, false)
	))

convert(sqlContext, schema, "hr_parquet.parquet", "/home/cloudera/Desktop/spark_project/dataset/hr_comma_sep.csv")

val dataset = sqlContext.read.load("/home/cloudera/Desktop/spark_project/dataset/hr_parquet.parquet")

//On genere nos dataframe de test et de train avec leurs labels 
val splitsDataset = dataset.randomSplit(Array{0.7,0.3}, seed = 1234L)
val trainSet = splitsDataset{0}
val test_set = splitsDataset{1}
val trainSetLabels = trainSet.col("left")
trainSet.drop(trainSet.col("left"))
val testSetLabels = trainSet.col("left")
testSet.drop(testSet.col("left"))

val labels = Seq("0", "1") 
val inputs = Array("satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "sales", "salary")
val layers = Array[Int](inputs.length, 10, labels.length)

trainSet.persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)
testSet.persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)

trainSet.count()
testSet.count()

val assembler = {new VectorAssembler().setInputsCols(inputs)}
val stringIndexer = {new StringIndexer().setInputsCol("label").fit(trainSet)}
val mlp = {new MultilayerPerceptronClassifier()
	.setLabelCol(stringIndexer.getOutputCol)
	.setFeaturesCol(assembler.getOutputCol)
	.setLayers(layers)
	.setSeed(1234L)
	.setBlockSize(128)
	.setMaxIter(1000)
	.setTol(1e-4)
}

val indexToString = {new IndexToString().setInputsCol(mlp.getPredictionCol).setLabels(StringIndexer.labels)}

val pipeline = {new Pipeline().setStages(Array(assembler, stringIndexer, mlp, indexToString))}

val model = pipeline.fit(train)

val result = model.transform(testSet)

val evaluator = {new MulticlassClassificationEvaluator()
	.setLabelCol(stringIndexer.getOutputCol)
	.setPredictionCol(mlp.getPredictionCol)
	.setMetricName("accuracy")
}

val precision = evaluator.evaluate(result)

val confusionMatrix = {
	result.select(col(stringIndexer.getInputCol), col(indexToString.getOutputCol))
	.orderBy(stringIndexer.getInputCol)
	.groupBy(stringIndexer.getInputCol)
	.pivot(indexToString.getOutputCol,labels)
	.count
}

println(s"Temps ecoule: ${System.currentTimeMillis - startTime/1000/60} minutes")
println(s"Matrice de confusion: Vertical -> attendus / Horizontal -> Prediction: ")
confusionMatrix.show
println(s"Accuracy: ${accuracy}")
