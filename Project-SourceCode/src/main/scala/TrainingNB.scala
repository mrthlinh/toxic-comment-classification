// Loading necessary library
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
// Machine Learning lib
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.feature.{HashingTF,IDF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Word2Vec,Word2VecModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel,DecisionTreeClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.{NaiveBayes,NaiveBayesModel}
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier,MultilayerPerceptronClassificationModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
object TrainingNB {
  def main(args: Array[String]){
    val spark = SparkSession.builder().master("local[2]")
      .appName("Comment Sentiment")
      .getOrCreate()

    // ========================================== PARAMETERS =======================================================
    // Change paremeters to build model for different label "toxic","severe_toxic","obscene","threat","insult","identity_hate"
    //    val colName = "identity_hate"
    val colName = args(0)
    // ======================================== Load Clean data =====================================================
    //Train data
    //    "../../resources/example.txt"
    val Path = new java.io.File(".").getCanonicalPath

    val train_url = Path + "/data/data_train_clean.csv"
    val data_train = spark.read.option("header", "true").option("inferSchema",true).csv(train_url)
    // Test data
    val test_url = Path + "/data/data_test_clean.csv"
    val data_test = spark.read.option("header", "true").option("inferSchema",true).csv(test_url)


    // ======================================= Feature Extraction ======================================
    // (Tokenizer -> StopWord Filter -> TFIDF / Word2Vec / CountVectorizer )
    // Reference
    // https://www.kaggle.com/danielokeeffe/text-classification-with-apache-spark

    val tokenizer = new Tokenizer().setInputCol("clean_comment").setOutputCol("words")
    val stopWord = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("stopWordFilter")

    val hashingTF = new HashingTF().setInputCol(stopWord.getOutputCol).setOutputCol("features")

    val PL = new Pipeline().setStages(Array(tokenizer, stopWord,hashingTF)).fit(data_train)
    // val pipeline = new Pipeline().setStages(Array(tokenizer, stopWord,word2Vec))
    val data_train_tran = PL.transform(data_train).select("id","features","toxic","severe_toxic","obscene","threat","insult","identity_hate")
    val data_train_pro = balanceDataset(data_train_tran,colName)

    val data_test_pro = PL.transform(data_test).select("id","features","toxic","severe_toxic","obscene","threat","insult","identity_hate")


    // ========================================== Naive Baye MODEL (dont need CV here) =============================================================
    val cv = new NaiveBayes()
      .setWeightCol("classWeightCol")
      .setLabelCol(colName)

    // ========================================== Fit the best CV model ================================================
    val cvModel = cv.fit(data_train_pro)

    // ======================================== EVALUATION on TRAIN / TEST =========================================================
    // On training data
    // val eval_data_train = cvModel.transform(data_train_pro)
    // evaluation(eval_data_train,colName)
    // On Test data
    val eval_data_test = cvModel.transform(data_test_pro)
    val output = evaluation(eval_data_test,colName)
//    val output_rdd =

    val trainRdd = spark.sparkContext.parallelize(output.toSeq).saveAsTextFile(Path+"/"+args(1))
  }

  // Description: Evaluate your model with various metrics
  // Parameter:
  // - data: DataFrame we need to evaluate the model on
  // - label_name: Name of Label Column
  // Return: Map of metrics
  // - accuracy, precision by Label, recall by Label, f1 by Label, TPR by Label, FPR by Label ( Label is in order '0', '1')
  def evaluation(data: DataFrame,label_name: String): Map[String,String] = {
    // Select prediction and label column
    val pred_label = data.select("prediction",label_name).withColumnRenamed(label_name, "label")

    // Convert to RDD
    val eval_rdd = pred_label.rdd.map{case Row(prediction:Double,label:Int) =>(prediction,label.toDouble)}

    val metric = new MulticlassMetrics(eval_rdd)

    val accuracy = Array(metric.accuracy)
    val confusionMatrix = metric.confusionMatrix
    val fMeasure = Array(metric.fMeasure(0),metric.fMeasure(1))
    val precision = Array(metric.precision(0),metric.precision(1))
    val recall = Array(metric.recall(0),metric.recall(1))
    val TPR = Array(metric.truePositiveRate(0),metric.truePositiveRate(1))
    val FPR = Array(metric.falsePositiveRate(0),metric.falsePositiveRate(1))

    // Return result
    val result = Map("acc" -> accuracy.mkString(" | "), "precision" -> precision.mkString(" | "),
      "recall" -> recall.mkString(" | "), "f1" -> fMeasure.mkString(" | "),
      "TPR" -> TPR.mkString(" | "), "FPR" -> FPR.mkString(" | "))
    return result
  }

  // Description: Add higher weight for label "1"
  // Parameter:
  // - dataset: DataFrame we need to transform
  // - label_name: Name of Label Column
  // Return: New DataFrame with "classWeightCol" Column
  def balanceDataset(dataset: DataFrame, label_name: String): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numPos = dataset.filter(dataset(label_name) === 1).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numPos).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 1.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset(label_name)))
    weightedDataset
  }
}
