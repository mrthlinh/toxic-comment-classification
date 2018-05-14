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
object TrainingNN {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[2]")
      .appName("Twitter Comment Sentiment")
      .getOrCreate()

    // ========================================== PARAMETERS =======================================================
    // Change paremeters to build model for different label "toxic","severe_toxic","obscene","threat","insult","identity_hate"
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

    // ======================================= Load Pre-Processing Pipeline ======================================
    val pipeline = PipelineModel.read.load("src/main/pipeline/pipeline_word2vec")
    var data_train_pro = pipeline.transform(data_train).select("id","features","toxic","severe_toxic","obscene","threat","insult","identity_hate")
    val data_test_pro = pipeline.transform(data_test).select("id","features","toxic","severe_toxic","obscene","threat","insult","identity_hate")

    // ========================================== Neural Network =========================================
    val layers = Array[Int](100,20, 10,2)

    val NN = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setSeed(1234L)
      .setMaxIter(1000)
      .setLabelCol(colName)

    val NN_Model = NN.fit(data_train_pro)

    // ======================================== EVALUATION on TRAIN / TEST =========================================================
    // On training data
    // val eval_data_train = cvModel.transform(data_train_pro)
    // evaluation(eval_data_train,colName)
    // On Test data
    val eval_data_test = NN_Model.transform(data_test_pro)

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
}
