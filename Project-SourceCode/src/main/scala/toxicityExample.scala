import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}

object toxicityExample {


  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[2]")
      .appName("Twitter Comment Sentiment")
      .getOrCreate()
    import spark.implicits._

    val currentDirectory = new java.io.File(".").getCanonicalPath
    val myFile = spark.sparkContext.textFile(currentDirectory+ "/data/example.txt")
    val message = myFile.collect().mkString("")
    val message_df = spark.sparkContext.parallelize(List(message)).toDF("clean_comment") // Convert to DF, need spark.implicits
    val toxic_type = toxicity_type(message_df,spark) // You get the Map here "toxic","severe_toxic","obscene","threat","insult","identity_hate"


    val output = spark.sparkContext.parallelize(toxic_type.toSeq).saveAsTextFile(currentDirectory+"/"+args(0))

  }
  def toxicity_type(message: DataFrame, spark: SparkSession): Map[String,Int] = {
    import spark.implicits._
    val currentDirectory = new java.io.File(".").getCanonicalPath
    // Load Pipeline of Pre-processing
    val pipeline_pre = PipelineModel.read.load( currentDirectory + "/pipeline/pipeline_word2vec")

    // Load 6 models "toxic","severe_toxic","obscene","threat","insult","identity_hate"
    val NN_model_toxic = MultilayerPerceptronClassificationModel.read.load(currentDirectory + "/pipeline/NN_toxic")
    val NN_model_severe = MultilayerPerceptronClassificationModel.read.load(currentDirectory + "/pipeline/NN_severe_toxic")
    val NN_model_obscene = MultilayerPerceptronClassificationModel.read.load(currentDirectory + "/pipeline/NN_obscene")
    val NN_model_threat = MultilayerPerceptronClassificationModel.read.load(currentDirectory + "/pipeline/NN_threat")
    val NN_model_insult = MultilayerPerceptronClassificationModel.read.load(currentDirectory + "/pipeline/NN_insult")
    val NN_model_hate = MultilayerPerceptronClassificationModel.read.load(currentDirectory + "/pipeline/NN_identity_hate")

    // Run message through the Pipeline Pre-processing
    val message_df = pipeline_pre.transform(message)

    // Run message through the 6 Neural Net model
    val result_toxic = NN_model_toxic.transform(message_df).select("prediction").map(x => x.getDouble(0)).collect()(0).toInt
    val result_severe = NN_model_severe.transform(message_df).select("prediction").map(x => x.getDouble(0)).collect()(0).toInt
    val result_obscene = NN_model_obscene.transform(message_df).select("prediction").map(x => x.getDouble(0)).collect()(0).toInt
    val result_threat = NN_model_threat.transform(message_df).select("prediction").map(x => x.getDouble(0)).collect()(0).toInt
    val result_insult = NN_model_insult.transform(message_df).select("prediction").map(x => x.getDouble(0)).collect()(0).toInt
    val result_hate = NN_model_hate.transform(message_df).select("prediction").map(x => x.getDouble(0)).collect()(0).toInt

    // Return Array
//    val resultArray = Array(result_toxic, result_severe,result_obscene,result_threat,result_insult,result_hate)

    // Return Map
    val resultMap = Map("toxic" -> result_toxic, "severe toxic" -> result_severe
                        ,"obscene" -> result_obscene, "threat" -> result_threat
                        ,"insult" -> result_insult, "identity hate" -> result_hate )
    resultMap
  }
}