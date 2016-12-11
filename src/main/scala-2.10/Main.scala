import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by Juan on 12/11/2016.
  */
object Main {
  def main(args: Array[String]): Unit = {
    FileUtils.cleanDirectory(new File("C:/Users/Juan/git/LearningRate/target/tmp/"))

    val conf = new SparkConf().setAppName("SVMWithSGDExample").setMaster("local")
    val sc = new SparkContext(conf)

    val filePath = "inFiles/sample_libsvm_data.txt"
    LogisticRegression.runLogisticRegression(sc, filePath)
    //SVM.runSVM(sc, filePath);
  }
}
