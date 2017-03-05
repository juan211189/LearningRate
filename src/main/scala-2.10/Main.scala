import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}

/**
  * Created by Juan on 12/11/2016.
  */
object Main {
  def main(args: Array[String]): Unit = {
    FileUtils.cleanDirectory(new File("C:/Users/JuanManuel/git/LearningRate/target/tmp/"))

    val conf = new SparkConf().setAppName("LearningRate").setMaster("local")
    val sc = new SparkContext(conf)

    //val filePath = "inFiles/kddb-raw-libsvm"
    //val filePath = "inFiles/covtype.txt"
    //val filePath = "inFiles/sample_libsvm_data.txt"
    val filePath = "inFiles/a9a.txt"

    //LogisticRegression.runLogisticRegression(sc, filePath)
    SVM.runSVM(sc, filePath);
  }
}