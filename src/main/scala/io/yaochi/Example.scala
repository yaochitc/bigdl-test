package io.yaochi

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.optim.{Adam, Optimizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import io.yaochi.model.ExampleModel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

object Example {
  def main(args: Array[String]): Unit = {
    val featureNum = 2
    val featureDim = 5
    val vocabSize = 20

    val batchSize = 2
    val learningRate = 0.01
    val learningRateDecay = 0.001
    val maxEpoches = 20


    val sampleRDD = getSamples
    val model = new ExampleModel[Float](featureNum, vocabSize, featureDim).build()

    val optimizzer = Optimizer(
      model,
      sampleRDD,
      BCECriterion[Float](),
      batchSize
    ).setOptimMethod(new Adam(learningRate = learningRate, learningRateDecay))

    optimizzer.optimize()
  }


  def getSamples: RDD[Sample[Float]] = {
    val sc = getSc
    val samples = Array.ofDim[Sample[Float]](10)

    val feature0 = Tensor[Float](2).zero()
    feature0.setValue(1, 0)
    feature0.setValue(2, 2)
    val label0 = Tensor[Float](1).zero()
    samples(0) = Sample[Float](feature0, label0)

    val feature1 = Tensor[Float](2).zero()
    feature1.setValue(1, 0)
    feature1.setValue(2, 3)
    val label1 = Tensor[Float](1).zero()
    samples(1) = Sample[Float](feature1, label1)

    val feature2 = Tensor[Float](2).zero()
    feature2.setValue(1, 0)
    feature2.setValue(2, 4)
    val label2 = Tensor[Float](1).zero()
    samples(2) = Sample[Float](feature2, label2)

    val feature3 = Tensor[Float](2).zero()
    val label3 = Tensor[Float](1).zero()
    feature3.setValue(1, 0)
    feature3.setValue(2, 5)
    samples(3) = Sample[Float](feature3, label3)

    val feature4 = Tensor[Float](2).zero()
    val label4 = Tensor[Float](1).zero()
    feature4.setValue(1, 0)
    feature4.setValue(2, 6)
    samples(4) = Sample[Float](feature4, label4)

    val feature5 = Tensor[Float](2).zero()
    val label5 = Tensor[Float](1).setValue(1, 1)
    feature5.setValue(1, 1)
    feature5.setValue(2, 2)
    samples(5) = Sample[Float](feature5, label5)

    val feature6 = Tensor[Float](2).zero()
    val label6 = Tensor[Float](1).setValue(1, 1)
    feature6.setValue(1, 1)
    feature6.setValue(2, 3)
    samples(6) = Sample[Float](feature6, label6)

    val feature7 = Tensor[Float](2).zero()
    val label7 = Tensor[Float](1).setValue(1, 1)
    feature7.setValue(1, 1)
    feature7.setValue(2, 4)
    samples(7) = Sample[Float](feature7, label7)

    val feature8 = Tensor[Float](2).zero()
    val label8 = Tensor[Float](1).setValue(1, 1)
    feature8.setValue(1, 1)
    feature8.setValue(2, 5)
    samples(8) = Sample[Float](feature8, label8)

    val feature9 = Tensor[Float](2).zero()
    val label9 = Tensor[Float](1).setValue(1, 1)
    feature9.setValue(1, 1)
    feature9.setValue(2, 6)
    samples(9) = Sample[Float](feature9, label9)

    sc.parallelize(samples)
  }

  def getSc: SparkContext = {
    var sparkConf = new SparkConf
    sparkConf.setMaster("local[2]")
    sparkConf.setAppName("example")
    sparkConf = Engine.createSparkConf(sparkConf)
    val sc = SparkContext.getOrCreate(sparkConf)
    Engine.init
    sc
  }
}
